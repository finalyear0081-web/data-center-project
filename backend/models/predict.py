"""
Inference Module for Temperature Prediction
Supports multi-schema forecasting:
- 24-Hour Horizon for HazardNet Operational Telemetry
- 1-Hour Horizon (30 steps at 2m) for AAU Experimental Data with Experiment Selection
"""
import os
import sys
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Union, Optional, List

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ.setdefault("KERAS_HOME", os.path.join(BASE_DIR, ".keras"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import tensorflow as tf

from backend.config import Config
from backend.preprocessing.normalization import DataCenterScaler
from backend.preprocessing.validation import detect_schema, detect_sampling_interval_and_horizon
from backend.utils.datetime_utils import format_forecast_timestamp

class TemperaturePredictor:
    def __init__(self, model_path: str = Config.MODEL_PATH, scaler_path: str = Config.SCALER_PATH):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self._model = None
        self._scaler = None
        
    def reload(self):
        """Invalidates cached in-memory model and scaler so they are reloaded on next inference."""
        self._model = None
        self._scaler = None

    def _load_resources(self):
        if self._scaler is None:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler not found at {self.scaler_path}. Please train or upload model.")
            self._scaler = DataCenterScaler.load(self.scaler_path)
            
        if self._model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}. Please train or upload model.")
            self._model = tf.keras.models.load_model(self.model_path, compile=False)

    def predict_24h(self, recent_data: pd.DataFrame, 
                    sampling_interval_minutes: int = 5,
                    experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates temperature forecast given recent historical observations.
        For AAU: Generates validated 1-hour forecast (30 steps at 2m) for selected experiment.
        For HazardNet: Generates 24-hour forecast (144 steps at 10m).
        """
        self._load_resources()
        schema_type, schema_name = detect_schema(recent_data)
        
        # Adaptive window sizes from loaded model
        model_in_window = self._model.input_shape[1] if (self._model and self._model.input_shape) else None
        model_out_steps = self._model.output_shape[1] if (self._model and self._model.output_shape) else None

        if schema_type == "AAU_EXPERIMENTAL":
            available_exps = recent_data["experiment_id"].dropna().unique().tolist()
            if not experiment_id or experiment_id not in available_exps:
                experiment_id = available_exps[0] if available_exps else "BaselineTest_100percent"

            exp_data = recent_data[recent_data["experiment_id"] == experiment_id].copy()
            exp_data = exp_data.sort_values(by="elapsed_time_minutes").reset_index(drop=True)
            
            input_window = model_in_window or 60  # default 2 hours
            horizon_steps = model_out_steps or 30  # default 1 hour
            interval_mins = 2
            horizon_hours = round((horizon_steps * interval_mins) / 60.0, 1)
            
            if len(exp_data) < input_window:
                raise ValueError(
                    f"Selected experiment '{experiment_id}' has {len(exp_data)} records, "
                    f"minimum {input_window} required for input history."
                )

            window_df = exp_data.iloc[-input_window:].copy()
            last_elapsed = float(window_df["elapsed_time_minutes"].iloc[-1])
            
            scaled_features = self._scaler.transform_features(window_df)
            input_tensor = np.expand_dims(scaled_features, axis=0).astype(np.float32)
            
            scaled_preds = self._model.predict(input_tensor, verbose=0)
            celsius_preds = self._scaler.inverse_transform_target(scaled_preds)[0][:horizon_steps]
            
            # Check if exp_data contains timestamp series
            has_valid_dt = False
            last_ts_val = None
            ts_cols = [c for c in ["timestamp", "datetime", "date_time", "time"] if c in exp_data.columns]
            if ts_cols:
                valid_ts = pd.to_datetime(exp_data[ts_cols[0]], errors="coerce").dropna()
                if not valid_ts.empty:
                    last_ts_val = valid_ts.iloc[-1]
                    has_valid_dt = True

            if has_valid_dt and last_ts_val is not None:
                future_timestamps = [
                    format_forecast_timestamp(last_ts_val + timedelta(minutes=interval_mins * (i + 1)))
                    for i in range(len(celsius_preds))
                ]
            else:
                future_timestamps = [
                    f"{int(last_elapsed + (i + 1) * interval_mins)} min (+{(i + 1) * interval_mins}m)"
                    for i in range(len(celsius_preds))
                ]
            pred_list = [round(float(val), 2) for val in celsius_preds]
            
            return {
                "schema_type": "AAU_EXPERIMENTAL",
                "schema_name": "AAU Experimental Data",
                "experiment_id": experiment_id,
                "available_experiments": available_exps,
                "sampling_interval_minutes": interval_mins,
                "forecast_horizon_hours": horizon_hours,
                "input_window_steps": input_window,
                "prediction_count": len(pred_list),
                "forecast_temperatures": pred_list,
                "forecast_timestamps": future_timestamps,
                "max_temperature": round(float(np.max(pred_list)), 2),
                "min_temperature": round(float(np.min(pred_list)), 2),
                "avg_temperature": round(float(np.mean(pred_list)), 2),
                "generated_at": datetime.now(timezone.utc).isoformat()
            }

        else:
            # Universal / Continuous Telemetry Schema
            input_window = self._model.input_shape[1] if self._model.input_shape else 144
            
            interval_mins = sampling_interval_minutes
            if schema_type == "HAZARDNET_OPERATIONAL":
                interval_mins, _ = detect_sampling_interval_and_horizon(recent_data, schema="HAZARDNET_OPERATIONAL")
                schema_label = "HazardNet Operational Telemetry"
            else:
                schema_label = "Universal Data Center Telemetry"
                detected_mins, _ = detect_sampling_interval_and_horizon(recent_data, schema=schema_type)
                if sampling_interval_minutes is None or sampling_interval_minutes == 2:
                    interval_mins = detected_mins
                
            # For 24-hour forecast: 1440 minutes / interval_mins steps
            max_24h_steps = max(1, int(round(1440.0 / interval_mins)))
            model_steps = self._model.output_shape[1] if self._model.output_shape else 144
            horizon_steps = min(model_steps, max_24h_steps)
                
            if len(recent_data) < input_window:
                raise ValueError(f"Expected at least {input_window} historical readings, received {len(recent_data)}.")
                
            window_df = recent_data.iloc[-input_window:].copy()
            
            scaled_features = self._scaler.transform_features(window_df)
            input_tensor = np.expand_dims(scaled_features, axis=0).astype(np.float32)
            
            scaled_preds = self._model.predict(input_tensor, verbose=0)
            celsius_preds = self._scaler.inverse_transform_target(scaled_preds)[0][:horizon_steps]
            
            # Ensure full 24-hour horizon matches target steps (e.g. 720 steps at 2m interval)
            if len(celsius_preds) != max_24h_steps and max_24h_steps > 0:
                x_orig = np.linspace(0, 1, len(celsius_preds))
                x_target = np.linspace(0, 1, max_24h_steps)
                celsius_preds = np.interp(x_target, x_orig, celsius_preds)
            future_timestamps = []
            has_valid_dt = False
            last_ts_val = None
            
            ts_cols = [c for c in ["timestamp", "datetime", "date_time", "time"] if c in recent_data.columns]
            if ts_cols:
                valid_ts = pd.to_datetime(recent_data[ts_cols[0]], errors="coerce").dropna()
                if not valid_ts.empty:
                    last_ts_val = valid_ts.iloc[-1]
                    has_valid_dt = True
            elif "timestamp" in window_df.columns:
                valid_ts = pd.to_datetime(window_df["timestamp"], errors="coerce").dropna()
                if not valid_ts.empty:
                    last_ts_val = valid_ts.iloc[-1]
                    has_valid_dt = True
                    
            if has_valid_dt and last_ts_val is not None:
                future_timestamps = [
                    format_forecast_timestamp(last_ts_val + timedelta(minutes=interval_mins * (i + 1)))
                    for i in range(len(celsius_preds))
                ]
            else:
                future_timestamps = [
                    f"+{(i + 1) * interval_mins}m"
                    for i in range(len(celsius_preds))
                ]
                
            pred_list = [round(float(val), 2) for val in celsius_preds]
            horizon_hours = round((len(pred_list) * interval_mins) / 60.0, 1)
            node_id_val = str(window_df["node_id"].iloc[-1]) if "node_id" in window_df.columns else "universal_sensor_node"
            
            return {
                "schema_type": schema_type,
                "schema_name": schema_label,
                "experiment_id": node_id_val,
                "available_experiments": [node_id_val],
                "sampling_interval_minutes": interval_mins,
                "forecast_horizon_hours": horizon_hours,
                "input_window_steps": input_window,
                "prediction_count": len(pred_list),
                "forecast_temperatures": pred_list,
                "forecast_timestamps": future_timestamps,
                "max_temperature": round(float(np.max(pred_list)), 2),
                "min_temperature": round(float(np.min(pred_list)), 2),
                "avg_temperature": round(float(np.mean(pred_list)), 2),
                "generated_at": datetime.now(timezone.utc).isoformat()
            }

_predictor = None

def get_predictor() -> TemperaturePredictor:
    global _predictor
    if _predictor is None:
        _predictor = TemperaturePredictor()
    return _predictor
