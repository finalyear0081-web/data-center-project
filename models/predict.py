"""
Inference Module for Real-Time and Batch Temperature Forecasting
Loads trained LSTM model and scaler, generates 24-hour (288-step) temperature forecasts.
"""
import os
import sys
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Union

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("KERAS_HOME", os.path.abspath(".keras"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import tensorflow as tf

from preprocessing.normalization import DataCenterScaler
from preprocessing.data_loader import NUMERIC_COLUMNS
from backend.utils.datetime_utils import format_forecast_timestamp

MODEL_PATH = "models/saved/lstm_temperature_model.keras"
SCALER_PATH = "models/saved/scaler.pkl"

class TemperaturePredictor:
    """
    Manages model loading and inference for data center temperature forecasting.
    """
    def __init__(self, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self._model = None
        self._scaler = None
        
    def _load_resources(self):
        if self._scaler is None:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found at {self.scaler_path}. Please train model first.")
            self._scaler = DataCenterScaler.load(self.scaler_path)
            
        if self._model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}. Please train model first.")
            self._model = tf.keras.models.load_model(self.model_path)
            
    def predict(self, recent_data: Union[pd.DataFrame, np.ndarray], 
                last_timestamp: datetime = None) -> Dict[str, Any]:
        """
        Generates 24-hour (288 reading) temperature forecast given recent 288 readings.
        
        Args:
            recent_data: pd.DataFrame with all numeric columns or (288, 9) ndarray.
            last_timestamp: datetime of the last observation.
            
        Returns:
            dict containing:
                - forecast_values: List[float] (288 temperatures in °C)
                - forecast_timestamps: List[str] (ISO timestamps)
                - max_temperature: float
                - min_temperature: float
                - avg_temperature: float
                - horizon_steps: int (288)
        """
        self._load_resources()
        
        if isinstance(recent_data, pd.DataFrame):
            if "timestamp" in recent_data.columns and last_timestamp is None:
                last_timestamp = pd.to_datetime(recent_data["timestamp"].iloc[-1])
            # Ensure only the last 288 records are used
            if len(recent_data) > 288:
                recent_data = recent_data.iloc[-288:]
            scaled_features = self._scaler.transform_features(recent_data)
        else:
            scaled_features = recent_data
            
        if len(scaled_features) != 288:
            raise ValueError(f"Expected 288 historical time-steps, received {len(scaled_features)}.")
            
        # Shape: (1, 288, num_features)
        input_tensor = np.expand_dims(scaled_features, axis=0).astype(np.float32)
        
        # Inference
        scaled_preds = self._model.predict(input_tensor, verbose=0) # shape (1, 288)
        celsius_preds = self._scaler.inverse_transform_target(scaled_preds)[0] # shape (288,)
        
        # Timestamps
        if last_timestamp is None:
            if isinstance(recent_data, pd.DataFrame):
                ts_cols = [c for c in ["timestamp", "datetime", "date_time"] if c in recent_data.columns]
                if ts_cols:
                    valid_ts = pd.to_datetime(recent_data[ts_cols[0]], errors="coerce").dropna()
                    if not valid_ts.empty:
                        last_timestamp = valid_ts.iloc[-1]
            if last_timestamp is None:
                last_timestamp = datetime.now()
            
        future_timestamps = [
            format_forecast_timestamp(last_timestamp + timedelta(minutes=5 * (i + 1)))
            for i in range(len(celsius_preds))
        ]
        
        pred_list = [round(float(val), 2) for val in celsius_preds]
        
        return {
            "forecast_values": pred_list,
            "forecast_timestamps": future_timestamps,
            "max_temperature": round(float(np.max(pred_list)), 2),
            "min_temperature": round(float(np.min(pred_list)), 2),
            "avg_temperature": round(float(np.mean(pred_list)), 2),
            "horizon_steps": len(pred_list),
            "generated_at": datetime.utcnow().isoformat()
        }

# Global singleton predictor
_predictor_instance = None

def get_predictor() -> TemperaturePredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = TemperaturePredictor()
    return _predictor_instance
