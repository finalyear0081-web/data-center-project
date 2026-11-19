"""
Pipeline Orchestration, Background Job Tracking, and Forecast Management Service
Supports multi-schema forecasting (HazardNet and AAU Experimental Data).
"""
import os
import io
import csv
import json
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np

from backend.config import Config
from backend.preprocessing.canonical_schema import CanonicalSchemaError
from backend.preprocessing.data_loader import load_and_parse_csv
from backend.preprocessing.cleaning import clean_and_repair_dataset
from backend.preprocessing.validation import detect_schema, detect_sampling_interval_and_horizon
from backend.models.predict import get_predictor
from backend.models.train import train_lstm_pipeline
from backend.models.evaluation import run_model_evaluation
from backend.cooling.controller import get_cooling_controller, determine_cooling_decision
from backend.cooling.alerts import get_alert_manager
from backend.services.dataset_service import get_dataset_service
from backend.utils.datetime_utils import format_forecast_timestamp

LATEST_FORECAST_CACHE = os.path.join(Config.PREDICTIONS_DIR, "latest_forecast.json")

def resample_forecast_15min(
    temps: List[float],
    timestamps: List[str],
    on_th: float = 28.5,
    off_th: float = 26.5,
    initial_chiller_state: str = "OFF"
) -> List[Dict[str, Any]]:
    """
    Resamples a 720-step (2-minute) forecast trajectory into exactly 96 records
    at 15-minute intervals covering the full 24-hour horizon (96 * 15m = 1440m).

    Mapping:
    - Step index k in [0, 95] (total 96 records).
    - Minute offset: k * 15 (0m for k=0, up to 1425m / 23h 45m for k=95).
    - Fractional step index in 2-minute sequence: (k * 15) / 2 = 7.5 * k.
    - Linear interpolation between floor(7.5*k) and ceil(7.5*k).
    - Sequential single-source hysteresis evaluates chiller state and thermal condition.
    """
    if not temps:
        return []

    num_records = 96
    results = []
    sim_state = "ON" if str(initial_chiller_state).strip().upper() in ["ON", "RUNNING"] else "OFF"

    # Determine base datetime from first timestamp
    dt0 = None
    if timestamps:
        try:
            dt0 = pd.to_datetime(timestamps[0])
        except Exception:
            dt0 = None

    for k in range(num_records):
        idx = 7.5 * k
        i_low = int(idx)
        i_high = min(i_low + 1, len(temps) - 1)
        alpha = idx - i_low

        if i_low >= len(temps):
            temp_k = float(temps[-1])
        elif i_low == i_high or alpha == 0.0:
            temp_k = float(temps[i_low])
        else:
            temp_k = (1.0 - alpha) * float(temps[i_low]) + alpha * float(temps[i_high])

        temp_k = round(temp_k, 2)

        # Compute timestamp
        if dt0 is not None:
            ts_k = format_forecast_timestamp(dt0 + pd.Timedelta(minutes=15 * k))
        elif timestamps and (timestamps[0].endswith("m") or "min" in timestamps[0]):
            ts_k = f"+{k * 15}m"
        elif timestamps:
            ts_k = timestamps[min(int(round(idx)), len(timestamps) - 1)]
        else:
            ts_k = f"+{k * 15}m"

        # Centralized hysteresis evaluation
        dec = determine_cooling_decision(
            temp_k,
            previous_state=sim_state,
            on_threshold=on_th,
            off_threshold=off_th
        )
        sim_state = dec["chiller_state"]

        results.append({
            "index": k + 1,
            "serial_number": k + 1,
            "timestamp": ts_k,
            "forecast_timestamp": ts_k,
            "temperature": temp_k,
            "predicted_inlet_temperature_celsius": temp_k,
            "chiller_state": dec["chiller_state"],
            "chiller_decision": dec["legacy_state"],
            "is_running": dec["is_running"],
            "temperature_status": dec["temperature_status"],
            "condition": dec["temperature_status"],
            "reason": dec["reason"]
        })

    return results

class PredictionService:
    def __init__(self):
        self.predictor = get_predictor()
        self.cooling_ctrl = get_cooling_controller()
        self.alert_mgr = get_alert_manager()
        self.dataset_svc = get_dataset_service()
        self._is_job_running = False
        self._job_status = self._load_job_status()

    def _load_job_status(self) -> Dict[str, Any]:
        if os.path.exists(Config.JOB_STATUS_PATH):
            try:
                with open(Config.JOB_STATUS_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "state": "IDLE",
            "progress": 0.0,
            "stage_message": "System ready. Awaiting dataset ingestion.",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "error": None
        }

    def _save_job_status(self):
        os.makedirs(Config.STORAGE_DIR, exist_ok=True)
        try:
            with open(Config.JOB_STATUS_PATH, "w", encoding="utf-8") as f:
                json.dump(self._job_status, f, indent=2)
        except Exception:
            pass

    def _update_job_status(self, state: str, progress: float, stage_message: str, error: Optional[str] = None):
        self._job_status = {
            "state": state,
            "progress": round(progress, 2),
            "stage_message": stage_message,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "error": error
        }
        self._save_job_status()

    def get_job_status(self) -> Dict[str, Any]:
        return self._job_status

    def start_pipeline_job(self, dataset_path: str, force_retrain: bool = False, 
                           experiment_id: Optional[str] = None,
                           architecture: str = "RESIDUAL_LSTM",
                           epochs: int = 10) -> Dict[str, Any]:
        """
        Launches end-to-end dataset processing, retraining, and prediction in a background worker thread.
        """
        if self._is_job_running:
            return {"accepted": False, "message": "A pipeline job is already in progress. Please wait for completion."}

        def worker():
            self._is_job_running = True
            try:
                self._update_job_status("VALIDATING", 0.10, "Loading and validating dataset...")
                raw_df = load_and_parse_csv(dataset_path)
                clean_df, _ = clean_and_repair_dataset(raw_df)

                self._update_job_status("PREPROCESSING", 0.25, "Preprocessing telemetry readings and evaluating suitability...")
                schema_type, schema_name = detect_schema(clean_df)
                interval_mins, horizon_steps = detect_sampling_interval_and_horizon(clean_df, schema=schema_type)
                
                # Check whether retraining is required
                hist_path = os.path.join(Config.MODELS_DIR, "loss_history.json")
                trained_source = None
                trained_schema = None
                if os.path.exists(hist_path):
                    try:
                        with open(hist_path, "r", encoding="utf-8") as f:
                            h = json.load(f)
                            trained_source = h.get("dataset_source")
                            trained_schema = h.get("schema_type")
                    except Exception:
                        pass

                min_required = 120 if schema_type == "AAU_EXPERIMENTAL" else 288
                need_train = (force_retrain 
                              or not os.path.exists(Config.MODEL_PATH) 
                              or not os.path.exists(Config.SCALER_PATH)
                              or trained_schema != schema_type
                              or trained_source != os.path.basename(dataset_path))
                              
                if need_train and len(clean_df) >= min_required:
                    self._update_job_status(
                        "TRAINING", 0.40, 
                        f"Training {architecture} for {schema_name} ({horizon_steps} steps, {epochs} epochs)..."
                    )
                    train_lstm_pipeline(dataset_path, epochs=epochs, batch_size=64, architecture=architecture)
                    try:
                        run_model_evaluation(dataset_path)
                    except Exception as eval_err:
                        print(f"Evaluation notice: {eval_err}")

                # Refresh predictor cache
                self.predictor.reload()

                self._update_job_status("PREDICTING", 0.75, f"Generating temperature forecast for {schema_name}...")
                pred_result = self.predictor.predict_24h(
                    clean_df, 
                    sampling_interval_minutes=interval_mins,
                    experiment_id=experiment_id
                )

                self._update_job_status("COOLING_EVALUATION", 0.90, "Evaluating predictive cooling hysteresis controller...")
                
                if schema_type == "AAU_EXPERIMENTAL":
                    active_exp = pred_result.get("experiment_id")
                    exp_df = clean_df[clean_df["experiment_id"] == active_exp]
                    current_inlet = float(exp_df["inlet_temperature"].iloc[-1])
                    current_outlet = float(exp_df["outlet_temperature"].iloc[-1]) if "outlet_temperature" in exp_df.columns else None
                    current_power = float(exp_df["it_power_consumption"].iloc[-1]) if "it_power_consumption" in exp_df.columns else None
                    current_fac_power = float(exp_df["facility_power"].iloc[-1]) if "facility_power" in exp_df.columns else None
                    node_label = f"AAU Exp: {active_exp}"
                    ts_cols = [c for c in ["timestamp", "datetime", "date_time", "time"] if c in exp_df.columns]
                    if ts_cols and pd.to_datetime(exp_df[ts_cols[0]], errors="coerce").dropna().shape[0] > 0:
                        last_val = pd.to_datetime(exp_df[ts_cols[0]], errors="coerce").dropna().iloc[-1]
                        reading_time = format_forecast_timestamp(last_val)
                    else:
                        reading_time = f"{int(exp_df['elapsed_time_minutes'].iloc[-1])} min"
                else:
                    current_inlet = float(clean_df["inlet_temperature"].iloc[-1])
                    current_outlet = float(clean_df["outlet_temperature"].iloc[-1]) if "outlet_temperature" in clean_df.columns else None
                    current_power = float(clean_df["it_power_consumption"].iloc[-1]) if "it_power_consumption" in clean_df.columns else None
                    current_fac_power = None
                    node_label = str(clean_df["node_id"].iloc[-1]) if "node_id" in clean_df.columns else "marconi_a2_mean_aggregate"
                    ts_cols = [c for c in ["timestamp", "datetime", "date_time", "time"] if c in clean_df.columns]
                    if ts_cols and pd.to_datetime(clean_df[ts_cols[0]], errors="coerce").dropna().shape[0] > 0:
                        last_val = pd.to_datetime(clean_df[ts_cols[0]], errors="coerce").dropna().iloc[-1]
                        reading_time = format_forecast_timestamp(last_val)
                    else:
                        reading_time = str(clean_df["timestamp"].iloc[-1]) if "timestamp" in clean_df.columns else "Unknown"

                cooling_result = self.cooling_ctrl.evaluate_forecast(
                    forecast_temps=pred_result["forecast_temperatures"],
                    forecast_timestamps=pred_result["forecast_timestamps"],
                    current_temp=current_inlet,
                    sampling_interval_minutes=interval_mins
                )

                # Cache combined forecast results
                combined = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "dataset_source": os.path.basename(dataset_path),
                    "schema_type": schema_type,
                    "schema_name": schema_name,
                    "telemetry_current": {
                        "inlet_temperature": current_inlet,
                        "outlet_temperature": current_outlet,
                        "cpu_usage": float(clean_df["cpu_usage"].iloc[-1]) if ("cpu_usage" in clean_df.columns and clean_df["cpu_usage"].notna().any()) else None,
                        "it_power_consumption": current_power,
                        "facility_power": current_fac_power,
                        "cooling_power_consumption": float(clean_df["cooling_power_consumption"].iloc[-1]) if ("cooling_power_consumption" in clean_df.columns and clean_df["cooling_power_consumption"].notna().any()) else None,
                        "node_id": node_label,
                        "last_reading_time": reading_time
                    },
                    "prediction": pred_result,
                    "cooling": cooling_result
                }
                
                os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)
                with open(LATEST_FORECAST_CACHE, "w", encoding="utf-8") as f:
                    json.dump(combined, f, indent=2)

                self._update_job_status("COMPLETED", 1.0, f"Pipeline completed successfully for {schema_name}. Dashboard updated.")
            except CanonicalSchemaError as cse:
                err_msg = str(cse)
                self._update_job_status("FAILED", 1.0, f"Validation Failed: {err_msg}", error=err_msg)
            except KeyError as ke:
                key_name = str(ke).strip("'\" ")
                err_msg = f"Missing required column '{key_name}'. Please verify your CSV columns or configure column remapping."
                self._update_job_status("FAILED", 1.0, f"Processing failed: {err_msg}", error=err_msg)
            except Exception as e:
                self._update_job_status("FAILED", 1.0, f"Processing failed: {str(e)}", error=str(e))
            finally:
                self._is_job_running = False

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return {"accepted": True, "message": "Automated dataset processing pipeline started."}

    def invalidate_forecast_cache(self):
        """Invalidates any cached forecast file to ensure historical dates are never reused."""
        if os.path.exists(LATEST_FORECAST_CACHE):
            try:
                os.remove(LATEST_FORECAST_CACHE)
            except Exception:
                pass

    def recalculate_cooling_with_thresholds(self, on_th: float, off_th: float) -> Dict[str, Any]:
        """
        Re-evaluates active forecast with new threshold setpoints, updates
        the cooling controller, resamples the 15-minute presentation forecast,
        and synchronizes storage/predictions/latest_forecast.json.
        """
        data = {}
        if os.path.exists(LATEST_FORECAST_CACHE):
            try:
                with open(LATEST_FORECAST_CACHE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        if not data or not data.get("prediction"):
            return self.get_latest_forecast(force_refresh=True)

        pred = data.get("prediction", {})
        temps = pred.get("forecast_temperatures", [])
        times = pred.get("forecast_timestamps", [])
        telemetry = data.get("telemetry_current", {})
        curr_inlet = telemetry.get("inlet_temperature", temps[0] if temps else 24.0)
        interval_mins = pred.get("sampling_interval_minutes", 2)

        cooling_result = self.cooling_ctrl.evaluate_forecast(
            temps,
            times,
            current_temp=curr_inlet,
            sampling_interval_minutes=interval_mins
        )

        f_15 = resample_forecast_15min(
            temps,
            times,
            on_th=on_th,
            off_th=off_th,
            initial_chiller_state="ON" if self.cooling_ctrl.is_running else "OFF"
        )

        data["cooling"] = cooling_result
        data["forecast_15min"] = f_15
        data["prediction"]["forecast_15min"] = f_15
        data["updated_at"] = datetime.now(timezone.utc).isoformat()

        try:
            os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)
            with open(LATEST_FORECAST_CACHE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

        return data

    def get_latest_forecast(self, force_refresh: bool = False, experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """Returns cached forecast or generates on-the-fly from active dataset."""
        dataset_path = self.dataset_svc.get_current_dataset_path()
        dataset_filename = os.path.basename(dataset_path) if dataset_path else ""

        th = self.cooling_ctrl.threshold_mgr.get_thresholds()
        on_th = th.on_threshold
        off_th = th.off_threshold

        if not force_refresh and os.path.exists(LATEST_FORECAST_CACHE):
            try:
                with open(LATEST_FORECAST_CACHE, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    cached_src = cached.get("dataset_source")
                    # Invalidate cache if active dataset changed
                    if cached_src == dataset_filename:
                        # Verify cached timestamps are in 12-hour AM/PM format
                        cached_ts = cached.get("prediction", {}).get("forecast_timestamps", [])
                        if cached_ts and ("AM" in cached_ts[0] or "PM" in cached_ts[0]):
                            if not experiment_id or cached.get("prediction", {}).get("experiment_id") == experiment_id:
                                # Verify threshold and 15-minute alignment
                                cached_cooling = cached.get("cooling", {})
                                if (
                                    cached_cooling.get("on_threshold") == on_th and
                                    cached_cooling.get("off_threshold") == off_th and
                                    "forecast_15min" in cached and
                                    len(cached.get("forecast_15min", [])) == 96
                                ):
                                    return cached
                                else:
                                    return self.recalculate_cooling_with_thresholds(on_th, off_th)
            except Exception:
                pass
                
        # Generate synchronously
        if dataset_path and os.path.exists(dataset_path):
            df = load_and_parse_csv(dataset_path)
            clean_df, _ = clean_and_repair_dataset(df)
            schema_type, schema_name = detect_schema(clean_df)
            interval_mins, _ = detect_sampling_interval_and_horizon(clean_df, schema=schema_type)
            
            pred_result = self.predictor.predict_24h(
                clean_df, 
                sampling_interval_minutes=interval_mins,
                experiment_id=experiment_id
            )
            
            if schema_type == "AAU_EXPERIMENTAL":
                active_exp = pred_result.get("experiment_id")
                exp_df = clean_df[clean_df["experiment_id"] == active_exp]
                current_inlet = float(exp_df["inlet_temperature"].iloc[-1])
                current_outlet = float(exp_df["outlet_temperature"].iloc[-1]) if "outlet_temperature" in exp_df.columns else None
                current_power = float(exp_df["it_power_consumption"].iloc[-1]) if "it_power_consumption" in exp_df.columns else None
                current_fac = float(exp_df["facility_power"].iloc[-1]) if "facility_power" in exp_df.columns else None
                node_label = f"AAU Exp: {active_exp}"
                
                # Check for valid timestamp
                ts_cols = [c for c in ["timestamp", "datetime", "date_time", "time"] if c in exp_df.columns]
                if ts_cols and pd.to_datetime(exp_df[ts_cols[0]], errors="coerce").dropna().shape[0] > 0:
                    last_val = pd.to_datetime(exp_df[ts_cols[0]], errors="coerce").dropna().iloc[-1]
                    reading_time = format_forecast_timestamp(last_val)
                else:
                    reading_time = f"{int(exp_df['elapsed_time_minutes'].iloc[-1])} min"
            else:
                current_inlet = float(clean_df["inlet_temperature"].iloc[-1])
                current_outlet = float(clean_df["outlet_temperature"].iloc[-1]) if "outlet_temperature" in clean_df.columns else None
                current_power = float(clean_df["it_power_consumption"].iloc[-1]) if "it_power_consumption" in clean_df.columns else None
                current_fac = None
                node_label = str(clean_df["node_id"].iloc[-1]) if "node_id" in clean_df.columns else "marconi_a2_mean_aggregate"
                
                ts_cols = [c for c in ["timestamp", "datetime", "date_time", "time"] if c in clean_df.columns]
                if ts_cols and pd.to_datetime(clean_df[ts_cols[0]], errors="coerce").dropna().shape[0] > 0:
                    last_val = pd.to_datetime(clean_df[ts_cols[0]], errors="coerce").dropna().iloc[-1]
                    reading_time = format_forecast_timestamp(last_val)
                else:
                    reading_time = str(clean_df["timestamp"].iloc[-1]) if "timestamp" in clean_df.columns else "Unknown"

            cooling_result = self.cooling_ctrl.evaluate_forecast(
                pred_result["forecast_temperatures"],
                pred_result["forecast_timestamps"],
                current_temp=current_inlet,
                sampling_interval_minutes=interval_mins
            )

            f_15 = resample_forecast_15min(
                pred_result["forecast_temperatures"],
                pred_result["forecast_timestamps"],
                on_th=on_th,
                off_th=off_th,
                initial_chiller_state="ON" if self.cooling_ctrl.is_running else "OFF"
            )

            data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset_source": os.path.basename(dataset_path),
                "schema_type": schema_type,
                "schema_name": schema_name,
                "telemetry_current": {
                    "inlet_temperature": current_inlet,
                    "outlet_temperature": current_outlet,
                    "cpu_usage": float(clean_df["cpu_usage"].iloc[-1]) if ("cpu_usage" in clean_df.columns and clean_df["cpu_usage"].notna().any()) else None,
                    "it_power_consumption": current_power,
                    "facility_power": current_fac,
                    "cooling_power_consumption": float(clean_df["cooling_power_consumption"].iloc[-1]) if ("cooling_power_consumption" in clean_df.columns and clean_df["cooling_power_consumption"].notna().any()) else None,
                    "node_id": node_label,
                    "last_reading_time": reading_time
                },
                "prediction": pred_result,
                "cooling": cooling_result,
                "forecast_15min": f_15
            }
            data["prediction"]["forecast_15min"] = f_15

            os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)
            with open(LATEST_FORECAST_CACHE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return data

        return {}

    def get_dashboard_summary(self) -> Dict[str, Any]:
        fc = self.get_latest_forecast()
        if not fc:
            return {}
            
        telemetry = fc.get("telemetry_current", {})
        cooling = fc.get("cooling", {})
        pred = fc.get("prediction", {})
        alerts = self.alert_mgr.get_recent(limit=5)
        
        curr_inlet = telemetry.get("inlet_temperature", 24.0)
        
        # Query authoritative threshold setpoints from threshold manager
        th = self.cooling_ctrl.threshold_mgr.get_thresholds()
        on_th = th.on_threshold
        off_th = th.off_threshold
        
        # Centralized cooling evaluation on current measured telemetry
        prev_st = self.cooling_ctrl.current_state
        imm_decision = determine_cooling_decision(
            curr_inlet,
            previous_state=prev_st,
            on_threshold=on_th,
            off_threshold=off_th
        )
        
        return {
            "current_inlet_temperature": curr_inlet,
            "current_outlet_temperature": telemetry.get("outlet_temperature", 27.0),
            "predicted_max_temperature": pred.get("max_temperature", 26.5),
            "predicted_avg_temperature": pred.get("avg_temperature", 24.5),
            "cpu_utilization": telemetry.get("cpu_usage"),
            "it_power_consumption": telemetry.get("it_power_consumption", 0.85),
            "facility_power": telemetry.get("facility_power"),
            "cooling_power_consumption": telemetry.get("cooling_power_consumption"),
            "node_id": telemetry.get("node_id", "industrial_data_center"),
            "dataset_source": fc.get("dataset_source", "unknown"),
            "schema_type": fc.get("schema_type", "HAZARDNET_OPERATIONAL"),
            "schema_name": fc.get("schema_name", "HazardNet Operational Telemetry"),
            "experiment_id": pred.get("experiment_id"),
            "available_experiments": pred.get("available_experiments", []),
            "forecast_horizon_hours": pred.get("forecast_horizon_hours", 24.0),
            "forecast_horizon_steps": pred.get("prediction_count", len(pred.get("forecast_temperatures", []))),
            "sampling_interval_minutes": pred.get("sampling_interval_minutes", 2),
            "last_reading_time": telemetry.get("last_reading_time"),
            "chiller_status": imm_decision["legacy_state"],  # "RUNNING" or "STOPPED"
            "chiller_state": imm_decision["chiller_state"],   # "ON" or "OFF"
            "is_running": imm_decision["is_running"],
            "temperature_status": imm_decision["temperature_status"],
            "activation_reason": imm_decision["reason"],
            "operating_mode": cooling.get("operating_mode", "AUTOMATIC"),
            "expected_activation_time": cooling.get("expected_activation_time"),
            "predicted_running_hours": cooling.get("predicted_running_hours", 0.0),
            "on_threshold": on_th,
            "off_threshold": off_th,
            "active_alerts_count": len(alerts),
            "last_updated": fc.get("timestamp")
        }

    def get_telemetry_history(self, hours: int = 24, experiment_id: Optional[str] = None) -> Dict[str, Any]:
        dataset_path = self.dataset_svc.get_current_dataset_path()
        if not dataset_path or not os.path.exists(dataset_path):
            return {}
            
        df = load_and_parse_csv(dataset_path)
        schema_type, _ = detect_schema(df)
        
        if schema_type == "AAU_EXPERIMENTAL":
            available_exps = df["experiment_id"].dropna().unique().tolist()
            if not experiment_id or experiment_id not in available_exps:
                experiment_id = available_exps[0] if available_exps else ""
            exp_df = df[df["experiment_id"] == experiment_id].copy()
            exp_df = exp_df.sort_values(by="elapsed_time_minutes").reset_index(drop=True)
            interval_mins = 2
            steps = min(len(exp_df), int(round((hours * 60.0) / interval_mins)))
            subset = exp_df.tail(steps).copy()
            timestamps = [f"{int(m)} min" for m in subset["elapsed_time_minutes"]]
        else:
            interval_mins, _ = detect_sampling_interval_and_horizon(df, schema="HAZARDNET_OPERATIONAL")
            steps = min(len(df), int(round((hours * 60.0) / interval_mins)))
            subset = df.iloc[-steps:].copy()
            timestamps = [format_forecast_timestamp(t) for t in subset["timestamp"]]
        
        # Calculate historical chiller state and temperature status sequentially for each timestamp
        th = self.cooling_ctrl.threshold_mgr.get_thresholds()
        on_th = th.on_threshold
        off_th = th.off_threshold
        
        sim_state = "OFF"
        hist_chiller_states = []
        hist_temp_statuses = []
        for temp_val in subset["inlet_temperature"]:
            t_flt = float(temp_val) if pd.notna(temp_val) else 24.0
            dec = determine_cooling_decision(t_flt, previous_state=sim_state, on_threshold=on_th, off_threshold=off_th)
            hist_chiller_states.append(dec["legacy_state"]) # RUNNING or STOPPED
            hist_temp_statuses.append(dec["temperature_status"])
            sim_state = dec["chiller_state"]
        
        return {
            "hours": hours,
            "schema_type": schema_type,
            "experiment_id": experiment_id,
            "total_points": len(subset),
            "timestamps": timestamps,
            "inlet_temperature": [round(float(x), 2) for x in subset["inlet_temperature"]],
            "outlet_temperature": [round(float(x), 2) for x in subset["outlet_temperature"]] if "outlet_temperature" in subset.columns else [],
            "it_power_consumption": [round(float(x), 2) for x in subset["it_power_consumption"]] if "it_power_consumption" in subset.columns else [],
            "facility_power": [round(float(x), 2) for x in subset["facility_power"]] if "facility_power" in subset.columns else None,
            "hot_aisle_temperature": [round(float(x), 2) for x in subset["hot_aisle_temperature"]] if ("hot_aisle_temperature" in subset.columns and subset["hot_aisle_temperature"].notna().any()) else None,
            "cold_aisle_temperature": [round(float(x), 2) for x in subset["cold_aisle_temperature"]] if ("cold_aisle_temperature" in subset.columns and subset["cold_aisle_temperature"].notna().any()) else None,
            "humidity": [round(float(x), 2) for x in subset["humidity"]] if ("humidity" in subset.columns and subset["humidity"].notna().any()) else None,
            "outdoor_temperature": [round(float(x), 2) for x in subset["outdoor_temperature"]] if ("outdoor_temperature" in subset.columns and subset["outdoor_temperature"].notna().any()) else None,
            "cpu_usage": [round(float(x), 2) for x in subset["cpu_usage"]] if ("cpu_usage" in subset.columns and subset["cpu_usage"].notna().any()) else None,
            "cooling_power_consumption": [round(float(x), 2) for x in subset["cooling_power_consumption"]] if ("cooling_power_consumption" in subset.columns and subset["cooling_power_consumption"].notna().any()) else None,
            "chiller_states": hist_chiller_states,
            "temperature_statuses": hist_temp_statuses
        }

    def export_predictions_csv(self, resolution: str = "15min") -> str:
        fc = self.get_latest_forecast()
        pred = fc.get("prediction", {})
        cooling = fc.get("cooling", {})

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["serial_number", "forecast_timestamp", "predicted_inlet_temperature_celsius", "chiller_decision", "temperature_status"])

        th = self.cooling_ctrl.threshold_mgr.get_thresholds()
        on_th = th.on_threshold
        off_th = th.off_threshold

        if resolution == "15min":
            records_15 = fc.get("forecast_15min")
            if not records_15 or len(records_15) != 96:
                temps = pred.get("forecast_temperatures", [])
                times = pred.get("forecast_timestamps", [])
                records_15 = resample_forecast_15min(
                    temps, times, on_th=on_th, off_th=off_th,
                    initial_chiller_state="ON" if self.cooling_ctrl.is_running else "OFF"
                )

            for r in records_15:
                writer.writerow([
                    r.get("serial_number") or r.get("index"),
                    r.get("forecast_timestamp") or r.get("timestamp"),
                    r.get("predicted_inlet_temperature_celsius") or r.get("temperature"),
                    r.get("chiller_decision") or ("RUNNING" if r.get("is_running") else "STOPPED"),
                    r.get("temperature_status") or r.get("condition")
                ])
        else:
            # Full 720-step internal resolution
            temps = pred.get("forecast_temperatures", [])
            timestamps = pred.get("forecast_timestamps", [])
            states = cooling.get("predicted_states", ["STOPPED"] * len(temps))
            statuses = cooling.get("temperature_statuses", [])
            sim_state = "OFF"

            for i, (t, ts) in enumerate(zip(temps, timestamps)):
                st = states[i] if i < len(states) else ("RUNNING" if t >= on_th else "STOPPED")
                if i < len(statuses):
                    stat = statuses[i]
                else:
                    dec = determine_cooling_decision(t, previous_state=sim_state, on_threshold=on_th, off_threshold=off_th)
                    stat = dec["temperature_status"]
                    sim_state = dec["chiller_state"]

                writer.writerow([i + 1, ts, round(float(t), 2), st, stat])

        return output.getvalue()

_prediction_service = None

def get_prediction_service() -> PredictionService:
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service
