"""
Prediction & System Orchestration Service
Coordinates data ingestion, LSTM inference, cooling controller evaluation,
and persistence to Firestore.
"""
import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np

from models.predict import get_predictor
from cooling.controller import get_cooling_controller
from cooling.threshold import get_threshold_manager
from cooling.alerts import get_alert_manager
from services.firebase_service import get_firebase_service
from preprocessing.data_loader import load_dataset, get_dataset_metadata

CACHE_PATH = "services/local_storage/latest_forecast_cache.json"

class PredictionService:
    def __init__(self, dataset_path: str = "dataset/data_center_temperature_dataset.csv"):
        self.dataset_path = dataset_path
        self.predictor = get_predictor()
        self.controller = get_cooling_controller()
        self.threshold_mgr = get_threshold_manager()
        self.alert_mgr = get_alert_manager()
        self.fb = get_firebase_service()
        self._cached_forecast: Optional[Dict[str, Any]] = self._load_cache()
        
    def _load_cache(self) -> Optional[Dict[str, Any]]:
        if os.path.exists(CACHE_PATH):
            try:
                with open(CACHE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None
        
    def _save_cache(self, data: Dict[str, Any]):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        try:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def run_prediction_cycle(self, custom_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Executes end-to-end forecasting:
          1. Extracts latest 288 records (24h)
          2. Runs LSTM inference -> 288 future temperatures
          3. Evaluates predictive cooling decisions with hysteresis
          4. Updates alerts & persists event metadata to Firestore
        """
        if custom_df is not None:
            df = custom_df
        else:
            df = load_dataset(self.dataset_path)
            
        recent_window = df.iloc[-288:].copy()
        current_inlet = float(recent_window["inlet_temperature"].iloc[-1])
        current_outlet = float(recent_window["outlet_temperature"].iloc[-1])
        current_cpu = float(recent_window["cpu_usage"].iloc[-1])
        current_it_power = float(recent_window["it_power_consumption"].iloc[-1])
        current_cooling_power = float(recent_window["cooling_power_consumption"].iloc[-1])
        
        # 1. Run LSTM inference
        forecast_result = self.predictor.predict(recent_window)
        
        forecast_vals = forecast_result["forecast_values"]
        forecast_times = forecast_result["forecast_timestamps"]
        
        # 2. Evaluate cooling controller
        cooling_eval = self.controller.evaluate_forecast(
            forecast_temps=forecast_vals,
            forecast_timestamps=forecast_times,
            current_temp=current_inlet
        )
        
        combined_result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "telemetry_current": {
                "inlet_temperature": current_inlet,
                "outlet_temperature": current_outlet,
                "cpu_usage": current_cpu,
                "it_power_consumption": current_it_power,
                "cooling_power_consumption": current_cooling_power,
                "last_reading_time": str(recent_window["timestamp"].iloc[-1])
            },
            "prediction": {
                "forecast_temperatures": forecast_vals,
                "forecast_timestamps": forecast_times,
                "max_temperature": forecast_result["max_temperature"],
                "min_temperature": forecast_result["min_temperature"],
                "avg_temperature": forecast_result["avg_temperature"],
                "horizon_steps": forecast_result["horizon_steps"]
            },
            "cooling": cooling_eval,
            "thresholds": {
                "on_threshold": cooling_eval["on_threshold"],
                "off_threshold": cooling_eval["off_threshold"]
            }
        }
        
        self._cached_forecast = combined_result
        self._save_cache(combined_result)
        
        # Save prediction summary to Firestore
        self.fb.save_record("predictions", {
            "timestamp": combined_result["timestamp"],
            "max_predicted_temp": forecast_result["max_temperature"],
            "avg_predicted_temp": forecast_result["avg_temperature"],
            "chiller_state": cooling_eval["current_state"],
            "expected_activation_time": cooling_eval["expected_activation_time"],
            "total_cooling_hours": cooling_eval["total_on_hours"]
        })
        
        return combined_result

    def get_latest_forecast(self) -> Dict[str, Any]:
        """Returns cached forecast or generates fresh cycle if none exists."""
        if self._cached_forecast is None:
            return self.run_prediction_cycle()
        return self._cached_forecast

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Returns unified dashboard KPI cards and status indicators."""
        forecast = self.get_latest_forecast()
        alerts = self.alert_mgr.get_recent_alerts(limit=5)
        active_critical = any(a["severity"] == "CRITICAL" for a in alerts)
        active_warning = any(a["severity"] == "WARNING" for a in alerts)
        
        telemetry = forecast["telemetry_current"]
        cooling = forecast["cooling"]
        pred = forecast["prediction"]
        
        return {
            "current_inlet_temperature": telemetry["inlet_temperature"],
            "current_outlet_temperature": telemetry["outlet_temperature"],
            "predicted_max_temperature": pred["max_temperature"],
            "predicted_avg_temperature": pred["avg_temperature"],
            "chiller_status": cooling["current_state"],
            "expected_activation_time": cooling["expected_activation_time"],
            "total_cooling_hours_next_24h": cooling["total_on_hours"],
            "cpu_utilization": telemetry["cpu_usage"],
            "it_power_consumption": telemetry["it_power_consumption"],
            "cooling_power_consumption": telemetry["cooling_power_consumption"],
            "on_threshold": cooling["on_threshold"],
            "off_threshold": cooling["off_threshold"],
            "active_alerts_count": len(alerts),
            "system_health": "CRITICAL" if active_critical else ("WARNING" if active_warning else "OPTIMAL"),
            "last_updated": forecast["timestamp"]
        }

    def get_historical_telemetry(self, hours: int = 24) -> Dict[str, Any]:
        """Returns historical time-series points for charts."""
        df = load_dataset(self.dataset_path)
        steps = min(len(df), int(hours * 12))
        subset = df.iloc[-steps:].copy()
        
        return {
            "timestamps": [str(t) for t in subset["timestamp"]],
            "inlet_temperature": [float(x) for x in subset["inlet_temperature"]],
            "outlet_temperature": [float(x) for x in subset["outlet_temperature"]],
            "hot_aisle_temperature": [float(x) for x in subset["hot_aisle_temperature"]],
            "cold_aisle_temperature": [float(x) for x in subset["cold_aisle_temperature"]],
            "humidity": [float(x) for x in subset["humidity"]],
            "outdoor_temperature": [float(x) for x in subset["outdoor_temperature"]],
            "cpu_usage": [float(x) for x in subset["cpu_usage"]],
            "it_power_consumption": [float(x) for x in subset["it_power_consumption"]],
            "cooling_power_consumption": [float(x) for x in subset["cooling_power_consumption"]],
            "total_points": steps,
            "hours": hours
        }

_prediction_service = None

def get_prediction_service() -> PredictionService:
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service
