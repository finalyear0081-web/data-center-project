"""
Alert Management Module
Generates, stores, and queries temperature and cooling system alerts.
"""
import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

ALERTS_LOG_PATH = "cooling/alerts_history.json"

class AlertManager:
    def __init__(self, log_path: str = ALERTS_LOG_PATH):
        self.log_path = log_path
        self.alerts: List[Dict[str, Any]] = self._load_alerts()
        
    def _load_alerts(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return []
        
    def _save_alerts(self):
        try:
            with open(self.log_path, "w") as f:
                json.dump(self.alerts[-200:], f, indent=2)  # retain last 200 alerts
        except Exception:
            pass

    def create_alert(self, severity: str, title: str, message: str, 
                     metric_value: Optional[float] = None, 
                     timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Creates and stores an alert.
        severity: "CRITICAL", "WARNING", "INFO", "SUCCESS"
        """
        alert = {
            "id": str(uuid.uuid4())[:8],
            "severity": severity.upper(),
            "title": title,
            "message": message,
            "metric_value": round(metric_value, 2) if metric_value is not None else None,
            "timestamp": timestamp or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "is_read": False
        }
        self.alerts.append(alert)
        self._save_alerts()
        return alert

    def evaluate_forecast_alerts(self, forecast_temps: List[float], 
                                 forecast_timestamps: List[str],
                                 on_threshold: float,
                                 critical_threshold: float = 29.5) -> List[Dict[str, Any]]:
        """
        Scans a 24-hour forecast and creates alerts for expected threshold violations.
        """
        generated = []
        max_t = max(forecast_temps) if forecast_temps else 0.0
        
        # Check critical limit
        if max_t >= critical_threshold:
            first_idx = next(i for i, t in enumerate(forecast_temps) if t >= critical_threshold)
            event_time = forecast_timestamps[first_idx]
            alert = self.create_alert(
                severity="CRITICAL",
                title="Critical Temperature Excursion Predicted",
                message=f"Inlet temperature predicted to reach {max_t:.2f}°C at {event_time}, exceeding critical limit of {critical_threshold}°C.",
                metric_value=max_t,
                timestamp=event_time
            )
            generated.append(alert)
        elif max_t >= on_threshold:
            first_idx = next(i for i, t in enumerate(forecast_temps) if t >= on_threshold)
            event_time = forecast_timestamps[first_idx]
            alert = self.create_alert(
                severity="WARNING",
                title="Chiller Activation Imminent",
                message=f"Inlet temperature predicted to cross ON threshold ({on_threshold}°C) reaching {max_t:.2f}°C at {event_time}.",
                metric_value=max_t,
                timestamp=event_time
            )
            generated.append(alert)
            
        return generated

    def get_recent_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        return sorted(self.alerts, key=lambda x: x["timestamp"], reverse=True)[:limit]

_alert_manager = None

def get_alert_manager() -> AlertManager:
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
