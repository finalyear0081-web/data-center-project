"""
Thermal Alert and Incident Logging Module
"""
import os
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from backend.config import Config

ALERTS_FILE = os.path.join(Config.STORAGE_DIR, "alerts.json")

class AlertManager:
    def __init__(self, filepath: str = ALERTS_FILE):
        self.filepath = filepath
        self.alerts: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.alerts[-200:], f, indent=2)
        except Exception:
            pass

    def add_alert(self, severity: str, title: str, message: str, 
                  metric_value: Optional[float] = None, 
                  timestamp: Optional[str] = None) -> Dict[str, Any]:
        alert = {
            "id": str(uuid.uuid4())[:8],
            "severity": severity.upper(),  # CRITICAL, WARNING, INFO, SUCCESS
            "title": title,
            "message": message,
            "metric_value": round(metric_value, 2) if metric_value is not None else None,
            "timestamp": timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        }
        self.alerts.append(alert)
        self._save()
        return alert

    def get_recent(self, limit: int = 30) -> List[Dict[str, Any]]:
        return sorted(self.alerts, key=lambda x: x["timestamp"], reverse=True)[:limit]

_alert_mgr = None

def get_alert_manager() -> AlertManager:
    global _alert_mgr
    if _alert_mgr is None:
        _alert_mgr = AlertManager()
    return _alert_mgr
