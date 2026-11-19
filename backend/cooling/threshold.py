"""
Cooling Threshold Management and Constraint Validation
"""
import os
import json
from dataclasses import dataclass
from typing import Tuple
from backend.config import Config

SETTINGS_FILE = os.path.join(Config.STORAGE_DIR, "threshold_settings.json")

@dataclass
class CoolingThresholds:
    on_threshold: float = Config.DEFAULT_ON_THRESHOLD
    off_threshold: float = Config.DEFAULT_OFF_THRESHOLD
    min_allowable: float = 16.0
    max_allowable: float = 38.0
    
    def validate(self) -> Tuple[bool, str]:
        if self.on_threshold <= self.off_threshold:
            return False, f"Chiller ON threshold ({self.on_threshold}°C) must be strictly greater than OFF threshold ({self.off_threshold}°C)."
        if self.off_threshold < self.min_allowable:
            return False, f"OFF threshold cannot be set below safety floor of {self.min_allowable}°C."
        if self.on_threshold > self.max_allowable:
            return False, f"ON threshold cannot exceed ceiling of {self.max_allowable}°C."
        if (self.on_threshold - self.off_threshold) < 0.5:
            return False, "Hysteresis deadband must be at least 0.5°C to avoid short-cycling equipment."
        return True, "Valid configuration."

class ThresholdManager:
    def __init__(self, filepath: str = SETTINGS_FILE):
        self.filepath = filepath
        self._thresholds = self._load()

    def _load(self) -> CoolingThresholds:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return CoolingThresholds(
                        on_threshold=float(data.get("on_threshold", Config.DEFAULT_ON_THRESHOLD)),
                        off_threshold=float(data.get("off_threshold", Config.DEFAULT_OFF_THRESHOLD))
                    )
            except Exception:
                pass
        return CoolingThresholds()

    def get_thresholds(self) -> CoolingThresholds:
        return self._thresholds

    def update_thresholds(self, on_threshold: float, off_threshold: float) -> Tuple[bool, str]:
        new_th = CoolingThresholds(on_threshold=float(on_threshold), off_threshold=float(off_threshold))
        is_valid, msg = new_th.validate()
        if not is_valid:
            return False, msg
            
        self._thresholds = new_th
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump({
                "on_threshold": self._thresholds.on_threshold,
                "off_threshold": self._thresholds.off_threshold
            }, f, indent=2)
        return True, "Thresholds successfully saved."

_threshold_manager = None

def get_threshold_manager() -> ThresholdManager:
    global _threshold_manager
    if _threshold_manager is None:
        _threshold_manager = ThresholdManager()
    return _threshold_manager
