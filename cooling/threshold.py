"""
Cooling Threshold Configuration and Validation
Manages administrative thresholds with safety constraints and hysteresis rules.
"""
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple

SETTINGS_FILE = "cooling/threshold_settings.json"

DEFAULT_ON_THRESHOLD = 28.0   # °C
DEFAULT_OFF_THRESHOLD = 26.0  # °C

@dataclass
class CoolingThresholds:
    on_threshold: float = DEFAULT_ON_THRESHOLD
    off_threshold: float = DEFAULT_OFF_THRESHOLD
    min_allowable: float = 16.0
    max_allowable: float = 38.0
    
    def validate(self) -> Tuple[bool, str]:
        """
        Validates threshold bounds and hysteresis gap.
        Returns: (is_valid, error_message)
        """
        if self.on_threshold <= self.off_threshold:
            return False, f"Chiller ON threshold ({self.on_threshold}°C) must be strictly greater than OFF threshold ({self.off_threshold}°C)."
        if self.off_threshold < self.min_allowable:
            return False, f"OFF threshold cannot be below safety limit of {self.min_allowable}°C."
        if self.on_threshold > self.max_allowable:
            return False, f"ON threshold cannot exceed maximum limit of {self.max_allowable}°C."
        if (self.on_threshold - self.off_threshold) < 0.5:
            return False, "Hysteresis band must be at least 0.5°C to avoid short-cycling."
        return True, "Valid"

class ThresholdManager:
    def __init__(self, filepath: str = SETTINGS_FILE):
        self.filepath = filepath
        self._thresholds = self._load()

    def _load(self) -> CoolingThresholds:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    data = json.load(f)
                    return CoolingThresholds(
                        on_threshold=float(data.get("on_threshold", DEFAULT_ON_THRESHOLD)),
                        off_threshold=float(data.get("off_threshold", DEFAULT_OFF_THRESHOLD))
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
        with open(self.filepath, "w") as f:
            json.dump({
                "on_threshold": self._thresholds.on_threshold,
                "off_threshold": self._thresholds.off_threshold,
                "updated_at": None
            }, f, indent=2)
        return True, "Thresholds updated successfully."

# Global singleton
_threshold_mgr = None

def get_threshold_manager() -> ThresholdManager:
    global _threshold_mgr
    if _threshold_mgr is None:
        _threshold_mgr = ThresholdManager()
    return _threshold_mgr
