"""
Predictive Cooling Controller with Dual Operating Modes (Automatic & Manual)
and Centralized Single-Source-of-Truth Decision Logic.
"""
import os
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from backend.config import Config
from backend.cooling.threshold import get_threshold_manager
from backend.cooling.alerts import get_alert_manager

EVENTS_FILE = os.path.join(Config.STORAGE_DIR, "cooling_events.json")
STATE_FILE = os.path.join(Config.STORAGE_DIR, "cooling_state.json")

def determine_cooling_decision(
    temperature: float,
    previous_state: str = "OFF",
    on_threshold: float = 28.5,
    off_threshold: float = 26.5
) -> Dict[str, Any]:
    """
    Centralized Single-Source-of-Truth Cooling State & Thermal Status Function.

    Hysteresis Logic:
    - If temperature >= on_threshold (e.g. 28.5°C):
        chiller_state = "ON"
        temperature_status = "HIGH TEMPERATURE"
        is_running = True
    - If temperature <= off_threshold (e.g. 26.5°C):
        chiller_state = "OFF"
        temperature_status = "NORMAL / COOLING NOT REQUIRED"
        is_running = False
    - If between off_threshold and on_threshold (26.5°C < T < 28.5°C):
        chiller_state = previous_state ("ON" or "OFF")
        temperature_status = "NORMAL / RETAIN CURRENT STATE"
        is_running = (chiller_state == "ON")
    """
    temp = float(temperature)
    on_th = float(on_threshold)
    off_th = float(off_threshold)

    prev_norm = "ON" if str(previous_state).strip().upper() in ["ON", "RUNNING", "TRUE", "1"] else "OFF"

    if temp >= on_th:
        state = "ON"
        status = "HIGH TEMPERATURE"
        is_running = True
        reason = f"Temperature {temp:.2f}°C >= ON threshold ({on_th:.1f}°C): Chiller ON."
    elif temp <= off_th:
        state = "OFF"
        status = "NORMAL / COOLING NOT REQUIRED"
        is_running = False
        reason = f"Temperature {temp:.2f}°C <= OFF threshold ({off_th:.1f}°C): Chiller OFF."
    else:
        state = prev_norm
        status = "NORMAL / RETAIN CURRENT STATE"
        is_running = (prev_norm == "ON")
        reason = f"Temperature {temp:.2f}°C in deadband ({off_th:.1f}°C - {on_th:.1f}°C): Retaining previous state ({prev_norm})."

    return {
        "temperature": round(temp, 4),
        "chiller_state": state,
        "legacy_state": "RUNNING" if is_running else "STOPPED",
        "is_running": is_running,
        "temperature_status": status,
        "reason": reason,
        "on_threshold": on_th,
        "off_threshold": off_th
    }


class PredictiveCoolingController:
    def __init__(self, events_file: str = EVENTS_FILE, state_file: str = STATE_FILE):
        self.events_file = events_file
        self.state_file = state_file
        self.threshold_mgr = get_threshold_manager()
        self.alert_mgr = get_alert_manager()
        
        # Load persisted state or defaults
        self.operating_mode: str = "AUTOMATIC"  # "AUTOMATIC" or "MANUAL"
        self.current_state: str = "STOPPED"     # "RUNNING" (ON) or "STOPPED" (OFF)
        self.temperature_status: str = "NORMAL / COOLING NOT REQUIRED"
        self.last_temperature: Optional[float] = None
        self.activation_reason: str = "System Initialized at standby setpoint."
        self.last_transition_time: str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        
        self.events: List[Dict[str, Any]] = self._load_events()
        self._load_state()

    def _load_events(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.events_file):
            try:
                with open(self.events_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_events(self):
        os.makedirs(os.path.dirname(self.events_file), exist_ok=True)
        try:
            with open(self.events_file, "w", encoding="utf-8") as f:
                json.dump(self.events[-300:], f, indent=2)
        except Exception:
            pass

    def _load_state(self):
        """Loads persisted chiller operational state to prevent cold-restart state desynchronization."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.operating_mode = data.get("operating_mode", "AUTOMATIC")
                    self.current_state = data.get("current_state", "STOPPED")
                    self.temperature_status = data.get("temperature_status", "NORMAL / COOLING NOT REQUIRED")
                    self.last_temperature = data.get("last_temperature")
                    self.activation_reason = data.get("activation_reason", self.activation_reason)
                    self.last_transition_time = data.get("last_transition_time", self.last_transition_time)
                    return
            except Exception:
                pass
        
        # Fallback to last recorded event if state file absent
        if self.events:
            last_ev = self.events[-1]
            st = last_ev.get("new_state", "STOPPED")
            self.current_state = "RUNNING" if st in ["RUNNING", "ON"] else "STOPPED"
            self.operating_mode = last_ev.get("mode", "AUTOMATIC")
            self.last_transition_time = last_ev.get("timestamp", self.last_transition_time)
            self.activation_reason = last_ev.get("reason", self.activation_reason)

    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        try:
            payload = {
                "operating_mode": self.operating_mode,
                "current_state": self.current_state,
                "chiller_state": "ON" if self.is_running else "OFF",
                "temperature_status": self.temperature_status,
                "last_temperature": self.last_temperature,
                "activation_reason": self.activation_reason,
                "last_transition_time": self.last_transition_time,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    @property
    def is_running(self) -> bool:
        return self.current_state in ["RUNNING", "ON"]

    def get_status(self) -> Dict[str, Any]:
        th = self.threshold_mgr.get_thresholds()
        power = 135.0 if self.is_running else 48.0
        canonical_state = "ON" if self.is_running else "OFF"
        return {
            "operating_mode": self.operating_mode,
            "chiller_state": self.current_state,  # Preserves RUNNING/STOPPED for backwards compat
            "canonical_chiller_state": canonical_state,
            "is_running": self.is_running,
            "cooling_power_consumption_kw": power,
            "on_threshold": th.on_threshold,
            "off_threshold": th.off_threshold,
            "temperature_status": self.temperature_status,
            "current_temperature": self.last_temperature,
            "activation_reason": self.activation_reason,
            "last_transition_time": self.last_transition_time,
            "is_simulated": True
        }

    def set_operating_mode(self, mode: str, forecast_temps: Optional[List[float]] = None,
                           forecast_timestamps: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Switches between AUTOMATIC and MANUAL mode.
        If switching to AUTOMATIC, immediately recalculates cooling requirements from forecast.
        """
        mode = mode.upper()
        if mode not in ["AUTOMATIC", "MANUAL"]:
            raise ValueError(f"Invalid operating mode: {mode}. Must be 'AUTOMATIC' or 'MANUAL'.")
            
        old_mode = self.operating_mode
        self.operating_mode = mode
        
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        self.alert_mgr.add_alert(
            severity="INFO",
            title=f"Cooling Mode Switched to {mode}",
            message=f"Operator changed cooling operating mode from {old_mode} to {mode}.",
            timestamp=now_str
        )
        
        if mode == "AUTOMATIC" and forecast_temps and forecast_timestamps:
            self.evaluate_forecast(forecast_temps, forecast_timestamps)
        else:
            self._save_state()
            
        return self.get_status()

    def manual_override(self, target_state: str) -> Dict[str, Any]:
        """
        Executes manual chiller override: 'RUNNING' (ON) or 'STOPPED' (OFF).
        Switches mode to MANUAL and records transition event.
        """
        target_state = "RUNNING" if target_state.upper() in ["ON", "RUNNING"] else "STOPPED"
        old_state = self.current_state
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        
        self.operating_mode = "MANUAL"
        self.current_state = target_state
        self.activation_reason = f"Manual Operator Override: Chiller turned {target_state} via control panel."
        self.last_transition_time = now_str
        self.temperature_status = "MANUAL OVERRIDE"
        
        event = {
            "timestamp": now_str,
            "mode": "MANUAL",
            "previous_state": old_state,
            "new_state": target_state,
            "trigger_temperature": None,
            "threshold": None,
            "reason": self.activation_reason
        }
        self.events.append(event)
        self._save_events()
        self._save_state()
        
        self.alert_mgr.add_alert(
            severity="WARNING" if target_state == "RUNNING" else "INFO",
            title=f"Chiller Manually Turned {target_state}",
            message=self.activation_reason,
            timestamp=now_str
        )
        return self.get_status()

    def evaluate_forecast(self, forecast_temps: List[float], 
                          forecast_timestamps: List[str], 
                          current_temp: Optional[float] = None,
                          sampling_interval_minutes: int = 2) -> Dict[str, Any]:
        """
        Evaluates a sequence of predicted temperatures using centralized single-source hysteresis logic.
        In AUTOMATIC mode, updates current chiller state accordingly.
        In MANUAL mode, computes upcoming projection without overriding manual override state.
        Calculates cooling duration strictly over the actual forecast horizon.
        """
        th = self.threshold_mgr.get_thresholds()
        on_th = th.on_threshold
        off_th = th.off_threshold
        
        sim_state = "ON" if self.is_running else "OFF"
        predicted_states = []
        temperature_statuses = []
        expected_activation_time = None
        upcoming_events = []
        
        for i, (temp, ts) in enumerate(zip(forecast_temps, forecast_timestamps)):
            decision = determine_cooling_decision(
                temp,
                previous_state=sim_state,
                on_threshold=on_th,
                off_threshold=off_th
            )
            
            next_state = decision["chiller_state"]
            legacy_state = decision["legacy_state"]
            predicted_states.append(legacy_state)  # RUNNING or STOPPED
            temperature_statuses.append(decision["temperature_status"])
            
            if next_state != sim_state:
                event_entry = {
                    "step_index": i,
                    "timestamp": ts,
                    "temperature": round(float(temp), 2),
                    "previous_state": "RUNNING" if sim_state == "ON" else "STOPPED",
                    "new_state": legacy_state,
                    "reason": decision["reason"]
                }
                upcoming_events.append(event_entry)
                if next_state == "ON" and expected_activation_time is None:
                    expected_activation_time = ts
                    
            sim_state = next_state

        # In AUTOMATIC mode, update immediate state based on immediate temp or step 0
        if self.operating_mode == "AUTOMATIC":
            imm_temp = current_temp if current_temp is not None else (forecast_temps[0] if forecast_temps else 25.0)
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            old_state = self.current_state
            prev_norm = "ON" if old_state in ["ON", "RUNNING"] else "OFF"
            
            imm_decision = determine_cooling_decision(
                imm_temp,
                previous_state=prev_norm,
                on_threshold=on_th,
                off_threshold=off_th
            )
            
            new_state = imm_decision["legacy_state"] # RUNNING or STOPPED
            self.last_temperature = float(imm_temp)
            self.temperature_status = imm_decision["temperature_status"]
            self.activation_reason = imm_decision["reason"]
            
            if new_state != old_state:
                self.current_state = new_state
                self.last_transition_time = now_str
                trans_event = {
                    "timestamp": now_str,
                    "mode": "AUTOMATIC",
                    "previous_state": old_state,
                    "new_state": new_state,
                    "trigger_temperature": round(float(imm_temp), 2),
                    "threshold": on_th if new_state == "RUNNING" else off_th,
                    "reason": imm_decision["reason"]
                }
                self.events.append(trans_event)
                self._save_events()
                
                self.alert_mgr.add_alert(
                    severity="WARNING" if new_state == "RUNNING" else "SUCCESS",
                    title=f"Automatic Chiller Transition: {new_state}",
                    message=imm_decision["reason"],
                    metric_value=imm_temp,
                    timestamp=now_str
                )
            else:
                self.current_state = new_state
                
            self._save_state()

        running_count = predicted_states.count("RUNNING")
        running_hours = round((running_count * sampling_interval_minutes) / 60.0, 2) if forecast_temps else 0.0
        total_hours = round((len(forecast_temps) * sampling_interval_minutes) / 60.0, 1) if forecast_temps else 0.0

        return {
            "operating_mode": self.operating_mode,
            "current_state": self.current_state,
            "canonical_chiller_state": "ON" if self.is_running else "OFF",
            "is_running": self.is_running,
            "temperature_status": self.temperature_status,
            "current_temperature": self.last_temperature,
            "activation_reason": self.activation_reason,
            "last_transition_time": self.last_transition_time,
            "on_threshold": on_th,
            "off_threshold": off_th,
            "expected_activation_time": expected_activation_time,
            "predicted_running_steps": running_count,
            "predicted_running_hours": running_hours,
            "total_forecast_hours": total_hours,
            "predicted_states": predicted_states,
            "temperature_statuses": temperature_statuses,
            "upcoming_events": upcoming_events
        }

    def get_event_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return sorted(self.events, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]

_cooling_controller = None

def get_cooling_controller() -> PredictiveCoolingController:
    global _cooling_controller
    if _cooling_controller is None:
        _cooling_controller = PredictiveCoolingController()
    return _cooling_controller
