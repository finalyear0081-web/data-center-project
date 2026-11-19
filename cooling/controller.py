"""
Predictive Cooling Controller with Hysteresis
Implements simulated chiller state management based on 24-hour predicted temperature sequence.
"""
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from cooling.threshold import get_threshold_manager, CoolingThresholds
from cooling.alerts import get_alert_manager

EVENTS_FILE = "cooling/cooling_events.json"

class PredictiveCoolingController:
    """
    Simulates intelligent chiller control using hysteresis rules:
      - Turn ON when inlet temperature >= on_threshold (default 28.0°C)
      - Turn OFF when inlet temperature <= off_threshold (default 26.0°C)
      - Retain current state when in deadband (26.0°C < T < 28.0°C)
    """
    def __init__(self, events_file: str = EVENTS_FILE):
        self.events_file = events_file
        self.current_state: str = "OFF"  # "ON" or "OFF"
        self.events: List[Dict[str, Any]] = self._load_events()
        self.threshold_mgr = get_threshold_manager()
        self.alert_mgr = get_alert_manager()
        
    def _load_events(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.events_file):
            try:
                with open(self.events_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return []
        
    def _save_events(self):
        try:
            with open(self.events_file, "w") as f:
                json.dump(self.events[-200:], f, indent=2)
        except Exception:
            pass

    def get_status(self) -> Dict[str, Any]:
        """Returns current simulated chiller operational status."""
        th = self.threshold_mgr.get_thresholds()
        power = 135.0 if self.current_state == "ON" else 48.0 # Simulated kW
        return {
            "chiller_state": self.current_state,
            "on_threshold": th.on_threshold,
            "off_threshold": th.off_threshold,
            "cooling_power_consumption_kw": power,
            "is_simulated": True,
            "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        }

    def evaluate_forecast(self, forecast_temps: List[float], 
                          forecast_timestamps: List[str],
                          current_temp: Optional[float] = None) -> Dict[str, Any]:
        """
        Evaluates the sequence of future temperatures and determines optimal cooling decisions.
        
        Args:
            forecast_temps: List of 288 future inlet temperature forecasts.
            forecast_timestamps: List of 288 corresponding timestamps.
            current_temp: Optional currently observed inlet temperature.
            
        Returns:
            Dict containing:
              - predicted_states: List of "ON"/"OFF" states for each future timestep
              - initial_chiller_decision: immediate state recommendation
              - expected_activation_time: timestamp when chiller will first turn ON (if any)
              - total_on_steps: count of steps in ON state
              - total_on_hours: hours chiller will run
              - transitions: list of transition event objects
        """
        th = self.threshold_mgr.get_thresholds()
        on_th = th.on_threshold
        off_th = th.off_threshold
        
        sim_state = self.current_state
        predicted_states = []
        transitions = []
        expected_activation_time = None
        
        # Step through time-series with hysteresis logic
        for i, (temp, ts) in enumerate(zip(forecast_temps, forecast_timestamps)):
            prev = sim_state
            if temp >= on_th:
                sim_state = "ON"
            elif temp <= off_th:
                sim_state = "OFF"
            # Else maintain previous sim_state
            
            predicted_states.append(sim_state)
            
            # Detect state transition
            if sim_state != prev:
                trans_event = {
                    "step_index": i,
                    "timestamp": ts,
                    "temperature": round(temp, 2),
                    "previous_state": prev,
                    "new_state": sim_state,
                    "reason": f"Temperature {temp:.2f}°C crossed {'ON' if sim_state == 'ON' else 'OFF'} threshold ({on_th if sim_state == 'ON' else off_th}°C)"
                }
                transitions.append(trans_event)
                if sim_state == "ON" and expected_activation_time is None:
                    expected_activation_time = ts

        # Update controller's current state based on first forecast step (or current temp)
        immediate_temp = current_temp if current_temp is not None else forecast_temps[0]
        if immediate_temp >= on_th:
            new_curr = "ON"
        elif immediate_temp <= off_th:
            new_curr = "OFF"
        else:
            new_curr = self.current_state
            
        if new_curr != self.current_state:
            now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            event_entry = {
                "timestamp": now_str,
                "previous_state": self.current_state,
                "current_state": new_curr,
                "trigger_temperature": round(immediate_temp, 2),
                "threshold": on_th if new_curr == "ON" else off_th
            }
            self.events.append(event_entry)
            self._save_events()
            self.alert_mgr.create_alert(
                severity="INFO",
                title=f"Chiller Switched {new_curr}",
                message=f"Cooling chiller transitioned to {new_curr} at temperature {immediate_temp:.2f}°C.",
                metric_value=immediate_temp,
                timestamp=now_str
            )
            self.current_state = new_curr
            
        # Also evaluate alerts for high temperature excursions
        self.alert_mgr.evaluate_forecast_alerts(forecast_temps, forecast_timestamps, on_th)
        
        total_on = predicted_states.count("ON")
        total_hours = round(total_on * 5.0 / 60.0, 2)
        
        return {
            "current_state": self.current_state,
            "immediate_temperature": round(immediate_temp, 2),
            "on_threshold": on_th,
            "off_threshold": off_th,
            "expected_activation_time": expected_activation_time,
            "total_on_steps": total_on,
            "total_on_hours": total_hours,
            "predicted_states": predicted_states,
            "transitions": transitions,
            "event_count": len(transitions)
        }

    def get_event_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return sorted(self.events, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]

_controller = None

def get_cooling_controller() -> PredictiveCoolingController:
    global _controller
    if _controller is None:
        _controller = PredictiveCoolingController()
    return _controller
