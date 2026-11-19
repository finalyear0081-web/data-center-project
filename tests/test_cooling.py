"""
Unit Tests for Predictive Cooling Hysteresis and Manual Control Modes
"""
import unittest
from datetime import datetime, timedelta
from backend.cooling.threshold import CoolingThresholds, get_threshold_manager
from backend.cooling.controller import PredictiveCoolingController

class TestCoolingController(unittest.TestCase):
    def setUp(self):
        self.ctrl = PredictiveCoolingController()
        self.ctrl.operating_mode = "AUTOMATIC"
        self.ctrl.current_state = "STOPPED"

    def tearDown(self):
        self.ctrl.operating_mode = "AUTOMATIC"
        self.ctrl.current_state = "STOPPED"
        self.ctrl._save_state()

    def test_threshold_validation_rules(self):
        """Verify threshold bounds and minimum 0.5°C hysteresis deadband."""
        valid_th = CoolingThresholds(on_threshold=28.0, off_threshold=26.0)
        is_val, _ = valid_th.validate()
        self.assertTrue(is_val)

        # Inverted thresholds
        invalid_th = CoolingThresholds(on_threshold=25.0, off_threshold=26.0)
        is_val, msg = invalid_th.validate()
        self.assertFalse(is_val)
        self.assertIn("strictly greater", msg)

        # Too narrow deadband (< 0.5°C)
        narrow_th = CoolingThresholds(on_threshold=26.2, off_threshold=26.0)
        is_val, msg = narrow_th.validate()
        self.assertFalse(is_val)
        self.assertIn("at least 0.5°C", msg)

    def test_automatic_hysteresis_logic(self):
        """
        Test hysteresis sequence:
        Start STOPPED -> 27.0°C (deadband, stays STOPPED)
                      -> 28.5°C (>= 28.0, switches to RUNNING)
                      -> 27.2°C (deadband, stays RUNNING)
                      -> 25.8°C (<= 26.0, switches to STOPPED)
                      -> 26.4°C (deadband, stays STOPPED)
        """
        temps = [24.0, 27.0, 28.5, 27.2, 25.8, 26.4]
        now = datetime(2024, 1, 1, 12, 0, 0)
        times = [(now + timedelta(minutes=5 * i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(len(temps))]

        res = self.ctrl.evaluate_forecast(temps, times)
        states = res["predicted_states"]

        self.assertEqual(states[0], "STOPPED")
        self.assertEqual(states[1], "STOPPED") # In deadband, retains STOPPED
        self.assertEqual(states[2], "RUNNING") # Crossed >= 28.0
        self.assertEqual(states[3], "RUNNING") # In deadband, retains RUNNING
        self.assertEqual(states[4], "STOPPED") # Crossed <= 26.0
        self.assertEqual(states[5], "STOPPED") # In deadband, retains STOPPED
        self.assertEqual(res["expected_activation_time"], times[2])

    def test_manual_override_on_off(self):
        """Test manual controls override automatic state."""
        # Manual ON
        status_on = self.ctrl.manual_override("ON")
        self.assertEqual(status_on["operating_mode"], "MANUAL")
        self.assertEqual(status_on["chiller_state"], "RUNNING")
        self.assertTrue(status_on["is_running"])
        self.assertEqual(status_on["cooling_power_consumption_kw"], 135.0)

        # Manual OFF
        status_off = self.ctrl.manual_override("OFF")
        self.assertEqual(status_off["operating_mode"], "MANUAL")
        self.assertEqual(status_off["chiller_state"], "STOPPED")
        self.assertFalse(status_off["is_running"])
        self.assertEqual(status_off["cooling_power_consumption_kw"], 48.0)

    def test_mode_switching_recalculation(self):
        """Verify returning to AUTOMATIC recalculates from forecast."""
        self.ctrl.manual_override("ON")
        self.assertEqual(self.ctrl.current_state, "RUNNING")

        # Now switch back to AUTOMATIC with a cool temperature profile (< 26°C)
        cool_temps = [24.0, 24.5, 25.0]
        times = ["T1", "T2", "T3"]
        status = self.ctrl.set_operating_mode("AUTOMATIC", forecast_temps=cool_temps, forecast_timestamps=times)
        
        self.assertEqual(status["operating_mode"], "AUTOMATIC")
        self.assertEqual(status["chiller_state"], "STOPPED")

    def test_event_history_recording(self):
        """Verify transitions are recorded in history."""
        self.ctrl.manual_override("ON")
        events = self.ctrl.get_event_history(limit=5)
        self.assertTrue(len(events) > 0)
        self.assertEqual(events[0]["new_state"], "RUNNING")

if __name__ == "__main__":
    unittest.main()
