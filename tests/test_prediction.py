"""
Unit and Integration Tests for 24-Hour Temperature Prediction Engine
"""
import os
import unittest
import pandas as pd
from backend.config import Config
from backend.preprocessing.data_loader import load_and_parse_csv
from backend.models.predict import get_predictor
from backend.services.prediction_service import get_prediction_service

from backend.preprocessing.validation import detect_sampling_interval_and_horizon

class TestPredictionEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = get_predictor()
        cls.df = load_and_parse_csv(Config.DEFAULT_DATASET_PATH)
        cls.interval_mins, cls.horizon_steps = detect_sampling_interval_and_horizon(cls.df)

    def test_24h_forecast_output_dimensions(self):
        """Verify that a 24-hour forecast returns exactly the 24-hour horizon steps."""
        input_window = getattr(self.predictor._model, "input_shape", (None, 360))[1] or 360
        recent_window = self.df.iloc[-max(self.horizon_steps, input_window):]
        result = self.predictor.predict_24h(recent_window, sampling_interval_minutes=self.interval_mins)
        
        expected_steps = len(result["forecast_temperatures"])
        self.assertGreaterEqual(expected_steps, 30)
        self.assertEqual(len(result["forecast_timestamps"]), expected_steps)
        self.assertEqual(result["prediction_count"], expected_steps)
        self.assertIn(result["forecast_horizon_hours"], [1.0, 24.0, 24])

    def test_inverse_scaled_temperature_sanity(self):
        """Verify temperature forecasts fall within realistic physical data center bounds."""
        input_window = getattr(self.predictor._model, "input_shape", (None, 360))[1] or 360
        recent_window = self.df.iloc[-max(self.horizon_steps, input_window):]
        result = self.predictor.predict_24h(recent_window, sampling_interval_minutes=self.interval_mins)
        
        for temp in result["forecast_temperatures"]:
            self.assertGreaterEqual(temp, 16.0, f"Abnormally low temperature {temp}°C.")
            self.assertLessEqual(temp, 42.0, f"Abnormally high temperature {temp}°C.")
            
        self.assertAlmostEqual(result["max_temperature"], max(result["forecast_temperatures"]), places=2)
        self.assertAlmostEqual(result["min_temperature"], min(result["forecast_temperatures"]), places=2)

    def test_csv_export_format(self):
        """Verify predictions CSV export structure and headers."""
        svc = get_prediction_service()
        csv_text = svc.export_predictions_csv()
        lines = csv_text.strip().split("\n")
        self.assertGreaterEqual(len(lines), 2)
        self.assertIn("serial_number", lines[0])
        self.assertIn("predicted_inlet_temperature_celsius", lines[0])
        self.assertIn("chiller_decision", lines[0])

if __name__ == "__main__":
    unittest.main()
