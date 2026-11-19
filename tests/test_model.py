"""
Unit and Integration Tests for Model Inference, Output Shapes, and Evaluation
"""
import os
import unittest
import numpy as np
import pandas as pd
from models.predict import TemperaturePredictor, get_predictor
from preprocessing.data_loader import load_dataset

class TestModelInference(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.predictor = get_predictor()
        cls.df = load_dataset("dataset/data_center_temperature_dataset.csv")
        
    def test_forecast_horizon_and_dimensions(self):
        """Verify that a 24-hour forecast at 5-minute resolution returns exactly 288 predicted values."""
        recent_window = self.df.iloc[-288:]
        result = self.predictor.predict(recent_window)
        
        forecast_vals = result["forecast_values"]
        timestamps = result["forecast_timestamps"]
        
        self.assertEqual(len(forecast_vals), 288, f"Expected exactly 288 predictions, got {len(forecast_vals)}")
        self.assertEqual(len(timestamps), 288, f"Expected exactly 288 timestamps, got {len(timestamps)}")
        self.assertEqual(result["horizon_steps"], 288)

    def test_temperature_prediction_bounds(self):
        """Verify that inverse-scaled temperature outputs lie within realistic physical data center bounds."""
        recent_window = self.df.iloc[-288:]
        result = self.predictor.predict(recent_window)
        
        for val in result["forecast_values"]:
            self.assertGreaterEqual(val, 15.0, f"Predicted temperature {val}°C is abnormally low.")
            self.assertLessEqual(val, 38.0, f"Predicted temperature {val}°C is abnormally high.")
            
        self.assertAlmostEqual(result["max_temperature"], max(result["forecast_values"]), places=2)
        self.assertAlmostEqual(result["min_temperature"], min(result["forecast_values"]), places=2)

    def test_saved_model_and_scaler_exist(self):
        """Ensure model and scaler files exist in models/saved/."""
        self.assertTrue(os.path.exists("models/saved/lstm_temperature_model.keras"))
        self.assertTrue(os.path.exists("models/saved/scaler.pkl"))
        self.assertTrue(os.path.exists("models/saved/training_history.json"))
        self.assertTrue(os.path.exists("models/saved/evaluation_metrics.json"))

if __name__ == "__main__":
    unittest.main()
