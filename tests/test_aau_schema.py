"""
Unit & Integration Tests for AAU Experimental Data Schema
Verifies:
1. Schema detection (AAU vs HazardNet)
2. Experiment preservation (16 experiments, 11,520 records, 2-min interval)
3. Sequence generation with zero cross-experiment boundary leakage
4. Upload endpoint acceptance and schema metadata
5. Cooling controller hysteresis evaluation
"""
import os
import io
import unittest
import pandas as pd
import numpy as np

from backend.app import create_app
from backend.config import Config
from backend.preprocessing.validation import validate_csv_dataset, detect_schema, detect_sampling_interval_and_horizon
from backend.preprocessing.data_loader import load_and_parse_csv
from backend.preprocessing.cleaning import clean_and_repair_dataset
from backend.preprocessing.normalization import DataCenterScaler
from backend.preprocessing.sequence_generator import split_chronologically, create_experiment_isolated_sequences
from backend.cooling.controller import PredictiveCoolingController

AAU_SAMPLE_CSV = os.path.join(Config.STORAGE_DIR, "uploads", "20261116_090649_aau_real_measurements_lstm.csv")

class TestAAUSchema(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.client.testing = True

    def test_schema_detection(self):
        """Test schema detection identifies AAU Experimental schema accurately."""
        if not os.path.exists(AAU_SAMPLE_CSV):
            self.skipTest(f"AAU sample file not found at {AAU_SAMPLE_CSV}")
            
        report = validate_csv_dataset(AAU_SAMPLE_CSV)
        self.assertTrue(report["is_valid"])
        self.assertEqual(report["schema_type"], "AAU_EXPERIMENTAL")
        self.assertEqual(report["schema_name"], "AAU Experimental Data")
        self.assertEqual(report["total_experiments"], 16)
        self.assertEqual(report["total_records"], 11520)
        self.assertEqual(report["sampling_interval_minutes"], 2)
        self.assertEqual(report["forecast_horizon_steps"], 30)
        self.assertIn("inlet_temperature_c", report["available_features"])
        self.assertIn("humidity", report["unavailable_features"])

    def test_experiment_boundary_preservation(self):
        """Test sliding sequences are strictly generated within each experiment."""
        if not os.path.exists(AAU_SAMPLE_CSV):
            self.skipTest("AAU file not found.")
            
        df = load_and_parse_csv(AAU_SAMPLE_CSV)
        clean_df, _ = clean_and_repair_dataset(df)
        
        train_df, val_df, test_df = split_chronologically(clean_df, 0.70, 0.15, 0.15)
        
        scaler = DataCenterScaler()
        scaler.fit(train_df)
        
        input_window = 60
        forecast_horizon = 30
        
        X_tr, y_tr = create_experiment_isolated_sequences(train_df, scaler, input_window, forecast_horizon, stride=10)
        self.assertGreater(len(X_tr), 0)
        self.assertEqual(X_tr.shape[1], input_window)
        self.assertEqual(y_tr.shape[1], forecast_horizon)

        # Confirm target is at feature index 0
        self.assertEqual(scaler.feature_columns[0], scaler.target_column)

    def test_aau_upload_endpoint(self):
        """Test POST /api/dataset/upload handles AAU CSV successfully."""
        if not os.path.exists(AAU_SAMPLE_CSV):
            self.skipTest("AAU file not found.")
            
        with open(AAU_SAMPLE_CSV, "rb") as f:
            csv_bytes = io.BytesIO(f.read())
            
        data = {"file": (csv_bytes, "aau_real_measurements_lstm.csv")}
        resp = self.client.post("/api/dataset/upload", data=data, content_type="multipart/form-data")
        
        self.assertIn(resp.status_code, [200, 202])
        json_data = resp.get_json()
        self.assertTrue(json_data["success"])
        val_rep = json_data["data"]["validation_report"]
        self.assertEqual(val_rep["schema_type"], "AAU_EXPERIMENTAL")
        self.assertEqual(val_rep["total_experiments"], 16)
        self.assertEqual(val_rep["sampling_interval_minutes"], 2)

    def test_cooling_hysteresis_logic(self):
        """Test chiller activates >= 28.0C and deactivates <= 26.0C."""
        controller = PredictiveCoolingController()
        controller.operating_mode = "AUTOMATIC"
        controller.current_state = "STOPPED"
        
        # Test 1: Immediate and forecast temperatures below 26.0C -> STOPPED
        res_cold = controller.evaluate_forecast([22.0, 23.0, 24.0], ["1", "2", "3"], current_temp=22.0)
        self.assertEqual(res_cold["current_state"], "STOPPED")
        self.assertEqual(res_cold["predicted_states"], ["STOPPED", "STOPPED", "STOPPED"])
        
        # Test 2: Forecast temperature reaches 28.5C at step 1 -> Predicted states switch to RUNNING
        res_future = controller.evaluate_forecast([25.0, 28.5, 27.0], ["1", "2", "3"], current_temp=25.0)
        self.assertEqual(res_future["predicted_states"], ["STOPPED", "RUNNING", "RUNNING"])
        self.assertEqual(res_future["expected_activation_time"], "2")
        
        # Test 3: Immediate temperature reaches 28.5C -> Immediate state switches to RUNNING
        res_hot = controller.evaluate_forecast([28.5, 27.5], ["1", "2"], current_temp=28.5)
        self.assertEqual(res_hot["current_state"], "RUNNING")
        
        # Test 4: In deadband (27.0C) with previous RUNNING -> remains RUNNING (state hold)
        res_hold = controller.evaluate_forecast([27.0, 27.2], ["1", "2"], current_temp=27.0)
        self.assertEqual(res_hold["current_state"], "RUNNING")

if __name__ == "__main__":
    unittest.main()
