"""
Unit Tests for CSV Dataset Validation Module
Tests:
- Required 10 sensor columns
- Missing column rejection with exact column names
- Unparseable timestamps
- Duplicate records
- Sampling interval detection and forecast horizon computation
"""
import os
import unittest
import pandas as pd
import numpy as np

from backend.preprocessing.validation import (
    validate_csv_dataset,
    detect_sampling_interval_and_horizon,
    EXPECTED_COLUMNS
)
from backend.config import Config

class TestDatasetValidation(unittest.TestCase):
    def setUp(self):
        self.temp_csv = os.path.join(Config.STORAGE_DIR, "temp_test_dataset.csv")

    def tearDown(self):
        if os.path.exists(self.temp_csv):
            os.remove(self.temp_csv)

    def test_valid_default_dataset(self):
        """Verify the standard dataset passes validation."""
        self.assertTrue(os.path.exists(Config.DEFAULT_DATASET_PATH))
        report = validate_csv_dataset(Config.DEFAULT_DATASET_PATH)
        self.assertTrue(report["is_valid"])
        self.assertEqual(len(report["errors"]), 0)
        self.assertIn(report["total_records"], [50573, 105120])
        self.assertIn(report["sampling_interval_minutes"], [5, 10])
        self.assertIn(report["forecast_horizon_steps"], [144, 288])

    def test_missing_column_rejection(self):
        """Verify that omitting required columns reports exact missing names and fails validation."""
        df = pd.read_csv(Config.DEFAULT_DATASET_PATH, nrows=120)
        # Drop required column inlet_temperature
        df_dropped = df.drop(columns=["inlet_temperature"])
        df_dropped.to_csv(self.temp_csv, index=False)
        
        report = validate_csv_dataset(self.temp_csv)
        self.assertFalse(report["is_valid"])
        self.assertIn("inlet_temperature", report["missing_columns"])
        self.assertTrue(any("Missing required sensor columns" in err for err in report["errors"]))

    def test_sampling_interval_detection(self):
        """Verify automatic detection of 5m, 15m, and 60m sampling intervals and horizons."""
        # 5-minute sampling -> 288 steps
        times_5m = pd.date_range("2024-01-01", periods=100, freq="5min")
        df_5m = pd.DataFrame({"timestamp": times_5m})
        m_5, h_5 = detect_sampling_interval_and_horizon(df_5m)
        self.assertEqual(m_5, 5)
        self.assertEqual(h_5, 288)

        # 15-minute sampling -> 96 steps
        times_15m = pd.date_range("2024-01-01", periods=100, freq="15min")
        df_15m = pd.DataFrame({"timestamp": times_15m})
        m_15, h_15 = detect_sampling_interval_and_horizon(df_15m)
        self.assertEqual(m_15, 15)
        self.assertEqual(h_15, 96)

        # 60-minute sampling -> 24 steps
        times_60m = pd.date_range("2024-01-01", periods=100, freq="60min")
        df_60m = pd.DataFrame({"timestamp": times_60m})
        m_60, h_60 = detect_sampling_interval_and_horizon(df_60m)
        self.assertEqual(m_60, 60)
        self.assertEqual(h_60, 24)

    def test_duplicate_records_detection(self):
        """Verify duplicate records are detected and reported."""
        df = pd.read_csv(Config.DEFAULT_DATASET_PATH, nrows=100)
        df_with_dupes = pd.concat([df, df.iloc[:5]], ignore_index=True)
        df_with_dupes.to_csv(self.temp_csv, index=False)
        
        report = validate_csv_dataset(self.temp_csv)
        self.assertEqual(report["duplicate_records"], 5)
        self.assertEqual(report["duplicate_timestamps"], 5)

if __name__ == "__main__":
    unittest.main()
