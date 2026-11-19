"""
Unit Tests for Centralized Canonical Schema Registry & Column Normalization
Tests CSV A (inlet_temperature), CSV B (inlet_temperature_c), and CSV C (Inlet Temp Avg)
Validates structured error handling for missing target (no raw KeyError).
"""
import os
import unittest
import tempfile
import pandas as pd
import numpy as np

# Ensure clean Keras/TensorFlow environment
os.environ["KERAS_HOME"] = os.path.abspath(".keras")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from backend.preprocessing.canonical_schema import (
    normalize_column_name,
    find_canonical_match,
    normalize_dataframe_to_canonical,
    CanonicalSchemaError,
    CANONICAL_TARGET_KEY,
    CANONICAL_REGISTRY
)
from backend.preprocessing.data_loader import load_and_parse_csv
from backend.preprocessing.normalization import DataCenterScaler
from backend.preprocessing.validation import validate_csv_dataset, detect_schema


class TestCanonicalMapping(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_normalize_column_name(self):
        """Verifies lowercase, whitespace trimming, and underscore conversion."""
        self.assertEqual(normalize_column_name("Inlet Temp Avg"), "inlet_temp_avg")
        self.assertEqual(normalize_column_name("  inlet_temperature_c  "), "inlet_temperature_c")
        self.assertEqual(normalize_column_name("P_IT"), "p_it")
        self.assertEqual(normalize_column_name("Date/Time"), "date_time")
        self.assertEqual(normalize_column_name("Cooling Supply Average"), "cooling_supply_average")
        self.assertEqual(normalize_column_name("Relative Humidity (%)"), "relative_humidity")

    def test_find_canonical_match(self):
        """Verifies canonical match resolution across diverse aliases."""
        self.assertEqual(find_canonical_match("inlet_temperature"), "inlet_temperature")
        self.assertEqual(find_canonical_match("inlet_temperature_c"), "inlet_temperature")
        self.assertEqual(find_canonical_match("Inlet Temp Avg"), "inlet_temperature")
        self.assertEqual(find_canonical_match("server_inlet_temp"), "inlet_temperature")
        self.assertEqual(find_canonical_match("inlet_temp"), "inlet_temperature")
        
        self.assertEqual(find_canonical_match("outlet_temperature"), "outlet_temperature")
        self.assertEqual(find_canonical_match("outlet_temperature_c"), "outlet_temperature")
        self.assertEqual(find_canonical_match("Outlet Avg Temp"), "outlet_temperature")
        
        self.assertEqual(find_canonical_match("it_power_p_it"), "it_power_consumption")
        self.assertEqual(find_canonical_match("P_IT"), "it_power_consumption")
        self.assertEqual(find_canonical_match("facility_power_p_dc"), "facility_power_consumption")
        self.assertEqual(find_canonical_match("P_DC"), "facility_power_consumption")

    def test_csv_a_inlet_temperature(self):
        """Case A: CSV with 'inlet_temperature' column."""
        csv_path = os.path.join(self.temp_dir.name, "test_csv_a.csv")
        df_raw = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="5min"),
            "inlet_temperature": [22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5],
            "outlet_temperature": [30.0, 30.5, 31.0, 31.5, 32.0, 32.5, 33.0, 33.5, 34.0, 34.5],
            "it_power_consumption": [1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65]
        })
        df_raw.to_csv(csv_path, index=False)

        loaded_df = load_and_parse_csv(csv_path)
        self.assertIn("inlet_temperature", loaded_df.columns)
        self.assertIn("inlet_temperature_c", loaded_df.columns)  # Backwards compatibility mirror
        self.assertEqual(loaded_df["inlet_temperature"].iloc[0], 22.0)
        self.assertEqual(loaded_df["inlet_temperature"].iloc[-1], 26.5)

    def test_csv_b_inlet_temperature_c(self):
        """Case B: CSV with 'inlet_temperature_c' column (AAU style)."""
        csv_path = os.path.join(self.temp_dir.name, "test_csv_b.csv")
        df_raw = pd.DataFrame({
            "experiment_id": ["exp_1"] * 10,
            "elapsed_time_minutes": list(range(0, 20, 2)),
            "inlet_temperature_c": [21.0, 21.2, 21.4, 21.6, 21.8, 22.0, 22.2, 22.4, 22.6, 22.8],
            "outlet_temperature_c": [28.0, 28.2, 28.4, 28.6, 28.8, 29.0, 29.2, 29.4, 29.6, 29.8],
            "it_power_p_it": [0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89],
            "facility_power_p_dc": [1.5, 1.51, 1.52, 1.53, 1.54, 1.55, 1.56, 1.57, 1.58, 1.59]
        })
        df_raw.to_csv(csv_path, index=False)

        loaded_df = load_and_parse_csv(csv_path)
        self.assertIn("inlet_temperature", loaded_df.columns)
        self.assertIn("inlet_temperature_c", loaded_df.columns)
        self.assertEqual(loaded_df["inlet_temperature"].iloc[0], 21.0)
        self.assertEqual(loaded_df["inlet_temperature"].iloc[-1], 22.8)
        self.assertIn("it_power_consumption", loaded_df.columns)
        self.assertIn("facility_power_consumption", loaded_df.columns)

    def test_csv_c_inlet_temp_avg(self):
        """Case C: CSV with 'Inlet Temp Avg' column (custom industrial format)."""
        csv_path = os.path.join(self.temp_dir.name, "test_csv_c.csv")
        df_raw = pd.DataFrame({
            "Date/Time": pd.date_range("2026-02-01", periods=10, freq="10min"),
            "Inlet Temp Avg": [23.1, 23.3, 23.5, 23.7, 23.9, 24.1, 24.3, 24.5, 24.7, 24.9],
            "Outlet Avg Temp": [32.1, 32.3, 32.5, 32.7, 32.9, 33.1, 33.3, 33.5, 33.7, 33.9],
            "P_IT": [1.1, 1.1, 1.2, 1.2, 1.3, 1.3, 1.4, 1.4, 1.5, 1.5]
        })
        df_raw.to_csv(csv_path, index=False)

        loaded_df = load_and_parse_csv(csv_path)
        self.assertIn("inlet_temperature", loaded_df.columns)
        self.assertEqual(loaded_df["inlet_temperature"].iloc[0], 23.1)
        self.assertEqual(loaded_df["inlet_temperature"].iloc[-1], 24.9)
        self.assertIn("outlet_temperature", loaded_df.columns)
        self.assertIn("it_power_consumption", loaded_df.columns)

    def test_missing_target_raises_canonical_schema_error(self):
        """Verifies structured CanonicalSchemaError is raised when inlet temperature is absent."""
        df_no_target = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="5min"),
            "humidity": [45.0, 46.0, 47.0, 48.0, 49.0],
            "cpu_usage": [50.0, 55.0, 60.0, 65.0, 70.0]
        })
        
        with self.assertRaises(CanonicalSchemaError) as ctx:
            normalize_dataframe_to_canonical(df_no_target)
            
        err = ctx.exception
        self.assertEqual(err.required_feature, "inlet_temperature")
        self.assertIn("humidity", err.detected_columns)
        self.assertIn("cpu_usage", err.detected_columns)
        self.assertIn("inlet_temperature", err.suggested_aliases)
        self.assertIn("inlet_temperature_c", err.suggested_aliases)
        self.assertIn("inlet_temp_avg", err.suggested_aliases)
        # Verify no raw KeyError
        self.assertFalse(isinstance(err, KeyError))

    def test_datacenter_scaler_with_canonical_schema(self):
        """Verifies DataCenterScaler trains with canonical inlet_temperature as target."""
        df = pd.DataFrame({
            "inlet_temperature": [20.0, 22.0, 24.0, 26.0, 28.0, 30.0],
            "outlet_temperature": [28.0, 30.0, 32.0, 34.0, 36.0, 38.0],
            "it_power_consumption": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        })
        scaler = DataCenterScaler()
        scaler.fit(df)
        self.assertEqual(scaler.target_column, "inlet_temperature")
        self.assertEqual(scaler.feature_columns[0], "inlet_temperature")
        
        # Test transform & inverse transform
        scaled = scaler.transform_features(df)
        self.assertEqual(scaled.shape, (6, 3))
        
        inv_target = scaler.inverse_transform_target(scaled[:, 0])
        np.testing.assert_allclose(inv_target, df["inlet_temperature"].values, atol=1e-5)

    def test_optional_features_do_not_cause_keyerror(self):
        """Verifies pipeline works even if optional features (e.g. cpu_usage, humidity) are absent."""
        df_minimal = pd.DataFrame({
            "inlet_temperature": [21.0, 22.0, 23.0, 24.0, 25.0]
        })
        canon_df, mapped, unmapped = normalize_dataframe_to_canonical(df_minimal)
        self.assertIn("inlet_temperature", canon_df.columns)
        
        scaler = DataCenterScaler()
        scaler.fit(canon_df)
        self.assertEqual(scaler.feature_columns, ["inlet_temperature"])
        scaled = scaler.transform_features(canon_df)
        self.assertEqual(scaled.shape, (5, 1))


if __name__ == "__main__":
    unittest.main()
