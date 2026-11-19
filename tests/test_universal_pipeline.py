"""
Universal Pipeline Integration & Scenario Tests
Verifies the 8 core scenarios for universal data ingestion, automatic suitability analysis,
and predictive thermal modeling:
1. AAU experimental dataset
2. HazardNet operational dataset
3. Alternative column names dataset
4. Missing values dataset
5. Irregular timestamps dataset
6. Multiple experiment IDs
7. Unsuitable dataset (target missing)
8. Authorized external REST API simulation
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.config import Config
from backend.preprocessing.file_parser import parse_dataset_file
from backend.preprocessing.schema_detector import UniversalSchemaDetector
from backend.preprocessing.suitability_engine import DatasetSuitabilityEngine
from backend.preprocessing.cleaning import clean_and_repair_dataset
from backend.preprocessing.normalization import DataCenterScaler
from backend.preprocessing.sequence_generator import create_experiment_isolated_sequences
from backend.services.datasource_service import DataSourceService

class TestUniversalPipeline(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = os.path.join(Config.STORAGE_DIR, "test_fixtures")
        os.makedirs(self.tmp_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Scenario 1: AAU Experimental Dataset
    # -------------------------------------------------------------
    def test_scenario_1_aau_dataset(self):
        aau_path = None
        for base in [Config.DATASET_DIR, Config.UPLOADS_DIR]:
            if os.path.exists(base):
                for f in os.listdir(base):
                    if "aau_real_measurements_lstm.csv" in f:
                        aau_path = os.path.join(base, f)
                        break
            if aau_path:
                break
                
        if not aau_path or not os.path.exists(aau_path):
            self.skipTest("AAU dataset not present on disk.")
            
        df = pd.read_csv(aau_path)
        suitability = DatasetSuitabilityEngine.evaluate_dataset(df)
        
        self.assertEqual(suitability["verdict"], "PARTIALLY_SUITABLE")
        self.assertTrue(suitability["schema_checks"]["target_detected"])
        self.assertEqual(suitability["schema_checks"]["target_source_column"], "inlet_temperature_c")
        self.assertEqual(suitability["time_series_analysis"]["experiments_count"], 16)
        self.assertEqual(suitability["time_series_analysis"]["inferred_interval_minutes"], 2)
        self.assertEqual(suitability["recommended_horizon_hours"], 1.0)
        self.assertEqual(suitability["recommended_horizon_steps"], 30)

    # -------------------------------------------------------------
    # Scenario 2: HazardNet Operational Dataset
    # -------------------------------------------------------------
    def test_scenario_2_hazardnet_dataset(self):
        hazard_path = os.path.join(Config.DATASET_DIR, "hazardnet_cleaned_aggregate_data.csv")
        if not os.path.exists(hazard_path):
            hazard_path = os.path.join(Config.DATASET_DIR, "marconi_a2_operational_telemetry_2019.csv")
        if not os.path.exists(hazard_path):
            self.skipTest("HazardNet dataset not present on disk.")

        df = pd.read_csv(hazard_path)
        suitability = DatasetSuitabilityEngine.evaluate_dataset(df)

        self.assertEqual(suitability["verdict"], "SUITABLE")
        self.assertTrue(suitability["schema_checks"]["target_detected"])
        self.assertEqual(suitability["schema_checks"]["target_source_column"], "inlet_temperature")
        self.assertTrue(suitability["time_series_analysis"]["is_calendar_timestamp"])
        self.assertEqual(suitability["time_series_analysis"]["inferred_interval_minutes"], 10)
        self.assertTrue(suitability["is_24h_feasible"])
        self.assertEqual(suitability["recommended_horizon_hours"], 24.0)
        self.assertEqual(suitability["recommended_horizon_steps"], 144)

    # -------------------------------------------------------------
    # Scenario 3: Alternative Column Names Dataset
    # -------------------------------------------------------------
    def test_scenario_3_alternative_column_names(self):
        n = 300
        dates = pd.date_range("2026-01-01 00:00", periods=n, freq="5min")
        alt_df = pd.DataFrame({
            "date_time": dates,
            "rack_inlet_temp": np.random.normal(23.5, 1.2, n),
            "rack_outlet_temp": np.random.normal(32.0, 2.0, n),
            "it_power_kw": np.random.normal(45.0, 5.0, n),
            "cpu_utilization": np.random.uniform(20.0, 80.0, n)
        })

        mapping = UniversalSchemaDetector.detect_canonical_mapping(alt_df)
        self.assertTrue(mapping["target_detected"])
        self.assertEqual(mapping["mappings"]["inlet_temperature"]["source_column"], "rack_inlet_temp")
        self.assertEqual(mapping["mappings"]["outlet_temperature"]["source_column"], "rack_outlet_temp")
        self.assertEqual(mapping["mappings"]["it_power_consumption"]["source_column"], "it_power_kw")
        self.assertEqual(mapping["mappings"]["cpu_usage"]["source_column"], "cpu_utilization")
        self.assertEqual(mapping["mappings"]["timestamp"]["source_column"], "date_time")

        suitability = DatasetSuitabilityEngine.evaluate_dataset(alt_df)
        self.assertIn(suitability["verdict"], ["SUITABLE", "PARTIALLY_SUITABLE"])
        self.assertTrue(suitability["schema_checks"]["target_detected"])

    # -------------------------------------------------------------
    # Scenario 4: Missing Values Dataset
    # -------------------------------------------------------------
    def test_scenario_4_missing_values(self):
        n = 200
        dates = pd.date_range("2026-01-01 00:00", periods=n, freq="5min")
        inlet = np.linspace(22.0, 26.0, n)
        # Inject sporadic missing values
        inlet[10:15] = np.nan
        inlet[50] = np.nan
        
        df = pd.DataFrame({
            "timestamp": dates,
            "inlet_temperature": inlet,
            "outlet_temperature": inlet + 8.0,
            "it_power_consumption": np.full(n, 40.0)
        })

        cleaned_df, report = clean_and_repair_dataset(df)
        self.assertEqual(cleaned_df["inlet_temperature"].isna().sum(), 0)
        self.assertGreater(report["missing_imputed"].get("inlet_temperature", 0), 0)

    # -------------------------------------------------------------
    # Scenario 5: Irregular Timestamps Dataset
    # -------------------------------------------------------------
    def test_scenario_5_irregular_timestamps(self):
        n = 100
        # Create jittered timestamps
        intervals = [2, 10, 3, 15, 2, 8, 20, 1] * (n // 8)
        accum_mins = np.cumsum(intervals)
        base = pd.Timestamp("2026-01-01 00:00:00")
        timestamps = [base + pd.Timedelta(minutes=int(m)) for m in accum_mins]

        df = pd.DataFrame({
            "timestamp": timestamps,
            "inlet_temperature": np.random.normal(23.0, 1.0, len(timestamps))
        })

        suitability = DatasetSuitabilityEngine.evaluate_dataset(df)
        self.assertFalse(suitability["time_series_analysis"]["is_regular_sampling"])
        self.assertTrue(any("Irregular sampling" in w for w in suitability["warnings"]))

    # -------------------------------------------------------------
    # Scenario 6: Multiple Experiment IDs
    # -------------------------------------------------------------
    def test_scenario_6_multiple_experiment_ids(self):
        dfs = []
        for exp in ["exp_alpha", "exp_beta", "exp_gamma"]:
            sub = pd.DataFrame({
                "experiment_id": exp,
                "elapsed_time_minutes": np.arange(0, 200, 2),
                "inlet_temperature": np.random.normal(24.0, 1.0, 100),
                "outlet_temperature": np.random.normal(32.0, 1.5, 100)
            })
            dfs.append(sub)
        multi_df = pd.concat(dfs, ignore_index=True)

        suitability = DatasetSuitabilityEngine.evaluate_dataset(multi_df)
        self.assertEqual(suitability["time_series_analysis"]["experiments_count"], 3)
        self.assertEqual(len(suitability["time_series_analysis"]["experiments_list"]), 3)

        scaler = DataCenterScaler()
        scaler.fit(multi_df)
        X, y = create_experiment_isolated_sequences(multi_df, scaler, input_window=30, forecast_horizon=15, stride=5)
        self.assertGreater(len(X), 0)
        self.assertEqual(X.shape[1], 30)
        self.assertEqual(y.shape[1], 15)

    # -------------------------------------------------------------
    # Scenario 7: Unsuitable Dataset (Target Missing)
    # -------------------------------------------------------------
    def test_scenario_7_unsuitable_dataset(self):
        n = 100
        df_no_target = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="5min"),
            "cpu_utilization": np.random.uniform(10, 90, n),
            "fan_rpm": np.random.uniform(2000, 4500, n)
        })

        suitability = DatasetSuitabilityEngine.evaluate_dataset(df_no_target)
        self.assertEqual(suitability["verdict"], "NOT_SUITABLE")
        self.assertFalse(suitability["schema_checks"]["target_detected"])
        self.assertTrue(any("CRITICAL" in r for r in suitability["reasons"]))

    # -------------------------------------------------------------
    # Scenario 8: Authorized External REST API Simulation
    # -------------------------------------------------------------
    @patch("requests.get")
    def test_scenario_8_external_rest_api_simulation(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.reason = "OK"
        
        telemetry_records = [
            {
                "timestamp": f"2026-02-01T{h:02d}:{m:02d}:00Z",
                "inlet_temp": 22.5 + (h * 0.2),
                "outlet_temp": 30.0 + (h * 0.3),
                "power_kw": 42.0
            }
            for h in range(24)
            for m in [0, 15, 30, 45]
        ]
        mock_resp.json.return_value = {"status": "success", "readings": telemetry_records}
        mock_get.return_value = mock_resp

        service = DataSourceService(config_file=os.path.join(self.tmp_dir, "test_ds_cfg.json"))
        
        # Test connection
        test_res = service.test_connection(
            endpoint_url="https://api.datacenter.internal/telemetry",
            auth_type="API_KEY_HEADER",
            api_key="secret-token-12345"
        )
        self.assertTrue(test_res["success"])
        self.assertEqual(test_res["records_detected"], 96)
        self.assertIn("inlet_temp", test_res["sample_keys"])

        # Save connection & sync
        service.save_connection(
            endpoint_url="https://api.datacenter.internal/telemetry",
            auth_type="API_KEY_HEADER",
            api_key="secret-token-12345",
            json_path="readings"
        )
        sync_res = service.sync_data()
        self.assertTrue(sync_res["success"])
        self.assertEqual(sync_res["total_records"], 96)
        self.assertTrue(os.path.exists(sync_res["file_path"]))
        self.assertTrue(sync_res["suitability"]["schema_checks"]["target_detected"])

if __name__ == "__main__":
    unittest.main()
