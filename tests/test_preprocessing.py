"""
Unit Tests for Data Preprocessing and Leakage Prevention
"""
import unittest
import numpy as np
import pandas as pd
from backend.config import Config
from backend.preprocessing.data_loader import load_and_parse_csv
from backend.preprocessing.cleaning import clean_and_repair_dataset
from backend.preprocessing.normalization import DataCenterScaler
from backend.preprocessing.sequence_generator import split_chronologically, create_sliding_sequences

class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_df = load_and_parse_csv(Config.DEFAULT_DATASET_PATH)

    def test_cleaning_and_interpolation(self):
        """Test that missing readings are properly imputed."""
        sample = self.raw_df.head(200).copy()
        # Inject intentional NaNs and out-of-boundary values
        sample.loc[10:15, "inlet_temperature"] = np.nan
        sample.loc[25, "inlet_temperature"] = 999.0  # Physically impossible data center inlet temp
        
        cleaned_df, report = clean_and_repair_dataset(sample)
        self.assertEqual(len(cleaned_df), len(sample))
        self.assertEqual(cleaned_df["inlet_temperature"].isna().sum(), 0)
        self.assertLess(cleaned_df["inlet_temperature"].max(), 50.0)

    def test_chronological_split_ratios(self):
        """Verify strict chronological splits (70/15/15) without time overlap."""
        train_df, val_df, test_df = split_chronologically(self.raw_df, 0.70, 0.15, 0.15)
        
        n_total = len(self.raw_df)
        self.assertEqual(len(train_df), int(n_total * 0.70))
        self.assertAlmostEqual(len(val_df), int(n_total * 0.15), delta=2)
        
        # Test monotonicity across sets
        self.assertTrue(train_df["timestamp"].iloc[-1] < val_df["timestamp"].iloc[0])
        self.assertTrue(val_df["timestamp"].iloc[-1] < test_df["timestamp"].iloc[0])

    def test_train_only_scaling_no_leakage(self):
        """Verify scaler fits exclusively on training split."""
        train_df, val_df, _ = split_chronologically(self.raw_df)
        scaler = DataCenterScaler()
        scaler.fit(train_df)
        
        train_scaled = scaler.transform_target(train_df)
        self.assertAlmostEqual(train_scaled.min(), 0.0, places=2)
        self.assertAlmostEqual(train_scaled.max(), 1.0, places=2)
        
        # Invert target back to original Celsius
        unscaled = scaler.inverse_transform_target(train_scaled)
        np.testing.assert_allclose(unscaled, train_df[["inlet_temperature"]].values, rtol=1e-4)

    def test_sliding_window_dimensions(self):
        """Verify sliding sequence output shapes for 288 input steps and 288 horizon steps."""
        feats = np.random.rand(800, 9).astype(np.float32)
        targ = np.random.rand(800, 1).astype(np.float32)
        
        X, y = create_sliding_sequences(feats, targ, input_window=288, forecast_horizon=288, stride=10)
        expected_samples = (800 - 576) // 10 + 1
        self.assertEqual(X.shape, (expected_samples, 288, 9))
        self.assertEqual(y.shape, (expected_samples, 288))

if __name__ == "__main__":
    unittest.main()
