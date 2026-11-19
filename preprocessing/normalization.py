"""
Normalization Module using Scikit-Learn MinMaxScaler
Prevents data leakage by fitting scalers strictly on training data.
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import MinMaxScaler
from preprocessing.data_loader import NUMERIC_COLUMNS

TARGET_COLUMN = "inlet_temperature"

class DataCenterScaler:
    """
    Manages feature and target Min-Max scalers with strict train-only fitting.
    """
    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self.feature_columns: List[str] = [c for c in NUMERIC_COLUMNS]
        self.target_column: str = TARGET_COLUMN
        self.is_fitted: bool = False
        
    def fit(self, train_df: pd.DataFrame) -> "DataCenterScaler":
        """
        Fit scalers strictly on training DataFrame.
        """
        features_data = train_df[self.feature_columns].values
        target_data = train_df[[self.target_column]].values
        
        self.feature_scaler.fit(features_data)
        self.target_scaler.fit(target_data)
        self.is_fitted = True
        return self
        
    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """Transform feature columns to [0, 1] range."""
        if not self.is_fitted:
            raise ValueError("DataCenterScaler must be fitted before transforming.")
        return self.feature_scaler.transform(df[self.feature_columns].values)
        
    def transform_target(self, df: pd.DataFrame) -> np.ndarray:
        """Transform target column to [0, 1] range."""
        if not self.is_fitted:
            raise ValueError("DataCenterScaler must be fitted before transforming.")
        return self.target_scaler.transform(df[[self.target_column]].values)
        
    def inverse_transform_target(self, scaled_target: np.ndarray) -> np.ndarray:
        """
        Invert scaled temperature predictions back to Celsius degrees.
        Supports 1D array, 2D array (samples, horizon), or flat lists.
        """
        if not self.is_fitted:
            raise ValueError("DataCenterScaler must be fitted before inverse transform.")
        orig_shape = scaled_target.shape
        flat = scaled_target.reshape(-1, 1)
        unscaled = self.target_scaler.inverse_transform(flat)
        return unscaled.reshape(orig_shape)

    def save(self, filepath: str = "models/saved/scaler.pkl"):
        """Save fitted scaler state to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "is_fitted": self.is_fitted
        }, filepath)
        
    @classmethod
    def load(cls, filepath: str = "models/saved/scaler.pkl") -> "DataCenterScaler":
        """Load fitted scaler state from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scaler file not found at: {filepath}")
        data = joblib.load(filepath)
        instance = cls()
        instance.feature_scaler = data["feature_scaler"]
        instance.target_scaler = data["target_scaler"]
        instance.feature_columns = data["feature_columns"]
        instance.target_column = data["target_column"]
        instance.is_fitted = data["is_fitted"]
        return instance
