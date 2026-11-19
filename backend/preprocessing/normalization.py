"""
Min-Max Normalization Module
Prevents data leakage by fitting scalers strictly on chronological training data.
Supports multi-schema sensor features (HazardNet and AAU).
Guarantees index 0 of features is always the target temperature for residual LSTM skip connections.
"""
import os
import joblib
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from backend.preprocessing.validation import (
    detect_schema,
    HAZARDNET_NUMERIC_COLUMNS,
    AAU_REQUIRED_COLUMNS,
    AAU_OPTIONAL_AVAILABLE_COLUMNS
)

class DataCenterScaler:
    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self.feature_columns: List[str] = []
        self.target_column: str = "inlet_temperature"
        self.schema_type: str = "UNKNOWN"
        self.is_fitted: bool = False
        
    def fit(self, train_df: pd.DataFrame, target_column: Optional[str] = None) -> "DataCenterScaler":
        """
        Fit feature and target scalers strictly on training data.
        Canonical target is ALWAYS 'inlet_temperature' at feature column index 0.
        Dynamically selects only valid available features without synthesizing missing ones.
        """
        self.schema_type, _ = detect_schema(train_df)
        df_work = self._align_dataframe(train_df)
        
        self.target_column = "inlet_temperature"
        if self.target_column not in df_work.columns:
            raise ValueError(
                f"DataCenterScaler fit error: Target '{self.target_column}' could not be identified in training columns."
            )

        candidate_features = [
            "inlet_temperature",
            "outlet_temperature",
            "hot_aisle_temperature",
            "cold_aisle_temperature",
            "it_power_consumption",
            "facility_power_consumption",
            "cooling_power_consumption",
            "cooling_supply_temperature",
            "cooling_return_temperature",
            "humidity",
            "cpu_usage",
            "PUE",
            "CoP"
        ]

        # Feature 0 is ALWAYS the canonical target
        feature_cols = [self.target_column]
        for c in candidate_features:
            if c != self.target_column and c in df_work.columns:
                valid_cnt = pd.to_numeric(df_work[c], errors="coerce").notna().sum()
                if valid_cnt > 0:
                    feature_cols.append(c)

        self.feature_columns = feature_cols

        features_data = df_work[self.feature_columns].values
        target_data = df_work[[self.target_column]].values

        self.feature_scaler.fit(features_data)
        self.target_scaler.fit(target_data)
        self.is_fitted = True
        return self

    def _align_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aligns incoming dataframe columns with canonical features and scaler's fitted feature columns."""
        aligned = df.copy()
        
        # Check all column aliases to populate canonical columns
        alias_mapping = {
            "inlet_temperature": [
                "inlet_temperature", "inlet_temperature_c", "inlet_temp", "inlet_temp_c",
                "server_inlet_temp", "inlet temp avg", "inlet_temp_avg", "inlet", "t_inlet",
                "temp_inlet", "rack_inlet_temp", "inlet_c"
            ],
            "outlet_temperature": [
                "outlet_temperature", "outlet_temperature_c", "outlet_temp", "outlet_temp_c",
                "server_outlet_temp", "outlet temp avg", "outlet_temp_avg", "outlet_avg_temp",
                "outlet", "t_outlet", "temp_outlet", "rack_outlet_temp", "outlet_c"
            ],
            "it_power_consumption": [
                "it_power_consumption", "it_power_p_it", "p_it", "it_power", "it_power_kw",
                "power_it", "it_load", "server_power", "it_load_kw"
            ],
            "facility_power_consumption": [
                "facility_power_consumption", "facility_power_p_dc", "p_dc", "facility_power",
                "power_dc", "total_power", "total_facility_power", "facility_load_kw", "total_power_kw"
            ],
            "cooling_supply_temperature": [
                "cooling_supply_temperature", "cooling_supply_temperature_c", "cooling_supply_average",
                "cooling supply average", "chilled_water_supply", "cws_temp", "cooling_supply_temp"
            ],
            "cooling_return_temperature": [
                "cooling_return_temperature", "cooling_return_temperature_c", "cooling_return_average",
                "cooling return average", "chilled_water_return", "cwr_temp", "cooling_return_temp"
            ],
            "cooling_power_consumption": [
                "cooling_power_consumption", "cooling_power", "chiller_power", "crac_power", "crah_power", "hvac_power"
            ],
            "hot_aisle_temperature": [
                "hot_aisle_temperature", "hot_aisle_temperature_c", "hot_aisle_temp", "hot_aisle", "exhaust_temp"
            ],
            "cold_aisle_temperature": [
                "cold_aisle_temperature", "cold_aisle_temperature_c", "cold_aisle_temp", "cold_aisle", "supply_air_temp"
            ],
            "cpu_usage": [
                "cpu_usage", "cpu_utilization", "cpu_load", "cpu_percent", "processor_load", "cpu"
            ],
            "humidity": [
                "humidity", "relative_humidity", "rh", "humidity_percent", "rh_percent"
            ]
        }

        col_lookup = {str(c).strip().lower().replace(" ", "_"): c for c in aligned.columns}

        for canon_col, aliases in alias_mapping.items():
            if canon_col not in aligned.columns:
                for a in aliases:
                    norm_a = a.lower().replace(" ", "_")
                    if norm_a in col_lookup:
                        aligned[canon_col] = pd.to_numeric(aligned[col_lookup[norm_a]], errors="coerce")
                        break
            else:
                aligned[canon_col] = pd.to_numeric(aligned[canon_col], errors="coerce")

        # Also support reverse mapping so legacy code referencing aliases won't fail
        if "inlet_temperature" in aligned.columns and "inlet_temperature_c" not in aligned.columns:
            aligned["inlet_temperature_c"] = aligned["inlet_temperature"]
        if "outlet_temperature" in aligned.columns and "outlet_temperature_c" not in aligned.columns:
            aligned["outlet_temperature_c"] = aligned["outlet_temperature"]
        if "it_power_consumption" in aligned.columns and "it_power_p_it" not in aligned.columns:
            aligned["it_power_p_it"] = aligned["it_power_consumption"]
        if "facility_power_consumption" in aligned.columns and "facility_power_p_dc" not in aligned.columns:
            aligned["facility_power_p_dc"] = aligned["facility_power_consumption"]
            aligned["facility_power"] = aligned["facility_power_consumption"]

        # If already fitted, ensure all required feature columns exist (fill with 0.0 if optional was missing)
        if self.is_fitted:
            for col in self.feature_columns:
                if col not in aligned.columns:
                    aligned[col] = 0.0

        return aligned

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("DataCenterScaler must be fitted before transforming features.")
        aligned = self._align_dataframe(df)
        return self.feature_scaler.transform(aligned[self.feature_columns].values)

    def transform_target(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("DataCenterScaler must be fitted before transforming target.")
        aligned = self._align_dataframe(df)
        return self.target_scaler.transform(aligned[[self.target_column]].values)

    def inverse_transform_target(self, scaled_target: np.ndarray) -> np.ndarray:
        """Invert scaled temperature predictions back to degrees Celsius (°C)."""
        if not self.is_fitted:
            raise ValueError("DataCenterScaler must be fitted before inverse transforming.")
        orig_shape = scaled_target.shape
        flat = scaled_target.reshape(-1, 1)
        unscaled = self.target_scaler.inverse_transform(flat)
        return unscaled.reshape(orig_shape)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "schema_type": self.schema_type,
            "is_fitted": self.is_fitted
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "DataCenterScaler":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scaler file not found at: {filepath}")
        data = joblib.load(filepath)
        instance = cls()
        instance.feature_scaler = data["feature_scaler"]
        instance.target_scaler = data["target_scaler"]
        instance.feature_columns = data["feature_columns"]
        instance.target_column = data["target_column"]
        instance.schema_type = data.get("schema_type", "UNKNOWN")
        instance.is_fitted = data["is_fitted"]
        return instance
