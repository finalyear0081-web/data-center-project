"""
Data Cleaning and Validation Module
Detects duplicates, validates sensor physics boundaries, handles missing values.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from preprocessing.data_loader import NUMERIC_COLUMNS

# Sensor operational thresholds for validation
SENSOR_BOUNDS = {
    "inlet_temperature": (10.0, 45.0),
    "outlet_temperature": (15.0, 60.0),
    "hot_aisle_temperature": (15.0, 65.0),
    "cold_aisle_temperature": (10.0, 35.0),
    "humidity": (10.0, 95.0),
    "cpu_usage": (0.0, 100.0),
    "it_power_consumption": (50.0, 1000.0),
    "outdoor_temperature": (-20.0, 55.0),
    "cooling_power_consumption": (10.0, 500.0)
}

def validate_and_clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate, clean, deduplicate, and handle missing/invalid values.
    
    Args:
        df: Input DataFrame with parsed timestamp and numeric columns.
        
    Returns:
        Tuple of (cleaned_df, report_dict)
    """
    report = {
        "initial_rows": len(df),
        "duplicate_timestamps_removed": 0,
        "missing_values_imputed": {},
        "invalid_values_repaired": {}
    }
    
    # 1. Deduplicate timestamps (keep first occurrence)
    dupes = df["timestamp"].duplicated(keep="first")
    num_dupes = int(dupes.sum())
    report["duplicate_timestamps_removed"] = num_dupes
    if num_dupes > 0:
        df = df[~dupes].copy()
    else:
        df = df.copy()
        
    # 2. Check and repair out-of-boundary sensor readings
    for col, (min_val, max_val) in SENSOR_BOUNDS.items():
        if col in df.columns:
            invalid_mask = (df[col] < min_val) | (df[col] > max_val)
            invalid_count = int(invalid_mask.sum())
            report["invalid_values_repaired"][col] = invalid_count
            if invalid_count > 0:
                # Replace out of bounds with NaN then interpolate
                df.loc[invalid_mask, col] = np.nan

    # 3. Handle missing values via time-series linear interpolation
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            nan_count = int(df[col].isna().sum())
            report["missing_values_imputed"][col] = nan_count
            if nan_count > 0:
                df[col] = df[col].interpolate(method="linear").bfill().ffill()
                
    report["final_rows"] = len(df)
    return df, report
