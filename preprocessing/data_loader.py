"""
Data Loader Module for Data Center Sensor Readings
Loads, validates, and parses historical time-series CSV datasets.
"""
import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

EXPECTED_COLUMNS: List[str] = [
    "timestamp",
    "inlet_temperature",
    "outlet_temperature",
    "hot_aisle_temperature",
    "cold_aisle_temperature",
    "humidity",
    "cpu_usage",
    "it_power_consumption",
    "outdoor_temperature",
    "cooling_power_consumption"
]

NUMERIC_COLUMNS: List[str] = [col for col in EXPECTED_COLUMNS if col != "timestamp"]

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load CSV dataset from path, parse timestamp, and sort chronologically.
    
    Args:
        file_path: Path to dataset CSV.
        
    Returns:
        pd.DataFrame sorted chronologically by timestamp.
        
    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If required columns are missing.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Validate columns
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
        
    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Ensure numeric columns are cast to float
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Sort chronologically
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    return df

def get_dataset_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract comprehensive summary statistics, missing values, and metadata.
    """
    missing_counts = df.isnull().sum().to_dict()
    stats = {}
    for col in NUMERIC_COLUMNS:
        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "25%": float(df[col].quantile(0.25)),
            "50%": float(df[col].median()),
            "75%": float(df[col].quantile(0.75)),
            "max": float(df[col].max())
        }
        
    time_diffs = df["timestamp"].diff().dropna()
    sampling_interval_minutes = float(time_diffs.median().total_seconds() / 60.0) if not time_diffs.empty else 5.0
    
    return {
        "total_records": len(df),
        "columns": list(df.columns),
        "start_time": str(df["timestamp"].iloc[0]) if not df.empty else None,
        "end_time": str(df["timestamp"].iloc[-1]) if not df.empty else None,
        "sampling_interval_minutes": sampling_interval_minutes,
        "missing_values": missing_counts,
        "duplicate_timestamps": int(df["timestamp"].duplicated().sum()),
        "statistics": stats,
        "is_synthetic": True,
        "description": "Synthetic simulated sensor data for academic and evaluation purposes."
    }
