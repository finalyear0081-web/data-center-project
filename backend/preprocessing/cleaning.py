"""
Data Cleaning and Imputation Module
Cleans duplicates and repairs physical boundaries while strictly respecting
experiment boundaries in multi-experiment datasets (AAU).
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from backend.preprocessing.validation import detect_schema, PHYSICAL_SENSOR_BOUNDS

def clean_and_repair_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans duplicates, repairs physical sensor boundaries, and interpolates missing readings.
    Preserves experimental boundaries: never concatenates or interpolates across experiment_id.
    """
    report = {
        "initial_rows": len(df),
        "duplicates_removed": 0,
        "out_of_bounds_repaired": {},
        "missing_imputed": {},
        "final_rows": 0
    }
    
    schema_type, _ = detect_schema(df)
    
    # 1. Deduplication
    if schema_type == "AAU_EXPERIMENTAL" and "experiment_id" in df.columns:
        dupes = df.duplicated(subset=["experiment_id", "elapsed_time_minutes"], keep="first")
    elif "timestamp" in df.columns:
        dupes = df["timestamp"].duplicated(keep="first")
    else:
        dupes = df.duplicated(keep="first")
        
    num_dupes = int(dupes.sum())
    report["duplicates_removed"] = num_dupes
    if num_dupes > 0:
        df = df[~dupes].copy()
    else:
        df = df.copy()

    # 2. Sort chronologically (within experiment if multi-experiment)
    if schema_type == "AAU_EXPERIMENTAL" and "experiment_id" in df.columns:
        df = df.sort_values(by=["experiment_id", "elapsed_time_minutes"]).reset_index(drop=True)
    elif "timestamp" in df.columns:
        df = df.sort_values(by="timestamp").reset_index(drop=True)

    # 3. Check and flag out-of-boundary sensor readings
    for col, (min_val, max_val) in PHYSICAL_SENSOR_BOUNDS.items():
        if col in df.columns:
            # Only check if column has valid numeric data
            valid_mask = df[col].notna()
            if valid_mask.any():
                invalid_mask = valid_mask & ((df[col] < min_val) | (df[col] > max_val))
                invalid_count = int(invalid_mask.sum())
                report["out_of_bounds_repaired"][col] = invalid_count
                if invalid_count > 0:
                    df.loc[invalid_mask, col] = np.nan

    # 4. Handle sporadic missing values via linear interpolation strictly WITHIN experiments
    # Note: Do not interpolate or zero-fill columns that are 100% missing (e.g. unmeasured sensors in AAU)
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["elapsed_time_minutes"]]
    
    if schema_type == "AAU_EXPERIMENTAL" and "experiment_id" in df.columns:
        # Clean per experiment
        cleaned_groups = []
        for exp_id, exp_group in df.groupby("experiment_id", sort=False):
            exp_group = exp_group.copy()
            for col in numeric_cols:
                nan_count = int(exp_group[col].isna().sum())
                # Only interpolate if there are some non-null values (do not invent data for unmeasured features)
                non_null_count = int(exp_group[col].notna().sum())
                if 0 < nan_count < len(exp_group) and non_null_count > 1:
                    exp_group[col] = exp_group[col].interpolate(method="linear").bfill().ffill()
                    report["missing_imputed"][col] = report["missing_imputed"].get(col, 0) + nan_count
            cleaned_groups.append(exp_group)
        df = pd.concat(cleaned_groups, ignore_index=True)
    else:
        for col in numeric_cols:
            nan_count = int(df[col].isna().sum())
            non_null_count = int(df[col].notna().sum())
            if 0 < nan_count < len(df) and non_null_count > 1:
                df[col] = df[col].interpolate(method="linear").bfill().ffill()
                report["missing_imputed"][col] = nan_count

    report["final_rows"] = len(df)
    return df, report
