"""
Dataset Validation Module for Industrial and Experimental Data Center Telemetry
Supports multiple schemas:
1. HazardNet Operational Telemetry Schema (calendar timestamp, inlet_temperature target)
2. AAU Experimental Data Schema (experiment_id, elapsed_time_minutes, inlet_temperature_c target)
"""
import os
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

# ==========================================
# Schema Definitions
# ==========================================

# 1. HazardNet / Standard Operational Schema
HAZARDNET_REQUIRED_COLUMNS: List[str] = [
    "timestamp",
    "inlet_temperature",
    "outlet_temperature",
    "it_power_consumption"
]

HAZARDNET_NUMERIC_COLUMNS: List[str] = [
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

# 2. AAU Experimental Schema
AAU_REQUIRED_COLUMNS: List[str] = [
    "experiment_id",
    "elapsed_time_minutes",
    "inlet_temperature_c"
]

AAU_OPTIONAL_AVAILABLE_COLUMNS: List[str] = [
    "outlet_temperature_c",
    "cooling_supply_temperature_c",
    "cooling_return_temperature_c",
    "it_power_p_it",
    "facility_power_p_dc",
    "PUE",
    "CoP"
]

AAU_UNAVAILABLE_COLUMNS: List[str] = [
    "humidity",
    "cpu_usage",
    "hot_aisle_temperature_c",
    "cold_aisle_temperature_c",
    "outdoor_temperature_c",
    "cooling_power_consumption"
]

# Combined known columns
EXPECTED_COLUMNS = ["timestamp"] + HAZARDNET_NUMERIC_COLUMNS
REQUIRED_CORE_COLUMNS = HAZARDNET_REQUIRED_COLUMNS
NUMERIC_COLUMNS = HAZARDNET_NUMERIC_COLUMNS
TARGET_COLUMN = "inlet_temperature"

# Physical domain thresholds
PHYSICAL_SENSOR_BOUNDS = {
    # HazardNet
    "inlet_temperature": (10.0, 50.0),
    "outlet_temperature": (15.0, 75.0),
    "hot_aisle_temperature": (15.0, 65.0),
    "cold_aisle_temperature": (10.0, 35.0),
    "humidity": (10.0, 95.0),
    "cpu_usage": (0.0, 100.0),
    "it_power_consumption": (0.1, 5000.0),
    "outdoor_temperature": (-25.0, 55.0),
    "cooling_power_consumption": (0.1, 600.0),
    # AAU
    "inlet_temperature_c": (10.0, 50.0),
    "outlet_temperature_c": (15.0, 75.0),
    "cooling_supply_temperature_c": (5.0, 45.0),
    "cooling_return_temperature_c": (10.0, 50.0),
    "it_power_p_it": (0.1, 5000.0),
    "facility_power_p_dc": (0.1, 10000.0),
    "PUE": (1.0, 15.0),
    "CoP": (0.05, 25.0)
}

from backend.preprocessing.canonical_schema import (
    find_canonical_match,
    normalize_column_name,
    CANONICAL_REGISTRY,
    CANONICAL_TARGET_KEY
)

def find_matching_source_column(columns: List[str], canonical_key: str) -> Optional[str]:
    """Finds a matching raw column name for a given canonical attribute using normalized aliases."""
    for col in columns:
        if find_canonical_match(col) == canonical_key:
            return col
    return None

def detect_schema(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Detects whether dataframe matches AAU Experimental Schema or Operational Telemetry Schema.
    Returns: Tuple of (schema_key, schema_display_name)
    """
    cols = list(df.columns)
    has_exp = find_matching_source_column(cols, "experiment_id") is not None
    has_elapsed = find_matching_source_column(cols, "elapsed_time_minutes") is not None
    has_ts = find_matching_source_column(cols, "timestamp") is not None

    if has_exp and ("experiment_id" in df.columns) and df["experiment_id"].nunique() > 1:
        return "AAU_EXPERIMENTAL", "AAU Experimental Data"
    if "inlet_temperature_c" in cols and has_elapsed and not has_ts:
        return "AAU_EXPERIMENTAL", "AAU Experimental Data"
    if has_ts or "node_id" in cols or "timestamp" in cols:
        if "node_id" in cols or "hazardnet" in str(cols).lower():
            return "HAZARDNET_OPERATIONAL", "HazardNet Operational Telemetry"
        return "HAZARDNET_OPERATIONAL", "Universal Operational Telemetry"
    if has_exp or has_elapsed:
        return "AAU_EXPERIMENTAL", "AAU Experimental Data"
    return "HAZARDNET_OPERATIONAL", "Universal Operational Telemetry"

def detect_sampling_interval_and_horizon(df: pd.DataFrame, schema: Optional[str] = None) -> Tuple[int, int]:
    """
    Detects sampling interval in minutes and valid forecast horizon step count.
    - 24-hour horizon for continuous time series (144 steps at 10m, 288 steps at 5m, 720 steps at 2m).
    - For multi-experiment datasets with short trials (e.g. 720 records per experiment):
      supported validated horizon is 30 steps (1 hour) to preserve chronological history.
    """
    if schema is None:
        schema, _ = detect_schema(df)

    # 1. Determine sampling interval
    interval_mins = 5
    if "elapsed_time_minutes" in df.columns:
        diffs = df["elapsed_time_minutes"].diff().dropna()
        diffs = diffs[diffs > 0]
        if not diffs.empty:
            interval_mins = max(1, int(round(diffs.median())))
        else:
            interval_mins = 2
    elif "timestamp" in df.columns and len(df) >= 2:
        try:
            timestamps = pd.to_datetime(df["timestamp"])
            diffs = timestamps.diff().dropna()
            if not diffs.empty:
                med_sec = diffs.median().total_seconds()
                interval_mins = max(1, int(round(med_sec / 60.0)))
        except Exception:
            interval_mins = 5

    # 2. Determine forecast horizon steps
    target_24h_steps = max(1, int(round(1440.0 / interval_mins)))

    # Check whether dataset has multiple short experiments (e.g. AAU with 720 records per exp)
    if "experiment_id" in df.columns and df["experiment_id"].nunique() > 1:
        min_exp_len = df.groupby("experiment_id").size().min()
        if min_exp_len < 1000:
            # Short-run experiment benchmark: validated 1-hour horizon (30 steps at 2m)
            return interval_mins, max(1, int(round(60.0 / interval_mins)))

    return interval_mins, target_24h_steps

def validate_csv_dataset(file_path: str) -> Dict[str, Any]:
    """
    Comprehensive multi-schema validation of an uploaded CSV dataset file.
    Supports both HazardNet Operational Telemetry and AAU Experimental Data.
    """
    report = {
        "is_valid": False,
        "filename": os.path.basename(file_path),
        "file_size_bytes": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
        "schema_type": "UNKNOWN",
        "schema_name": "Unknown",
        "errors": [],
        "warnings": [],
        "total_records": 0,
        "total_experiments": 1,
        "experiment_ids": [],
        "columns": [],
        "available_features": [],
        "unavailable_features": [],
        "missing_columns": [],
        "unexpected_columns": [],
        "sampling_interval_minutes": 5,
        "forecast_horizon_steps": 288,
        "forecast_horizon_hours": 24.0,
        "input_window_steps": 288,
        "available_history_per_experiment": 0,
        "duplicate_timestamps": 0,
        "duplicate_records": 0,
        "missing_values": {},
        "out_of_bounds_counts": {},
        "is_chronological": True,
        "start_time": None,
        "end_time": None,
        "preview_records": []
    }

    if not os.path.exists(file_path):
        report["errors"].append(f"File not found on disk: {file_path}")
        return report

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        report["errors"].append(f"Invalid CSV structure or syntax error: {str(e)}")
        return report

    report["total_records"] = len(df)
    report["columns"] = list(df.columns)

    if len(df) < 50:
        report["errors"].append(f"Dataset has only {len(df)} records. Minimum of 50 records required.")

    # 1. Detect Schema
    schema_type, schema_name = detect_schema(df)
    report["schema_type"] = schema_type
    report["schema_name"] = schema_name

    if schema_type == "AAU_EXPERIMENTAL":
        _validate_aau_schema(df, report)
    elif schema_type == "HAZARDNET_OPERATIONAL":
        _validate_hazardnet_schema(df, report)
    else:
        missing_hz = [c for c in HAZARDNET_REQUIRED_COLUMNS if c not in df.columns]
        missing_aau = [c for c in AAU_REQUIRED_COLUMNS if c not in df.columns]
        report["missing_columns"] = missing_hz if len(missing_hz) <= len(missing_aau) else missing_aau
        report["errors"].append(
            "Dataset does not match any supported schema. "
            f"Expected HazardNet columns: {HAZARDNET_REQUIRED_COLUMNS} OR "
            f"AAU Experimental columns: {AAU_REQUIRED_COLUMNS}."
        )

    report["is_valid"] = (len(report["errors"]) == 0)
    return report

def _validate_aau_schema(df: pd.DataFrame, report: Dict[str, Any]):
    """Validation logic specifically tailored to AAU Experimental Data Schema."""
    cols = df.columns.tolist()
    
    # 1. Verify required columns using alias-aware matching
    target_col = find_matching_source_column(cols, "inlet_temperature")
    exp_col = find_matching_source_column(cols, "experiment_id")
    elapsed_col = find_matching_source_column(cols, "elapsed_time_minutes")
    
    missing_req = []
    if not target_col:
        missing_req.append("inlet_temperature")
    if not exp_col:
        missing_req.append("experiment_id")
    if not elapsed_col:
        missing_req.append("elapsed_time_minutes")

    if missing_req:
        report["errors"].append(
            f"Missing required columns: {missing_req}. "
            f"Supported inlet temperature aliases: {CANONICAL_REGISTRY['inlet_temperature']['aliases']}."
        )
        return

    # 2. Detect and preserve experiments
    exp_series = df[exp_col].dropna().astype(str)
    unique_exps = exp_series.unique().tolist()
    report["total_experiments"] = len(unique_exps)
    report["experiment_ids"] = unique_exps
    
    # 3. Features available vs unavailable
    avail = []
    unavail = []
    for c in AAU_REQUIRED_COLUMNS + AAU_OPTIONAL_AVAILABLE_COLUMNS + AAU_UNAVAILABLE_COLUMNS:
        match_col = find_matching_source_column(cols, c) or (c if c in cols else None)
        if match_col:
            non_null_cnt = df[match_col].dropna().count()
            if non_null_cnt > 0:
                avail.append(c)
            else:
                unavail.append(c)
        else:
            unavail.append(c)

    report["available_features"] = list(dict.fromkeys(avail))
    report["unavailable_features"] = list(dict.fromkeys(unavail))

    # 4. Sampling interval & horizons
    interval_mins, horizon_steps = detect_sampling_interval_and_horizon(df, schema="AAU_EXPERIMENTAL")
    report["sampling_interval_minutes"] = interval_mins
    report["forecast_horizon_steps"] = horizon_steps
    report["forecast_horizon_hours"] = round((horizon_steps * interval_mins) / 60.0, 1)
    report["input_window_steps"] = 60  # 2 hours history
    
    # Average available history per experiment
    counts_per_exp = df.groupby(exp_col).size()
    min_exp_len = int(counts_per_exp.min()) if not counts_per_exp.empty else 0
    report["available_history_per_experiment"] = min_exp_len

    # 5. Check sampling intervals and monotonicity per experiment
    is_all_chronological = True
    total_dupe_time = 0
    for exp_id, exp_df in df.groupby(exp_col):
        diffs = exp_df[elapsed_col].diff().dropna()
        if not diffs.empty and not (diffs > 0).all():
            is_all_chronological = False
            report["warnings"].append(f"Experiment '{exp_id}' has non-monotonic elapsed times.")
        dupes = exp_df[elapsed_col].duplicated().sum()
        total_dupe_time += int(dupes)

    report["is_chronological"] = is_all_chronological
    report["duplicate_timestamps"] = total_dupe_time
    report["start_time"] = f"0 min (Exp: {unique_exps[0]})" if unique_exps else "0 min"
    max_elapsed = df[elapsed_col].max()
    report["end_time"] = f"{int(max_elapsed)} min ({report['total_experiments']} experiments)"

    # 6. Check numeric bounds and missing values on available features
    for col in report["available_features"]:
        if col in [exp_col, "experiment_id"]:
            continue
        actual_col = find_matching_source_column(cols, col) or col
        if actual_col in df.columns:
            num_s = pd.to_numeric(df[actual_col], errors="coerce")
            nan_count = int(num_s.isna().sum())
            report["missing_values"][col] = nan_count
            
            # Domain bounds
            min_b, max_b = PHYSICAL_SENSOR_BOUNDS.get(col, (-np.inf, np.inf))
            valid_vals = num_s.dropna()
            oob_count = int(((valid_vals < min_b) | (valid_vals > max_b)).sum())
            report["out_of_bounds_counts"][col] = oob_count
            if oob_count > 0:
                report["warnings"].append(f"Column '{actual_col}' has {oob_count} readings outside bounds [{min_b}, {max_b}].")

    # 7. Preview records (first 10 records)
    preview_cols = [c for c in [exp_col, elapsed_col, target_col] if c and c in df.columns]
    report["preview_records"] = df[preview_cols].head(10).to_dict(orient="records")

def _validate_hazardnet_schema(df: pd.DataFrame, report: Dict[str, Any]):
    """Validation logic for HazardNet / Operational Telemetry Schema."""
    cols = df.columns.tolist()
    target_col = find_matching_source_column(cols, "inlet_temperature")
    time_col = find_matching_source_column(cols, "timestamp")
    
    missing_core = []
    if not target_col:
        missing_core.append("inlet_temperature")
    if not time_col:
        missing_core.append("timestamp")

    report["missing_columns"] = missing_core
    if missing_core:
        report["errors"].append(
            f"Missing required sensor columns: {missing_core}. "
            f"Supported inlet temperature aliases: {CANONICAL_REGISTRY['inlet_temperature']['aliases']}."
        )
        return

    report["total_experiments"] = 1
    report["experiment_ids"] = ["marconi_a2_aggregate"]
    
    avail = [c for c in df.columns if c in HAZARDNET_NUMERIC_COLUMNS or c == time_col]
    unavail = [c for c in HAZARDNET_NUMERIC_COLUMNS if c not in df.columns and find_matching_source_column(df.columns, c) is None]
    report["available_features"] = avail
    report["unavailable_features"] = unavail

    # Monotonic timestamps
    try:
        parsed_ts = pd.to_datetime(df[time_col], errors="coerce")
        invalid_ts_count = int(parsed_ts.isna().sum())
        if invalid_ts_count > 0:
            report["errors"].append(f"Found {invalid_ts_count} unparseable timestamp values.")
        else:
            is_sorted = parsed_ts.is_monotonic_increasing
            report["is_chronological"] = bool(is_sorted)
            report["start_time"] = str(parsed_ts.min())
            report["end_time"] = str(parsed_ts.max())
            report["duplicate_timestamps"] = int(parsed_ts.duplicated().sum())

            interval_mins, horizon_steps = detect_sampling_interval_and_horizon(df, schema="HAZARDNET_OPERATIONAL")
            report["sampling_interval_minutes"] = interval_mins
            report["forecast_horizon_steps"] = horizon_steps
            report["forecast_horizon_hours"] = 24.0
            report["input_window_steps"] = horizon_steps
            report["available_history_per_experiment"] = len(df)
    except Exception as e:
        report["errors"].append(f"Failed to process timestamp column: {str(e)}")

    # Duplicate records
    present_cols = [c for c in HAZARDNET_NUMERIC_COLUMNS if c in df.columns]
    report["duplicate_records"] = int(df.duplicated(subset=present_cols).sum()) if present_cols else 0

    # Numeric bounds
    for col in HAZARDNET_NUMERIC_COLUMNS:
        match_col = find_matching_source_column(cols, col) or col
        if match_col in df.columns:
            num_s = pd.to_numeric(df[match_col], errors="coerce")
            nan_count = int(num_s.isna().sum())
            report["missing_values"][col] = nan_count
            min_b, max_b = PHYSICAL_SENSOR_BOUNDS.get(col, (-np.inf, np.inf))
            valid_vals = num_s.dropna()
            oob_count = int(((valid_vals < min_b) | (valid_vals > max_b)).sum())
            report["out_of_bounds_counts"][col] = oob_count

    preview_cols = [c for c in df.columns if c in HAZARDNET_NUMERIC_COLUMNS or c in [time_col, "node_id"]]
    report["preview_records"] = df[preview_cols].head(10).to_dict(orient="records")
