"""
Data Loader and Parsing Module
Supports multi-schema ingestion:
- HazardNet Operational Telemetry (calendar timestamps)
- AAU Experimental Data (experiment_id, elapsed_time_minutes)
"""
import os
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from backend.preprocessing.validation import detect_schema, HAZARDNET_NUMERIC_COLUMNS
from backend.preprocessing.file_parser import parse_dataset_file
from backend.preprocessing.canonical_schema import (
    normalize_dataframe_to_canonical,
    CanonicalSchemaError,
    CANONICAL_TARGET_KEY,
    CANONICAL_REGISTRY
)

def load_and_parse_csv(file_path: str, manual_overrides: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Loads dataset file (CSV, XLSX, XLS, JSON), applies centralized canonical normalization,
    and structures data cleanly for downstream processing.
    Guarantees 'inlet_temperature' is the prediction target.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".xlsx", ".xls", ".json"]:
        df = parse_dataset_file(file_path)
    else:
        df = pd.read_csv(file_path)
        
    df_canon, mapped_features, unmapped = normalize_dataframe_to_canonical(
        df, manual_overrides=manual_overrides
    )
    return df_canon

def get_detailed_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes comprehensive numerical statistics for all active numeric sensor features."""
    stats = {}
    schema_type, _ = detect_schema(df)
    
    candidate_cols = list(set(
        HAZARDNET_NUMERIC_COLUMNS + 
        [
            "inlet_temperature_c", "outlet_temperature_c",
            "cooling_supply_temperature_c", "cooling_return_temperature_c",
            "it_power_p_it", "facility_power_p_dc", "PUE", "CoP",
            "inlet_temperature", "outlet_temperature", "it_power_consumption",
            "hot_aisle_temperature", "cold_aisle_temperature", "humidity",
            "cpu_usage", "facility_power_consumption", "cooling_power_consumption"
        ]
    ))

    for col in candidate_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if not s.empty:
                stats[col] = {
                    "mean": round(float(s.mean()), 2),
                    "std": round(float(s.std()), 2),
                    "min": round(float(s.min()), 2),
                    "25%": round(float(s.quantile(0.25)), 2),
                    "50%": round(float(s.median()), 2),
                    "75%": round(float(s.quantile(0.75)), 2),
                    "max": round(float(s.max()), 2)
                }

    if schema_type == "AAU_EXPERIMENTAL":
        if "inlet_temperature_c" in stats:
            stats["inlet_temperature"] = stats["inlet_temperature_c"]
        if "outlet_temperature_c" in stats:
            stats["outlet_temperature"] = stats["outlet_temperature_c"]
        if "it_power_p_it" in stats:
            stats["it_power_consumption"] = stats["it_power_p_it"]
    return stats
