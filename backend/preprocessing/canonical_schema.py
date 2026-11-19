"""
Centralized Canonical Schema Registry & Column Normalization Module
Standardizes all sensor telemetry attributes across arbitrary naming conventions.
Guarantees consistent internal representation:
- Target: 'inlet_temperature'
- Optional features: 'outlet_temperature', 'hot_aisle_temperature', 'cold_aisle_temperature',
  'it_power_consumption', 'facility_power_consumption', 'cooling_power_consumption',
  'cooling_supply_temperature', 'cooling_return_temperature', 'humidity', 'cpu_usage'
- Optional indices: 'timestamp' (calendar datetime), 'elapsed_time_minutes' (numeric elapsed),
  'experiment_id' (batch identifier)
"""
import re
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

class CanonicalSchemaError(ValueError):
    """Raised when dataset fails canonical schema normalization."""
    def __init__(self, message: str, required_feature: str, detected_columns: List[str], suggested_aliases: List[str]):
        super().__init__(message)
        self.required_feature = required_feature
        self.detected_columns = detected_columns
        self.suggested_aliases = suggested_aliases

def normalize_column_name(col_name: str) -> str:
    """
    Normalizes a column name using lowercase, whitespace trimming,
    and consistent underscore formatting.
    Examples:
      'Inlet Temp Avg' -> 'inlet_temp_avg'
      'inlet_temperature_c' -> 'inlet_temperature_c'
      'P_IT' -> 'p_it'
      'Date/Time' -> 'date_time'
    """
    s = str(col_name).strip().lower()
    # Replace slashes, dashes, dots, spaces with underscores
    s = re.sub(r"[\s\-\.\/]+", "_", s)
    # Strip any characters that are not alphanumeric or underscore
    s = re.sub(r"[^\w_]", "", s)
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

# Centralized canonical registry with all supported aliases
CANONICAL_TARGET_KEY = "inlet_temperature"

CANONICAL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "inlet_temperature": {
        "display_name": "Server Inlet Temperature (Target)",
        "unit": "°C",
        "is_target": True,
        "required": True,
        "expected_range": (10.0, 50.0),
        "aliases": [
            "inlet_temperature",
            "inlet_temperature_c",
            "inlet_temp",
            "inlet_temp_c",
            "server_inlet_temp",
            "inlet_temp_avg",
            "inlet_air_temp",
            "inlet",
            "t_inlet",
            "t_in",
            "temp_inlet",
            "temp_in",
            "rack_inlet_temp",
            "rack_inlet_temperature",
            "inlet_c",
            "temperature_inlet",
            "temperature"
        ]
    },
    "outlet_temperature": {
        "display_name": "Server Outlet Temperature",
        "unit": "°C",
        "is_target": False,
        "required": False,
        "expected_range": (15.0, 75.0),
        "aliases": [
            "outlet_temperature",
            "outlet_temperature_c",
            "outlet_temp",
            "outlet_temp_c",
            "server_outlet_temp",
            "outlet_temp_avg",
            "outlet_avg_temp",
            "outlet_air_temp",
            "outlet",
            "t_outlet",
            "t_out",
            "temp_outlet",
            "temp_out",
            "rack_outlet_temp",
            "rack_outlet_temperature",
            "outlet_c",
            "temperature_outlet"
        ]
    },
    "hot_aisle_temperature": {
        "display_name": "Hot Aisle Air Temperature",
        "unit": "°C",
        "is_target": False,
        "required": False,
        "expected_range": (15.0, 65.0),
        "aliases": [
            "hot_aisle_temperature",
            "hot_aisle_temperature_c",
            "hot_aisle_temp",
            "hot_aisle",
            "t_hot_aisle",
            "exhaust_temp",
            "hot_aisle_c",
            "exhaust_temperature"
        ]
    },
    "cold_aisle_temperature": {
        "display_name": "Cold Aisle Air Temperature",
        "unit": "°C",
        "is_target": False,
        "required": False,
        "expected_range": (10.0, 35.0),
        "aliases": [
            "cold_aisle_temperature",
            "cold_aisle_temperature_c",
            "cold_aisle_temp",
            "cold_aisle",
            "t_cold_aisle",
            "supply_air_temp",
            "cold_aisle_c",
            "supply_air_temperature"
        ]
    },
    "it_power_consumption": {
        "display_name": "IT Compute Power Consumption",
        "unit": "kW / W",
        "is_target": False,
        "required": False,
        "expected_range": (0.05, 5000.0),
        "aliases": [
            "it_power_consumption",
            "it_power_p_it",
            "p_it",
            "it_power",
            "power_it",
            "it_load",
            "server_power",
            "it_load_kw",
            "it_power_kw"
        ]
    },
    "facility_power_consumption": {
        "display_name": "Total Facility Power Consumption",
        "unit": "kW / W",
        "is_target": False,
        "required": False,
        "expected_range": (0.1, 10000.0),
        "aliases": [
            "facility_power_consumption",
            "facility_power_p_dc",
            "p_dc",
            "facility_power",
            "power_dc",
            "total_power",
            "total_facility_power",
            "facility_load_kw",
            "total_power_kw"
        ]
    },
    "cooling_power_consumption": {
        "display_name": "Cooling Power Consumption",
        "unit": "kW",
        "is_target": False,
        "required": False,
        "expected_range": (0.1, 600.0),
        "aliases": [
            "cooling_power_consumption",
            "cooling_power",
            "chiller_power",
            "crac_power",
            "crah_power",
            "hvac_power",
            "cooling_power_kw"
        ]
    },
    "cooling_supply_temperature": {
        "display_name": "Cooling Supply Temperature",
        "unit": "°C",
        "is_target": False,
        "required": False,
        "expected_range": (5.0, 45.0),
        "aliases": [
            "cooling_supply_temperature",
            "cooling_supply_temperature_c",
            "cooling_supply_average",
            "chilled_water_supply",
            "cws_temp",
            "cooling_supply_temp"
        ]
    },
    "cooling_return_temperature": {
        "display_name": "Cooling Return Temperature",
        "unit": "°C",
        "is_target": False,
        "required": False,
        "expected_range": (10.0, 50.0),
        "aliases": [
            "cooling_return_temperature",
            "cooling_return_temperature_c",
            "cooling_return_average",
            "chilled_water_return",
            "cwr_temp",
            "cooling_return_temp"
        ]
    },
    "timestamp": {
        "display_name": "Calendar Timestamp",
        "unit": "ISO / Datetime",
        "is_target": False,
        "required": False,
        "aliases": [
            "timestamp",
            "datetime",
            "date_time",
            "time",
            "sample_time",
            "recorded_at",
            "time_utc",
            "utc_time",
            "date",
            "ts"
        ]
    },
    "elapsed_time_minutes": {
        "display_name": "Elapsed Time",
        "unit": "Minutes",
        "is_target": False,
        "required": False,
        "aliases": [
            "elapsed_time_minutes",
            "elapsed_minutes",
            "elapsed_time",
            "elapsed_min",
            "minutes",
            "time_index"
        ]
    },
    "experiment_id": {
        "display_name": "Experiment / Batch Identifier",
        "unit": "ID",
        "is_target": False,
        "required": False,
        "aliases": [
            "experiment_id",
            "experiment",
            "test_id",
            "exp_id",
            "test_configuration",
            "trial_id",
            "run_id",
            "batch_id"
        ]
    },
    "humidity": {
        "display_name": "Relative Humidity",
        "unit": "%",
        "is_target": False,
        "required": False,
        "expected_range": (10.0, 95.0),
        "aliases": [
            "humidity",
            "relative_humidity",
            "rh",
            "humidity_percent",
            "rh_percent",
            "ambient_humidity"
        ]
    },
    "cpu_usage": {
        "display_name": "Server CPU Utilization",
        "unit": "%",
        "is_target": False,
        "required": False,
        "expected_range": (0.0, 100.0),
        "aliases": [
            "cpu_usage",
            "cpu_utilization",
            "cpu_load",
            "cpu_percent",
            "processor_load",
            "cpu",
            "it_load_percent"
        ]
    }
}

# Pre-normalized alias lookup table for exact O(1) matching
_NORMALIZED_ALIAS_LOOKUP: Dict[str, str] = {}
for canon_key, meta in CANONICAL_REGISTRY.items():
    for alias in meta["aliases"]:
        _NORMALIZED_ALIAS_LOOKUP[normalize_column_name(alias)] = canon_key

def find_canonical_match(raw_col: str) -> Optional[str]:
    """Finds matching canonical key for a raw column name."""
    norm = normalize_column_name(raw_col)
    if norm in _NORMALIZED_ALIAS_LOOKUP:
        return _NORMALIZED_ALIAS_LOOKUP[norm]
    # Check if exact normalized canonical key
    if norm in CANONICAL_REGISTRY:
        return norm
    return None

def normalize_dataframe_to_canonical(
    df: pd.DataFrame,
    manual_overrides: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Transforms any incoming DataFrame into the canonical internal schema.
    
    Returns:
        Tuple of:
        - df_canon: DataFrame containing all standard canonical columns.
        - mapped_features: Dict of {canonical_key: raw_column_name}
        - unmapped_columns: List of raw columns that did not match any canonical feature.
        
    Raises:
        CanonicalSchemaError: If prediction target ('inlet_temperature') is missing.
    """
    manual_overrides = manual_overrides or {}
    raw_columns = list(df.columns)
    mapped_features: Dict[str, str] = {}
    used_raw_columns = set()

    # 1. Apply administrator overrides first
    for canon_key, raw_col in manual_overrides.items():
        if canon_key in CANONICAL_REGISTRY and raw_col in raw_columns:
            mapped_features[canon_key] = raw_col
            used_raw_columns.add(raw_col)

    # 2. Automated alias matching for remaining canonical attributes
    for raw_col in raw_columns:
        if raw_col in used_raw_columns:
            continue
        canon_match = find_canonical_match(raw_col)
        if canon_match and canon_match not in mapped_features:
            mapped_features[canon_match] = raw_col
            used_raw_columns.add(raw_col)

    # 3. Check for mandatory prediction target
    if CANONICAL_TARGET_KEY not in mapped_features:
        target_meta = CANONICAL_REGISTRY[CANONICAL_TARGET_KEY]
        error_msg = (
            f"Schema Normalization Failed: Missing required prediction target '{CANONICAL_TARGET_KEY}'. "
            f"Available dataset columns: {raw_columns}. "
            f"Expected one of supported aliases: {target_meta['aliases']}."
        )
        raise CanonicalSchemaError(
            message=error_msg,
            required_feature=CANONICAL_TARGET_KEY,
            detected_columns=raw_columns,
            suggested_aliases=target_meta["aliases"]
        )

    # 4. Construct canonical representation DataFrame
    df_canon = df.copy()

    # Apply mappings with type conversion
    for canon_key, raw_col in mapped_features.items():
        if canon_key == "timestamp":
            # If calendar timestamp, parse datetime
            dt_parsed = pd.to_datetime(df_canon[raw_col], errors="coerce")
            # If values are not small numbers (i.e. not step indices or elapsed mins)
            num_check = pd.to_numeric(df_canon[raw_col], errors="coerce")
            if dt_parsed.notna().mean() >= 0.8 and (num_check.isna().mean() >= 0.5 or num_check.max() >= 1_000_000):
                df_canon["timestamp"] = dt_parsed
                df_canon["time_index"] = dt_parsed
                df_canon["time_label"] = dt_parsed.astype(str)
            else:
                # Treat as elapsed minutes or step index
                if "elapsed_time_minutes" not in mapped_features:
                    df_canon["elapsed_time_minutes"] = pd.to_numeric(df_canon[raw_col], errors="coerce")
                    df_canon["time_index"] = df_canon["elapsed_time_minutes"]
                    df_canon["time_label"] = df_canon["elapsed_time_minutes"].apply(lambda m: f"{int(m)} min" if pd.notnull(m) else "")
                    df_canon["timestamp"] = df_canon["time_label"]
        elif canon_key == "elapsed_time_minutes":
            df_canon["elapsed_time_minutes"] = pd.to_numeric(df_canon[raw_col], errors="coerce")
            if "time_index" not in df_canon.columns:
                df_canon["time_index"] = df_canon["elapsed_time_minutes"]
                df_canon["time_label"] = df_canon["elapsed_time_minutes"].apply(lambda m: f"{int(m)} min" if pd.notnull(m) else "")
                if "timestamp" not in df_canon.columns:
                    df_canon["timestamp"] = df_canon["time_label"]
        elif canon_key == "experiment_id":
            df_canon["experiment_id"] = df_canon[raw_col].astype(str)
        else:
            # Numeric sensor feature
            df_canon[canon_key] = pd.to_numeric(df_canon[raw_col], errors="coerce")

    # Ensure canonical target is always present as numeric float
    target_raw_col = mapped_features[CANONICAL_TARGET_KEY]
    df_canon["inlet_temperature"] = pd.to_numeric(df_canon[target_raw_col], errors="coerce")

    # Add legacy alias mirrors so existing modules referencing legacy names never fail
    df_canon["inlet_temperature_c"] = df_canon["inlet_temperature"]
    if "outlet_temperature" in df_canon.columns:
        df_canon["outlet_temperature_c"] = df_canon["outlet_temperature"]
    if "it_power_consumption" in df_canon.columns:
        df_canon["it_power_p_it"] = df_canon["it_power_consumption"]
        df_canon["p_it"] = df_canon["it_power_consumption"]
    if "facility_power_consumption" in df_canon.columns:
        df_canon["facility_power_p_dc"] = df_canon["facility_power_consumption"]
        df_canon["facility_power"] = df_canon["facility_power_consumption"]
        df_canon["p_dc"] = df_canon["facility_power_consumption"]
    if "cooling_supply_temperature" in df_canon.columns:
        df_canon["cooling_supply_temperature_c"] = df_canon["cooling_supply_temperature"]
    if "cooling_return_temperature" in df_canon.columns:
        df_canon["cooling_return_temperature_c"] = df_canon["cooling_return_temperature"]

    # Ensure time label and index exist
    if "time_index" not in df_canon.columns:
        if "timestamp" in df_canon.columns:
            df_canon["time_index"] = df_canon["timestamp"]
            df_canon["time_label"] = df_canon["timestamp"].astype(str)
        elif "elapsed_time_minutes" in df_canon.columns:
            df_canon["time_index"] = df_canon["elapsed_time_minutes"]
            df_canon["time_label"] = df_canon["elapsed_time_minutes"].apply(lambda m: f"{int(m)} min" if pd.notnull(m) else "")
            df_canon["timestamp"] = df_canon["time_label"]
        else:
            df_canon["time_index"] = np.arange(len(df_canon))
            df_canon["time_label"] = df_canon["time_index"].astype(str)
            df_canon["timestamp"] = df_canon["time_label"]

    # Chronological sort (within experiment_id if partitioned)
    if "experiment_id" in df_canon.columns and df_canon["experiment_id"].notna().any():
        df_canon["experiment_id"] = df_canon["experiment_id"].astype(str)
        if "time_index" in df_canon.columns:
            df_canon = df_canon.sort_values(by=["experiment_id", "time_index"]).reset_index(drop=True)
    elif "time_index" in df_canon.columns:
        df_canon = df_canon.sort_values(by="time_index").reset_index(drop=True)

    unmapped = [c for c in raw_columns if c not in used_raw_columns]
    return df_canon, mapped_features, unmapped
