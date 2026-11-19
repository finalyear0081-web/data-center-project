"""
Universal Schema Detection & Feature Mapping Engine
Identifies data center sensor attributes dynamically across arbitrary naming conventions,
inspects data types, value ranges, and generates canonical schema metadata with confidence scores.
Supports administrator-assisted mapping overrides.
"""
import re
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

# Canonical data center sensor catalog
CANONICAL_CATALOG: Dict[str, Dict[str, Any]] = {
    "timestamp": {
        "display_name": "Time Index / Timestamp",
        "unit": "ISO / Elapsed Min",
        "is_time": True,
        "required": True,
        "aliases": [
            "elapsed_time_minutes", "timestamp", "datetime", "date_time", "elapsed_time",
            "elapsed_min", "time_index", "time_utc", "utc_time", "sample_time",
            "date", "ts", "recorded_at", "time", "minutes"
        ]
    },
    "inlet_temperature": {
        "display_name": "Server Inlet Temperature (Target)",
        "unit": "°C",
        "is_target": True,
        "required": True,
        "expected_range": (10.0, 50.0),
        "aliases": [
            "inlet_temperature", "inlet_temperature_c", "inlet_temp", "inlet_temp_c",
            "inlet_air_temp", "inlet temp avg", "inlet_avg_temp", "inlet", "t_inlet",
            "t_in", "temp_inlet", "temp_in", "server_inlet_temp", "rack_inlet_temp",
            "rack_inlet_temperature", "inlet_c", "temperature_inlet", "temperature"
        ]
    },
    "outlet_temperature": {
        "display_name": "Server Outlet Temperature",
        "unit": "°C",
        "required": False,
        "expected_range": (15.0, 75.0),
        "aliases": [
            "outlet_temperature", "outlet_temperature_c", "outlet_temp", "outlet_temp_c",
            "outlet_air_temp", "outlet avg temp", "outlet_avg_temp", "outlet", "t_outlet",
            "t_out", "temp_outlet", "temp_out", "server_outlet_temp", "rack_outlet_temp",
            "rack_outlet_temperature", "outlet_c", "temperature_outlet"
        ]
    },
    "hot_aisle_temperature": {
        "display_name": "Hot Aisle Air Temperature",
        "unit": "°C",
        "required": False,
        "expected_range": (15.0, 65.0),
        "aliases": [
            "hot_aisle_temperature", "hot_aisle_temperature_c", "hot_aisle_temp",
            "hot_aisle", "t_hot_aisle", "exhaust_temp", "hot_aisle_c", "exhaust_temperature"
        ]
    },
    "cold_aisle_temperature": {
        "display_name": "Cold Aisle Air Temperature",
        "unit": "°C",
        "required": False,
        "expected_range": (10.0, 35.0),
        "aliases": [
            "cold_aisle_temperature", "cold_aisle_temperature_c", "cold_aisle_temp",
            "cold_aisle", "t_cold_aisle", "supply_air_temp", "cold_aisle_c", "supply_air_temperature"
        ]
    },
    "humidity": {
        "display_name": "Relative Humidity",
        "unit": "%",
        "required": False,
        "expected_range": (10.0, 95.0),
        "aliases": [
            "humidity", "relative_humidity", "rh", "humidity_percent", "rh_percent", "ambient_humidity"
        ]
    },
    "cpu_usage": {
        "display_name": "Server CPU Utilization",
        "unit": "%",
        "required": False,
        "expected_range": (0.0, 100.0),
        "aliases": [
            "cpu_usage", "cpu_utilization", "cpu_load", "cpu_percent", "processor_load",
            "cpu", "it_load_percent"
        ]
    },
    "it_power_consumption": {
        "display_name": "IT Compute Power Consumption",
        "unit": "kW / W",
        "required": False,
        "expected_range": (0.05, 5000.0),
        "aliases": [
            "it_power_consumption", "it_power", "it_power_p_it", "p_it", "power_it",
            "it_load", "server_power", "it_load_kw", "it_power_kw"
        ]
    },
    "facility_power_consumption": {
        "display_name": "Total Facility Power Consumption",
        "unit": "kW / W",
        "required": False,
        "expected_range": (0.1, 10000.0),
        "aliases": [
            "facility_power_consumption", "facility_power", "facility_power_p_dc", "p_dc",
            "power_dc", "total_power", "total_facility_power", "facility_load_kw"
        ]
    },
    "cooling_supply_temperature": {
        "display_name": "Cooling Supply Temperature",
        "unit": "°C",
        "required": False,
        "expected_range": (5.0, 45.0),
        "aliases": [
            "cooling_supply_temperature", "cooling_supply_temperature_c", "cooling_supply_average",
            "cooling supply average", "chilled_water_supply", "cws_temp", "cooling_supply_temp"
        ]
    },
    "cooling_return_temperature": {
        "display_name": "Cooling Return Temperature",
        "unit": "°C",
        "required": False,
        "expected_range": (10.0, 50.0),
        "aliases": [
            "cooling_return_temperature", "cooling_return_temperature_c", "cooling_return_average",
            "cooling return average", "chilled_water_return", "cwr_temp", "cooling_return_temp"
        ]
    },
    "cooling_power_consumption": {
        "display_name": "Cooling Power Consumption",
        "unit": "kW",
        "required": False,
        "expected_range": (0.1, 600.0),
        "aliases": [
            "cooling_power_consumption", "cooling_power", "chiller_power", "crac_power",
            "crah_power", "hvac_power", "cooling_power_kw"
        ]
    },
    "experiment_id": {
        "display_name": "Experiment / Batch Run Identifier",
        "unit": "ID",
        "required": False,
        "aliases": [
            "experiment_id", "exp_id", "test_configuration", "trial_id", "run_id", "batch_id"
        ]
    },
    "device_id": {
        "display_name": "Node / Device Identifier",
        "unit": "ID",
        "required": False,
        "aliases": [
            "node_id", "device_id", "node", "server_id", "rack_id", "chassis_id"
        ]
    }
}

def normalize_name(name: str) -> str:
    """Normalizes string for robust matching: lowercased, underscores, stripped."""
    s = str(name).strip().lower()
    s = re.sub(r"[\s\-\.]+", "_", s)
    s = re.sub(r"[^\w_]", "", s)
    return s

class UniversalSchemaDetector:
    """
    Automatic Schema Detection and Canonical Mapping Engine.
    """

    @classmethod
    def detect_canonical_mapping(cls, df: pd.DataFrame, 
                                 manual_overrides: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Scans dataframe columns, evaluates alias matches, type compatibility, and value ranges
        to map dataset features to canonical keys.
        
        Args:
            df: Input raw DataFrame.
            manual_overrides: Optional dict mapping {canonical_key: raw_column_name} by administrator.
            
        Returns:
            Dict containing:
            - mappings: {canonical_key: {source_column, confidence, unit, missing_pct, available}}
            - unmapped_columns: list of raw columns not mapped to any canonical feature
            - target_detected: bool (whether inlet_temperature is successfully mapped)
            - detected_schema_name: readable label for the dataset schema
        """
        raw_columns = list(df.columns)
        normalized_map = {normalize_name(c): c for c in raw_columns}
        used_raw_columns = set()
        
        mappings: Dict[str, Dict[str, Any]] = {}
        manual_overrides = manual_overrides or {}

        # 1. Apply administrator overrides first
        for canon_key, raw_col in manual_overrides.items():
            if canon_key in CANONICAL_CATALOG and raw_col in raw_columns:
                series = df[raw_col]
                missing_pct = round(float(series.isna().mean() * 100), 2)
                mappings[canon_key] = {
                    "canonical_key": canon_key,
                    "display_name": CANONICAL_CATALOG[canon_key]["display_name"],
                    "source_column": raw_col,
                    "unit": CANONICAL_CATALOG[canon_key]["unit"],
                    "confidence": 1.0,
                    "is_manual": True,
                    "available": (missing_pct < 100.0),
                    "missing_percentage": missing_pct
                }
                used_raw_columns.add(raw_col)

        # 2. Automated detection for remaining canonical features
        for canon_key, meta in CANONICAL_CATALOG.items():
            if canon_key in mappings:
                continue

            best_match: Optional[str] = None
            best_confidence = 0.0

            # Step A: Exact alias match
            for alias in meta["aliases"]:
                norm_alias = normalize_name(alias)
                if norm_alias in normalized_map:
                    candidate = normalized_map[norm_alias]
                    if candidate not in used_raw_columns:
                        best_match = candidate
                        best_confidence = 1.0
                        break

            # Step B: Partial / substring match
            if not best_match:
                for norm_raw, raw_orig in normalized_map.items():
                    if raw_orig in used_raw_columns:
                        continue
                    for alias in meta["aliases"]:
                        norm_alias = normalize_name(alias)
                        if norm_alias in norm_raw or norm_raw in norm_alias:
                            # Verify value range if expected_range specified
                            confidence = 0.80
                            if "expected_range" in meta:
                                s = pd.to_numeric(df[raw_orig], errors="coerce").dropna()
                                if not s.empty:
                                    min_b, max_b = meta["expected_range"]
                                    if s.mean() >= min_b and s.mean() <= max_b:
                                        confidence = 0.90
                                    else:
                                        confidence = 0.60
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_match = raw_orig

            # Step C: Record detection outcome
            if best_match and best_confidence >= 0.50:
                series = df[best_match]
                missing_pct = round(float(series.isna().mean() * 100), 2)
                mappings[canon_key] = {
                    "canonical_key": canon_key,
                    "display_name": meta["display_name"],
                    "source_column": best_match,
                    "unit": meta["unit"],
                    "confidence": round(best_confidence, 2),
                    "is_manual": False,
                    "available": (missing_pct < 100.0),
                    "missing_percentage": missing_pct
                }
                used_raw_columns.add(best_match)
            else:
                mappings[canon_key] = {
                    "canonical_key": canon_key,
                    "display_name": meta["display_name"],
                    "source_column": None,
                    "unit": meta["unit"],
                    "confidence": 0.0,
                    "is_manual": False,
                    "available": False,
                    "missing_percentage": 100.0
                }

        unmapped = [c for c in raw_columns if c not in used_raw_columns]
        target_detected = mappings.get("inlet_temperature", {}).get("available", False)

        # Detect high-level schema categorization
        if mappings.get("experiment_id", {}).get("available"):
            detected_schema_name = "Multi-Experiment Benchmark Schema"
        elif "node_id" in raw_columns or "device_id" in mappings:
            detected_schema_name = "Distributed Cluster Telemetry Schema"
        else:
            detected_schema_name = "Continuous Data Center Facility Telemetry"

        return {
            "mappings": mappings,
            "unmapped_columns": unmapped,
            "target_detected": target_detected,
            "detected_schema_name": detected_schema_name
        }

    @classmethod
    def apply_canonical_transformation(cls, df: pd.DataFrame, 
                                       mapping_result: Dict[str, Any]) -> pd.DataFrame:
        """
        Transforms raw DataFrame into canonical representation according to detection report.
        Preserves original columns as well, adding canonical alias columns.
        """
        df_canon = df.copy()
        mappings = mapping_result["mappings"]

        for canon_key, meta in mappings.items():
            src_col = meta.get("source_column")
            if src_col and src_col in df_canon.columns and meta.get("available"):
                df_canon[canon_key] = df_canon[src_col]
                # Enforce numeric for numerical features
                if canon_key not in ["timestamp", "experiment_id", "device_id"]:
                    df_canon[canon_key] = pd.to_numeric(df_canon[canon_key], errors="coerce")

        return df_canon
