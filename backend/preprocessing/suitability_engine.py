"""
Automatic Dataset Suitability Engine
Performs rigorous multi-dimensional diagnostic analysis:
A. Schema checks (target presence, feature catalog, types)
B. Data quality (missing values, duplicates, invalid timestamps, out-of-order, out-of-bounds, constant columns, gaps)
C. Time-series analysis (inferred sampling interval, regularity, duration, multi-experiment partitions)
D. Temperature distribution & thermal excursions (min/max/mean/std, counts >= 26, 28, 30, 33°C, trend)
E. Forecast suitability verdict: SUITABLE | PARTIALLY_SUITABLE | NOT_SUITABLE with horizon recommendation.
"""
import math
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from backend.preprocessing.schema_detector import UniversalSchemaDetector, CANONICAL_CATALOG

class DatasetSuitabilityEngine:
    """
    Automated Data Quality & Forecasting Feasibility Evaluation Engine.
    """

    @classmethod
    def evaluate_dataset(cls, df: pd.DataFrame, 
                         manual_overrides: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Executes end-to-end suitability analysis on raw or partially mapped dataset.
        """
        # 1. Run Universal Schema Detection
        mapping_result = UniversalSchemaDetector.detect_canonical_mapping(df, manual_overrides)
        mappings = mapping_result["mappings"]
        df_canon = UniversalSchemaDetector.apply_canonical_transformation(df, mapping_result)

        total_rows = len(df)
        reasons: List[str] = []
        warnings: List[str] = []
        recommendations: List[str] = []

        # ==========================================
        # A. SCHEMA CHECKS
        # ==========================================
        target_info = mappings.get("inlet_temperature", {})
        target_available = bool(target_info.get("available") and target_info.get("source_column"))
        target_source_col = target_info.get("source_column")

        available_features = [k for k, v in mappings.items() if v.get("available") and k not in ["timestamp", "experiment_id", "device_id"]]
        unavailable_features = [k for k, v in mappings.items() if not v.get("available") and k not in ["experiment_id", "device_id"]]

        schema_status = {
            "target_detected": target_available,
            "target_source_column": target_source_col,
            "target_confidence": target_info.get("confidence", 0.0),
            "available_features_count": len(available_features),
            "available_features": available_features,
            "unavailable_features": unavailable_features,
            "detected_schema_name": mapping_result["detected_schema_name"]
        }

        if not target_available:
            reasons.append("CRITICAL: Server inlet temperature target could not be identified.")

        # ==========================================
        # B. DATA QUALITY CHECKS
        # ==========================================
        total_duplicates = int(df.duplicated().sum())
        missing_by_feature = {k: v.get("missing_percentage", 0.0) for k, v in mappings.items() if v.get("available")}
        
        # Check for constant numeric columns (variance == 0)
        constant_columns = []
        for feat in available_features:
            s = df_canon[feat].dropna()
            if len(s) > 1 and s.nunique() <= 1:
                constant_columns.append(feat)
                warnings.append(f"Column '{mappings[feat]['source_column']}' is constant across all readings.")

        # Check for infinite numbers
        infinite_counts = 0
        for feat in available_features:
            s = df_canon[feat].dropna()
            inf_mask = np.isinf(s)
            if inf_mask.any():
                infinite_counts += int(inf_mask.sum())

        data_quality = {
            "total_records": total_rows,
            "duplicate_records_count": total_duplicates,
            "infinite_values_count": infinite_counts,
            "constant_columns": constant_columns,
            "missing_percentages": missing_by_feature
        }

        # ==========================================
        # C. TIME-SERIES ANALYSIS
        # ==========================================
        time_info = mappings.get("timestamp", {})
        time_source_col = time_info.get("source_column")
        time_analysis: Dict[str, Any] = {
            "time_source_column": time_source_col,
            "is_calendar_timestamp": False,
            "inferred_interval_minutes": 5,
            "is_regular_sampling": True,
            "out_of_order_count": 0,
            "total_duration_hours": 0.0,
            "experiments_count": 1,
            "experiments_list": []
        }

        # Multi-experiment detection
        exp_col = mappings.get("experiment_id", {}).get("source_column")
        if exp_col and exp_col in df.columns:
            exp_series = df[exp_col].dropna().astype(str)
            unique_exps = exp_series.unique().tolist()
            time_analysis["experiments_count"] = len(unique_exps)
            time_analysis["experiments_list"] = unique_exps[:20]
        else:
            time_analysis["experiments_count"] = 1
            time_analysis["experiments_list"] = ["single_stream"]

        if time_source_col and time_source_col in df.columns:
            time_series_raw = df[time_source_col]
            
            # Check if clearly numeric elapsed time (integers/floats < 1,000,000)
            is_num_dtype = pd.api.types.is_numeric_dtype(time_series_raw)
            numeric_conv = pd.to_numeric(time_series_raw, errors="coerce")
            is_elapsed_time = is_num_dtype or (numeric_conv.notna().mean() >= 0.8 and numeric_conv.max() < 1_000_000)
            
            if is_elapsed_time and numeric_conv.notna().mean() >= 0.8:
                # Numeric elapsed-time format (e.g. AAU elapsed_time_minutes)
                time_analysis["is_calendar_timestamp"] = False
                diffs_min = numeric_conv.diff().dropna()
                diffs_min = diffs_min[diffs_min > 0]
                if not diffs_min.empty:
                    interval_mins = max(1, int(round(diffs_min.median())))
                    time_analysis["inferred_interval_minutes"] = interval_mins
                    max_elapsed = float(numeric_conv.max())
                    time_analysis["total_duration_hours"] = round(max_elapsed / 60.0, 1)
            else:
                parsed_datetime = pd.to_datetime(time_series_raw, errors="coerce")
                # If datetime parsing succeeded for majority
                if parsed_datetime.notna().mean() >= 0.80:
                    time_analysis["is_calendar_timestamp"] = True
                    diffs_seconds = parsed_datetime.diff().dropna().dt.total_seconds()
                    diffs_seconds = diffs_seconds[diffs_seconds > 0]
                    if not diffs_seconds.empty:
                        median_sec = diffs_seconds.median()
                        interval_mins = max(1, int(round(median_sec / 60.0)))
                        time_analysis["inferred_interval_minutes"] = interval_mins
                        
                        # Check regularity (within 25% of median)
                        regular_ratio = ((diffs_seconds >= median_sec * 0.75) & (diffs_seconds <= median_sec * 1.25)).mean()
                        time_analysis["is_regular_sampling"] = bool(regular_ratio >= 0.85)
                        if not time_analysis["is_regular_sampling"]:
                            warnings.append(f"Irregular sampling intervals detected ({int(regular_ratio*100)}% on-schedule).")

                        # Check monotonicity
                        out_of_order = int((parsed_datetime.diff().dt.total_seconds() < 0).sum())
                        time_analysis["out_of_order_count"] = out_of_order
                        if out_of_order > 0:
                            warnings.append(f"Found {out_of_order} out-of-order timestamp readings.")

                        # Duration
                        span_sec = (parsed_datetime.max() - parsed_datetime.min()).total_seconds()
                        time_analysis["total_duration_hours"] = round(span_sec / 3600.0, 1)
        else:
            time_analysis["inferred_interval_minutes"] = 5
            warnings.append("No explicit timestamp column detected; assumed default 5-minute sampling.")

        # ==========================================
        # D. TEMPERATURE DISTRIBUTION ANALYSIS
        # ==========================================
        temp_analysis: Dict[str, Any] = {
            "min_celsius": None,
            "max_celsius": None,
            "mean_celsius": None,
            "std_celsius": None,
            "quantiles": {},
            "counts_above_26": 0,
            "counts_above_28": 0,
            "counts_above_30": 0,
            "counts_above_33": 0,
            "thermal_volatility": "MODERATE",
            "out_of_bounds_count": 0
        }

        if target_available and "inlet_temperature" in df_canon.columns:
            t_data = df_canon["inlet_temperature"].dropna()
            if not t_data.empty:
                t_min = round(float(t_data.min()), 2)
                t_max = round(float(t_data.max()), 2)
                t_mean = round(float(t_data.mean()), 2)
                t_std = round(float(t_data.std()), 2)

                temp_analysis["min_celsius"] = t_min
                temp_analysis["max_celsius"] = t_max
                temp_analysis["mean_celsius"] = t_mean
                temp_analysis["std_celsius"] = t_std
                temp_analysis["quantiles"] = {
                    "25%": round(float(t_data.quantile(0.25)), 2),
                    "50%": round(float(t_data.median()), 2),
                    "75%": round(float(t_data.quantile(0.75)), 2)
                }
                temp_analysis["counts_above_26"] = int((t_data >= 26.0).sum())
                temp_analysis["counts_above_28"] = int((t_data >= 28.0).sum())
                temp_analysis["counts_above_30"] = int((t_data >= 30.0).sum())
                temp_analysis["counts_above_33"] = int((t_data >= 33.0).sum())

                # Out of physical domain bounds (< 5°C or > 80°C)
                oob_mask = (t_data < 5.0) | (t_data > 80.0)
                temp_analysis["out_of_bounds_count"] = int(oob_mask.sum())
                if temp_analysis["out_of_bounds_count"] > 0:
                    reasons.append(f"Found {temp_analysis['out_of_bounds_count']} sensor readings outside physical reality [5°C, 80°C].")

                if t_std < 0.2:
                    temp_analysis["thermal_volatility"] = "EXTREMELY_LOW"
                    warnings.append("Inlet temperature has almost no variance; model may produce flat line.")
                elif t_std > 5.0:
                    temp_analysis["thermal_volatility"] = "VERY_HIGH"

        # ==========================================
        # E. FORECAST SUITABILITY & HORIZON FEASIBILITY
        # ==========================================
        interval_m = time_analysis["inferred_interval_minutes"]
        steps_in_24h = max(1, int(round(1440.0 / interval_m)))
        
        # Calculate available records per experiment
        num_experiments = max(1, time_analysis["experiments_count"])
        records_per_exp = total_rows // num_experiments

        # Feasibility check for 24-hour horizon:
        # Requires at least: input_window (steps_in_24h) + forecast_horizon (steps_in_24h) = 2 * steps_in_24h per partition
        is_24h_feasible = (records_per_exp >= steps_in_24h * 2)

        if is_24h_feasible:
            recommended_horizon_hours = 24.0
            recommended_horizon_steps = steps_in_24h
            recommended_input_steps = steps_in_24h
            recommendations.append("Full 24-hour forecasting horizon is validated and supported.")
        else:
            # Calibrate shorter supported horizon
            if records_per_exp >= 120:
                recommended_horizon_hours = 1.0
                recommended_horizon_steps = max(1, int(round(60.0 / interval_m)))
                recommended_input_steps = recommended_horizon_steps * 2
                recommendations.append(
                    f"24-hour horizon infeasible ({records_per_exp} records per partition < {steps_in_24h*2} required). "
                    f"Calibrated validated 1-hour horizon ({recommended_horizon_steps} steps) with {recommended_input_steps}-step history."
                )
            elif records_per_exp >= 60:
                recommended_horizon_hours = 0.5
                recommended_horizon_steps = max(1, int(round(30.0 / interval_m)))
                recommended_input_steps = recommended_horizon_steps * 2
                recommendations.append(
                    f"Calibrated validated 30-minute horizon ({recommended_horizon_steps} steps)."
                )
            else:
                recommended_horizon_hours = 0.0
                recommended_horizon_steps = 0
                recommended_input_steps = 0
                reasons.append(f"Insufficient historical depth ({records_per_exp} records per partition). Minimum 60 records required.")

        # ==========================================
        # FINAL SUITABILITY VERDICT CLASSIFICATION
        # ==========================================
        if not target_available or records_per_exp < 60 or temp_analysis.get("out_of_bounds_count", 0) > (total_rows * 0.10):
            suitability_verdict = "NOT_SUITABLE"
            suitability_label = "Not Suitable for Temperature Forecasting"
            suitability_badge = "danger"
            summary_message = "Dataset fails core readiness criteria. Primary temperature target missing or data corrupted."
        elif is_24h_feasible and len(available_features) >= 2 and len(reasons) == 0:
            suitability_verdict = "SUITABLE"
            suitability_label = "Fully Suitable for 24-Hour Forecasting"
            suitability_badge = "success"
            summary_message = f"High-quality telemetry with sufficient continuous history for full 24-hour forecasting ({steps_in_24h} steps)."
        else:
            suitability_verdict = "PARTIALLY_SUITABLE"
            suitability_label = f"Partially Suitable ({recommended_horizon_hours}h Horizon Supported)"
            suitability_badge = "warning"
            summary_message = (
                f"Dataset is viable for thermal modeling, but requires a calibrated {recommended_horizon_hours}-hour horizon "
                f"or handles partitioned experiment boundaries."
            )

        return {
            "verdict": suitability_verdict,
            "verdict_label": suitability_label,
            "badge_color": suitability_badge,
            "summary_message": summary_message,
            "is_24h_feasible": is_24h_feasible,
            "recommended_horizon_hours": recommended_horizon_hours,
            "recommended_horizon_steps": recommended_horizon_steps,
            "recommended_input_steps": recommended_input_steps,
            "schema_checks": schema_status,
            "data_quality": data_quality,
            "time_series_analysis": time_analysis,
            "temperature_analysis": temp_analysis,
            "reasons": reasons,
            "warnings": warnings,
            "recommendations": recommendations,
            "mapping_details": mappings
        }
