"""
Dataset Operations Service
Supports multi-schema dataset management (HazardNet & AAU).
"""
import os
import json
from typing import Dict, Any, Optional
import pandas as pd
from werkzeug.utils import secure_filename

from backend.config import Config
from backend.preprocessing.validation import validate_csv_dataset, detect_schema
from backend.preprocessing.data_loader import load_and_parse_csv, get_detailed_statistics
from backend.preprocessing.file_parser import parse_dataset_file
from backend.preprocessing.schema_detector import UniversalSchemaDetector
from backend.preprocessing.suitability_engine import DatasetSuitabilityEngine
from backend.services.storage_service import get_storage_service

SUITABILITY_CACHE_FILE = os.path.join(Config.STORAGE_DIR, "latest_suitability_report.json")
OVERRIDES_FILE = os.path.join(Config.STORAGE_DIR, "column_mapping_overrides.json")

class DatasetService:
    def __init__(self):
        self.storage = get_storage_service()
        self.active_dataset_path = self._determine_active_dataset()

    def _determine_active_dataset(self) -> str:
        # Check explicit persistent active choice first
        saved_active = self.storage.get_active_dataset_path()
        if saved_active and os.path.exists(saved_active):
            return saved_active

        history = self.storage.get_dataset_history()
        # Find active item in history
        for item in history:
            if item.get("is_active") and item.get("file_path") and os.path.exists(item["file_path"]):
                return item["file_path"]

        if history:
            for item in reversed(history):
                p = item.get("file_path")
                if p and os.path.exists(p):
                    return p
        if os.path.exists(Config.DEFAULT_DATASET_PATH):
            return Config.DEFAULT_DATASET_PATH
        return ""

    def activate_dataset(self, file_path_or_id: str) -> Dict[str, Any]:
        """
        Activates a specific dataset from history or file path.
        Re-evaluates suitability and updates active dataset choice.
        If the file is not present locally, auto-downloads from Cloudinary.
        """
        target_path = None
        matched_item = None
        history = self.storage.get_dataset_history()
        for item in history:
            if item.get("dataset_id") == file_path_or_id or item.get("filename") == file_path_or_id or item.get("file_path") == file_path_or_id:
                target_path = item.get("file_path")
                matched_item = item
                break

        # Auto-download from Cloudinary if local file was removed or after server restart
        if not target_path or not os.path.exists(target_path):
            if matched_item and matched_item.get("cloudinary_public_id"):
                try:
                    from backend.services.cloudinary_service import get_cloudinary_service
                    cld = get_cloudinary_service()
                    target_path = cld.download_dataset(
                        matched_item["cloudinary_public_id"],
                        dest_filename=matched_item.get("original_filename") or matched_item.get("filename"),
                        target_dir=Config.UPLOADS_DIR
                    )
                    matched_item["file_path"] = target_path
                    self.storage.record_dataset_history(matched_item)
                except Exception:
                    pass

        if not target_path or not os.path.exists(target_path):
            if os.path.exists(file_path_or_id):
                target_path = file_path_or_id
            else:
                raise FileNotFoundError(f"Dataset '{file_path_or_id}' not found.")

        self.active_dataset_path = target_path
        self.storage.set_active_dataset_path(target_path)

        # Invalidate any forecast cache associated with the previous dataset
        # and regenerate dynamic forecast timestamps based on the newly active dataset
        try:
            from backend.services.prediction_service import get_prediction_service
            pred_svc = get_prediction_service()
            pred_svc.invalidate_forecast_cache()
            pred_svc.get_latest_forecast(force_refresh=True)
        except Exception as e:
            # Non-blocking if predictor dependencies need init
            pass

        # Run suitability analysis for activated dataset
        df = parse_dataset_file(target_path)
        overrides = self.get_column_overrides()
        suitability_report = DatasetSuitabilityEngine.evaluate_dataset(df, manual_overrides=overrides)
        self._cache_suitability(suitability_report)

        return {
            "active_dataset_path": target_path,
            "filename": os.path.basename(target_path),
            "suitability": suitability_report
        }

    def get_current_dataset_path(self) -> str:
        if not self.active_dataset_path or not os.path.exists(self.active_dataset_path):
            self.active_dataset_path = self._determine_active_dataset()
        return self.active_dataset_path

    def get_column_overrides(self) -> Dict[str, str]:
        if os.path.exists(OVERRIDES_FILE):
            try:
                with open(OVERRIDES_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def set_column_overrides(self, overrides: Dict[str, str]):
        os.makedirs(os.path.dirname(OVERRIDES_FILE), exist_ok=True)
        try:
            with open(OVERRIDES_FILE, "w", encoding="utf-8") as f:
                json.dump(overrides, f, indent=2)
        except Exception:
            pass

    def handle_upload(self, file_storage) -> Dict[str, Any]:
        """
        Saves uploaded file (CSV, XLSX, XLS, JSON), uploads to Cloudinary,
        downloads securely to temporary processing directory for verification and suitability analysis,
        and records complete metadata in the application database.
        """
        from datetime import datetime, timezone
        from backend.services.cloudinary_service import get_cloudinary_service, compute_file_hash, TEMP_PROCESSING_DIR
        cld_svc = get_cloudinary_service()

        raw_filename = secure_filename(file_storage.filename or "dataset.csv")
        saved_path = self.storage.save_uploaded_file(file_storage, raw_filename)
        file_hash = compute_file_hash(saved_path)

        # Generate unique dataset_id
        upload_date = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dataset_id = f"ds_{upload_date}_{file_hash[:8]}"

        # 1. Cloudinary upload or deduplication check
        cloudinary_info = {}
        history = self.storage.get_dataset_history()
        for item in history:
            if item.get("file_hash") == file_hash and item.get("cloudinary_public_id"):
                cloudinary_info = {
                    "dataset_id": item.get("dataset_id", dataset_id),
                    "original_filename": raw_filename,
                    "cloudinary_public_id": item["cloudinary_public_id"],
                    "cloudinary_asset_id": item.get("cloudinary_asset_id", item["cloudinary_public_id"]),
                    "secure_url": item.get("cloudinary_secure_url", ""),
                    "file_size": item.get("file_size", os.path.getsize(saved_path)),
                    "file_format": os.path.splitext(raw_filename)[1].lower()
                }
                dataset_id = item.get("dataset_id", dataset_id)
                break

        if not cloudinary_info and cld_svc.is_configured():
            try:
                cloudinary_info = cld_svc.upload_dataset(saved_path, raw_filename, dataset_id=dataset_id)
            except Exception as e:
                # Log upload failure and continue with local fallback
                cloudinary_info = {
                    "dataset_id": dataset_id,
                    "original_filename": raw_filename,
                    "cloudinary_public_id": f"local_{dataset_id}",
                    "cloudinary_asset_id": f"local_{dataset_id}",
                    "file_size": os.path.getsize(saved_path),
                    "file_format": os.path.splitext(raw_filename)[1].lower(),
                    "upload_error": str(e)
                }
        elif not cloudinary_info:
            cloudinary_info = {
                "dataset_id": dataset_id,
                "original_filename": raw_filename,
                "cloudinary_public_id": f"local_{dataset_id}",
                "cloudinary_asset_id": f"local_{dataset_id}",
                "file_size": os.path.getsize(saved_path),
                "file_format": os.path.splitext(raw_filename)[1].lower()
            }

        # 2. Secure download to temporary processing directory for verification
        processing_path = saved_path
        temp_download_path = None
        if cld_svc.is_configured() and cloudinary_info.get("cloudinary_public_id") and not cloudinary_info.get("cloudinary_public_id", "").startswith("local_"):
            try:
                temp_download_path = cld_svc.download_dataset(
                    cloudinary_info["cloudinary_public_id"],
                    dest_filename=f"verify_{raw_filename}"
                )
                processing_path = temp_download_path
            except Exception:
                processing_path = saved_path

        # 3. Dataset Validation & Preprocessing on downloaded file
        df = parse_dataset_file(processing_path)
        overrides = self.get_column_overrides()
        suitability_report = DatasetSuitabilityEngine.evaluate_dataset(df, manual_overrides=overrides)

        try:
            validation_report = validate_csv_dataset(processing_path)
        except Exception:
            validation_report = {
                "is_valid": (suitability_report["verdict"] != "NOT_SUITABLE"),
                "schema_type": suitability_report["schema_checks"].get("detected_schema_name", "UNIVERSAL"),
                "schema_name": suitability_report["schema_checks"].get("detected_schema_name", "Universal Schema"),
                "total_records": len(df),
                "sampling_interval_minutes": suitability_report["time_series_analysis"]["inferred_interval_minutes"],
                "forecast_horizon_steps": suitability_report["recommended_horizon_steps"],
                "forecast_horizon_hours": suitability_report["recommended_horizon_hours"],
                "errors": suitability_report["reasons"],
                "warnings": suitability_report["warnings"],
                "filename": raw_filename
            }

        # 4. Clean up temporary processing file
        if temp_download_path:
            cld_svc.cleanup_temp_file(temp_download_path)

        # 5. Build Unified Metadata Record (Strictly without API secrets)
        metadata = cld_svc.build_metadata(
            file_path=saved_path,
            original_filename=raw_filename,
            dataset_id=dataset_id,
            cloudinary_upload_info=cloudinary_info,
            suitability_report=suitability_report,
            validation_report=validation_report
        )

        self.active_dataset_path = saved_path
        self._cache_suitability(suitability_report)
        self.storage.record_dataset_history(metadata)

        return {
            "file_path": saved_path,
            "dataset_id": dataset_id,
            "cloudinary_public_id": metadata["cloudinary_public_id"],
            "cloudinary_asset_id": metadata["cloudinary_asset_id"],
            "validation": validation_report,
            "suitability": suitability_report,
            "metadata": metadata
        }

    def _cache_suitability(self, report: Dict[str, Any]):
        os.makedirs(os.path.dirname(SUITABILITY_CACHE_FILE), exist_ok=True)
        try:
            with open(SUITABILITY_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception:
            pass

    def get_suitability_analysis(self, force_refresh: bool = False, overrides: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Returns cached or dynamically evaluated suitability analysis for active dataset."""
        if not force_refresh and overrides is None and os.path.exists(SUITABILITY_CACHE_FILE):
            try:
                with open(SUITABILITY_CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        p = self.get_current_dataset_path()
        if not p or not os.path.exists(p):
            return {
                "verdict": "NOT_SUITABLE",
                "verdict_label": "No Active Dataset",
                "badge_color": "secondary",
                "summary_message": "No dataset currently active in system. Please upload a file or connect an API.",
                "reasons": ["No dataset loaded."],
                "warnings": [],
                "mapping_details": {}
            }

        df = parse_dataset_file(p)
        active_overrides = overrides if overrides is not None else self.get_column_overrides()
        report = DatasetSuitabilityEngine.evaluate_dataset(df, manual_overrides=active_overrides)
        self._cache_suitability(report)
        return report

    def remap_dataset(self, new_mappings: Dict[str, str]) -> Dict[str, Any]:
        """Updates administrator manual column mappings and re-evaluates active dataset."""
        current_overrides = self.get_column_overrides()
        current_overrides.update(new_mappings)
        self.set_column_overrides(current_overrides)
        return self.get_suitability_analysis(force_refresh=True, overrides=current_overrides)

    def get_preview(self, limit: int = 20) -> Dict[str, Any]:
        p = self.get_current_dataset_path()
        if not p or not os.path.exists(p):
            return {"count": 0, "records": [], "schema_type": "UNKNOWN", "schema_name": "No dataset active"}
            
        df = parse_dataset_file(p)
        schema_type, schema_name = detect_schema(df)
        
        if schema_type == "AAU_EXPERIMENTAL":
            priority_cols = [
                "experiment_id", "elapsed_time_minutes", "inlet_temperature_c",
                "outlet_temperature_c", "cooling_supply_temperature_c", "cooling_return_temperature_c",
                "it_power_p_it", "facility_power_p_dc", "PUE", "CoP"
            ]
            cols = [c for c in priority_cols if c in df.columns]
            df_preview = df[cols].head(limit).copy()
        else:
            df_preview = df.head(limit).copy()

        records = df_preview.to_dict(orient="records")
        for r in records:
            for k, v in r.items():
                if pd.isna(v):
                    r[k] = "-"
                elif isinstance(v, float):
                    r[k] = round(v, 2)
                    
        return {
            "count": len(records),
            "records": records,
            "schema_type": schema_type,
            "schema_name": schema_name,
            "total_records": len(df),
            "total_experiments": int(df["experiment_id"].nunique()) if "experiment_id" in df.columns else 1,
            "filename": os.path.basename(p)
        }

    def get_info(self) -> Dict[str, Any]:
        """Returns structured metadata about active dataset schema, features, and experiment counts."""
        p = self.get_current_dataset_path()
        if not p or not os.path.exists(p):
            return {"is_valid": False, "schema_name": "No dataset active"}
        return validate_csv_dataset(p)

    def get_statistics(self) -> Dict[str, Any]:
        p = self.get_current_dataset_path()
        if not p or not os.path.exists(p):
            return {}
        try:
            df = load_and_parse_csv(p)
        except Exception:
            df = parse_dataset_file(p)
        return get_detailed_statistics(df)

_dataset_service = None

def get_dataset_service() -> DatasetService:
    global _dataset_service
    if _dataset_service is None:
        _dataset_service = DatasetService()
    return _dataset_service
