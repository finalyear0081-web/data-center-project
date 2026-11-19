"""
Cloudinary Dataset Storage Service
Manages secure remote storage, authenticated raw asset upload, metadata extraction,
and secure download of data center telemetry datasets (.csv, .xlsx, .xls, .json).
"""
import os
import time
import logging
import urllib.request
import urllib.error
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import cloudinary
import cloudinary.uploader
import cloudinary.api
import cloudinary.utils
import pandas as pd

from backend.config import Config
from backend.preprocessing.file_parser import parse_dataset_file

logger = logging.getLogger("cloudinary_service")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

TEMP_PROCESSING_DIR = os.path.join(Config.STORAGE_DIR, "temp_processing")


def compute_file_hash(file_path: str) -> str:
    """Computes SHA-256 hash of a local file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class CloudinaryService:
    def __init__(self):
        self.cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME") or getattr(Config, "CLOUDINARY_CLOUD_NAME", "")
        self.api_key = os.getenv("CLOUDINARY_API_KEY") or getattr(Config, "CLOUDINARY_API_KEY", "")
        self.api_secret = os.getenv("CLOUDINARY_API_SECRET") or getattr(Config, "CLOUDINARY_API_SECRET", "")
        self.cloudinary_url = os.getenv("CLOUDINARY_URL") or getattr(Config, "CLOUDINARY_URL", "")
        self.base_folder = getattr(Config, "CLOUDINARY_FOLDER", "data_center_datasets")

        os.makedirs(TEMP_PROCESSING_DIR, exist_ok=True)
        self._configure()

    def _configure(self):
        """Initializes Cloudinary SDK without exposing credentials."""
        if self.cloud_name and self.api_key and self.api_secret:
            cloudinary.config(
                cloud_name=self.cloud_name,
                api_key=self.api_key,
                api_secret=self.api_secret,
                secure=True
            )
            logger.info("Cloudinary SDK configured with cloud_name: %s", self.cloud_name)
        elif self.cloudinary_url:
            cloudinary.config(secure=True)
            logger.info("Cloudinary SDK configured via CLOUDINARY_URL")
        else:
            logger.warning("Cloudinary credentials not fully supplied in environment.")

    def is_configured(self) -> bool:
        """Returns True if Cloudinary credentials are fully configured."""
        return bool((self.cloud_name and self.api_key and self.api_secret) or self.cloudinary_url)

    def ping(self) -> Dict[str, Any]:
        """Pings Cloudinary API to verify connectivity."""
        if not self.is_configured():
            return {"status": "unconfigured", "error": "Cloudinary credentials missing."}
        try:
            return cloudinary.api.ping()
        except Exception as e:
            logger.error("Cloudinary ping failed: %s", str(e))
            return {"status": "error", "error": str(e)}

    def upload_dataset(
        self,
        file_path: str,
        original_filename: str,
        dataset_id: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Uploads a non-media dataset file (.csv, .xlsx, .xls, .json) to Cloudinary
        using resource_type="raw" in a structured folder hierarchy:
        data_center_datasets/{upload_date}/{dataset_id}/{original_filename}.
        Includes retry and error logging.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File to upload not found: {file_path}")

        if not self.is_configured():
            raise RuntimeError("Cloudinary is not configured. Missing CLOUDINARY_* environment variables.")

        upload_date = datetime.now(timezone.utc).strftime("%Y%m%d")
        if not dataset_id:
            file_hash = compute_file_hash(file_path)[:8]
            dataset_id = f"ds_{upload_date}_{int(time.time())}_{file_hash}"

        # Clean public_id preserving original filename inside structured folder
        clean_filename = os.path.basename(original_filename).replace(" ", "_")
        public_id = f"{self.base_folder}/{upload_date}/{dataset_id}/{clean_filename}"

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "Uploading '%s' to Cloudinary (Attempt %d/%d) as '%s'...",
                    original_filename, attempt, max_retries, public_id
                )
                
                upload_res = cloudinary.uploader.upload(
                    file_path,
                    resource_type="raw",
                    public_id=public_id,
                    overwrite=True,
                    invalidate=True,
                    use_filename=True,
                    unique_filename=False
                )

                logger.info(
                    "Cloudinary upload successful: public_id=%s, asset_id=%s, bytes=%s",
                    upload_res.get("public_id"),
                    upload_res.get("asset_id"),
                    upload_res.get("bytes")
                )

                return {
                    "dataset_id": dataset_id,
                    "original_filename": original_filename,
                    "cloudinary_public_id": upload_res.get("public_id"),
                    "cloudinary_asset_id": upload_res.get("asset_id") or upload_res.get("public_id"),
                    "secure_url": upload_res.get("secure_url"),
                    "file_size": upload_res.get("bytes", os.path.getsize(file_path)),
                    "file_format": os.path.splitext(original_filename)[1].lower(),
                    "created_at": upload_res.get("created_at") or datetime.now(timezone.utc).isoformat(),
                    "resource_type": "raw",
                    "upload_date": upload_date
                }
            except Exception as e:
                last_error = e
                logger.warning(
                    "Cloudinary upload attempt %d/%d failed: %s",
                    attempt, max_retries, str(e)
                )
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        raise RuntimeError(f"Cloudinary upload failed after {max_retries} attempts: {str(last_error)}")

    def download_dataset(
        self,
        cloudinary_public_id: str,
        dest_filename: Optional[str] = None,
        target_dir: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30
    ) -> str:
        """
        Securely downloads a dataset file from Cloudinary into a local processing directory.
        Returns the absolute local path to the downloaded file.
        Includes retry logic and integrity checks.
        """
        if not self.is_configured():
            raise RuntimeError("Cloudinary is not configured. Missing CLOUDINARY_* environment variables.")

        if not target_dir:
            target_dir = TEMP_PROCESSING_DIR
        os.makedirs(target_dir, exist_ok=True)

        if not dest_filename:
            dest_filename = os.path.basename(cloudinary_public_id)

        dest_path = os.path.join(target_dir, dest_filename)

        # Generate authenticated/signed download URL
        signed_url, _ = cloudinary.utils.cloudinary_url(
            cloudinary_public_id,
            resource_type="raw",
            secure=True,
            sign_url=True
        )

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "Downloading dataset '%s' from Cloudinary (Attempt %d/%d)...",
                    cloudinary_public_id, attempt, max_retries
                )

                req = urllib.request.Request(
                    signed_url,
                    headers={"User-Agent": "DataCenterCloudinaryClient/1.0"}
                )

                with urllib.request.urlopen(req, timeout=timeout) as response:
                    if response.status != 200:
                        raise urllib.error.HTTPError(
                            signed_url, response.status, f"HTTP Error {response.status}", response.headers, None
                        )
                    with open(dest_path, "wb") as out_file:
                        while chunk := response.read(65536):
                            out_file.write(chunk)

                if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                    logger.info("Successfully downloaded %d bytes to '%s'", os.path.getsize(dest_path), dest_path)
                    return dest_path
                else:
                    raise IOError("Downloaded file is empty or missing on disk.")

            except Exception as e:
                last_error = e
                logger.warning(
                    "Cloudinary download attempt %d/%d failed: %s",
                    attempt, max_retries, str(e)
                )
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        raise RuntimeError(f"Cloudinary download failed after {max_retries} attempts: {str(last_error)}")

    def cleanup_temp_file(self, file_path: str):
        """Safely removes temporary processing files."""
        if file_path and os.path.exists(file_path):
            try:
                # Ensure we only delete from temp processing directory
                real_path = os.path.realpath(file_path)
                real_temp = os.path.realpath(TEMP_PROCESSING_DIR)
                if real_path.startswith(real_temp):
                    os.remove(file_path)
                    logger.info("Cleaned up temporary processing file: %s", file_path)
            except Exception as e:
                logger.warning("Failed to clean up temp file '%s': %s", file_path, str(e))

    def build_metadata(
        self,
        file_path: str,
        original_filename: str,
        dataset_id: str,
        cloudinary_upload_info: Dict[str, Any],
        suitability_report: Optional[Dict[str, Any]] = None,
        validation_report: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Builds the unified database record conforming to all required metadata fields:
        - dataset_id
        - original_filename
        - cloudinary_public_id
        - cloudinary_asset_id
        - file_format
        - file_size
        - upload_timestamp
        - dataset_start_timestamp
        - dataset_end_timestamp
        - row_count
        - column_count
        - processing_status
        - dataset_validation_status
        Never includes API secrets.
        """
        suitability_report = suitability_report or {}
        validation_report = validation_report or {}

        file_format = os.path.splitext(original_filename)[1].lower()
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else cloudinary_upload_info.get("file_size", 0)
        file_hash = compute_file_hash(file_path) if os.path.exists(file_path) else None

        # Inspect dataframe dimensions and temporal bounds
        row_count = 0
        column_count = 0
        start_ts = None
        end_ts = None

        try:
            df = parse_dataset_file(file_path)
            row_count = len(df)
            column_count = len(df.columns)

            # Find potential timestamp columns
            ts_cols = [c for c in df.columns if any(k in str(c).lower() for k in ["time", "date", "ts"])]
            if ts_cols:
                ts_series = pd.to_datetime(df[ts_cols[0]], errors="coerce").dropna().sort_values()
                if not ts_series.empty:
                    start_ts = str(ts_series.iloc[0])
                    end_ts = str(ts_series.iloc[-1])
        except Exception as e:
            logger.warning("Could not extract deep dimensions for '%s': %s", original_filename, str(e))
            row_count = validation_report.get("total_records", 0)

        verdict = suitability_report.get("verdict", "UNKNOWN")
        is_valid = validation_report.get("is_valid", verdict in ["SUITABLE", "PARTIALLY_SUITABLE"])
        validation_status = "VALID" if is_valid else "INVALID"

        return {
            "dataset_id": dataset_id,
            "original_filename": original_filename,
            "filename": original_filename,
            "file_path": file_path,
            "file_hash": file_hash,
            "cloudinary_public_id": cloudinary_upload_info.get("cloudinary_public_id", ""),
            "cloudinary_asset_id": cloudinary_upload_info.get("cloudinary_asset_id", ""),
            "cloudinary_secure_url": cloudinary_upload_info.get("secure_url", ""),
            "file_format": file_format,
            "file_size": file_size,
            "upload_timestamp": datetime.now(timezone.utc).isoformat(),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "dataset_start_timestamp": start_ts,
            "dataset_end_timestamp": end_ts,
            "row_count": row_count,
            "column_count": column_count,
            "total_records": row_count,
            "processing_status": "COMPLETED" if is_valid else "FAILED",
            "dataset_validation_status": validation_status,
            "suitability_verdict": verdict,
            "schema_type": suitability_report.get("schema_checks", {}).get("detected_schema_name", "UNIVERSAL"),
            "schema_name": suitability_report.get("schema_checks", {}).get("detected_schema_name", "Universal Schema"),
            "sampling_interval_minutes": suitability_report.get("time_series_analysis", {}).get("inferred_interval_minutes", 2),
            "forecast_horizon_steps": suitability_report.get("recommended_horizon_steps", 720),
            "forecast_horizon_hours": suitability_report.get("recommended_horizon_hours", 24.0),
            "is_active": False
        }


_cloudinary_service = None

def get_cloudinary_service() -> CloudinaryService:
    global _cloudinary_service
    if _cloudinary_service is None:
        _cloudinary_service = CloudinaryService()
    return _cloudinary_service
