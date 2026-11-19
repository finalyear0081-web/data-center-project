"""
Storage Management Service for Datasets, Models, and Artifacts
With strict hash-based deduplication and persistent active dataset management.
"""
import os
import json
import hashlib
import shutil
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from backend.config import Config

DATASET_HISTORY_FILE = os.path.join(Config.STORAGE_DIR, "dataset_history.json")
ACTIVE_DATASET_FILE = os.path.join(Config.STORAGE_DIR, "active_dataset.json")

def compute_file_hash(file_path: str) -> str:
    """Computes SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

class StorageService:
    def __init__(self):
        os.makedirs(Config.UPLOADS_DIR, exist_ok=True)
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)

    def save_uploaded_file(self, file_storage, filename: str) -> str:
        """
        Saves uploaded file and performs SHA-256 deduplication.
        If an identical file exists, returns the existing file path.
        """
        temp_name = f"temp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{filename}"
        temp_path = os.path.join(Config.UPLOADS_DIR, temp_name)
        file_storage.save(temp_path)

        file_hash = compute_file_hash(temp_path)

        # Check existing files in uploads for matching hash
        for existing_name in os.listdir(Config.UPLOADS_DIR):
            if existing_name == temp_name:
                continue
            existing_path = os.path.join(Config.UPLOADS_DIR, existing_name)
            if os.path.isfile(existing_path):
                try:
                    if compute_file_hash(existing_path) == file_hash:
                        # Duplicate found - clean up temp and return existing
                        os.remove(temp_path)
                        return existing_path
                except Exception:
                    pass

        # Unique file - promote from temp to permanent safe name
        safe_name = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{filename}"
        dest_path = os.path.join(Config.UPLOADS_DIR, safe_name)
        os.rename(temp_path, dest_path)
        return dest_path

    def record_dataset_history(self, entry: Dict[str, Any]):
        """
        Records or updates dataset metadata in history without duplicates.
        Ensures all Cloudinary asset metadata is preserved and secrets are never stored.
        """
        # Sanitize against accidental secret leakage
        sanitized_entry = {k: v for k, v in entry.items() if "secret" not in k.lower()}

        history = self.get_dataset_history()
        file_path = sanitized_entry.get("file_path")
        file_hash = sanitized_entry.get("file_hash")
        dataset_id = sanitized_entry.get("dataset_id")
        cld_id = sanitized_entry.get("cloudinary_public_id")

        if not file_hash and file_path and os.path.exists(file_path):
            file_hash = compute_file_hash(file_path)
            sanitized_entry["file_hash"] = file_hash

        # Update if already exists, else append
        updated = False
        for i, item in enumerate(history):
            matches = (
                (dataset_id and item.get("dataset_id") == dataset_id) or
                (cld_id and item.get("cloudinary_public_id") == cld_id) or
                (file_path and item.get("file_path") == file_path) or
                (file_hash and item.get("file_hash") == file_hash)
            )
            if matches:
                history[i].update(sanitized_entry)
                updated = True
                break

        if not updated:
            history.append(sanitized_entry)

        try:
            with open(DATASET_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        except Exception:
            pass

    def get_dataset_history(self) -> List[Dict[str, Any]]:
        if os.path.exists(DATASET_HISTORY_FILE):
            try:
                with open(DATASET_HISTORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def get_cloudinary_dataset_history(self) -> List[Dict[str, Any]]:
        """
        Retrieves ONLY datasets whose original files are successfully stored in Cloudinary.
        Excludes local-only files, temporary stubs, and external API syncs without Cloudinary assets.
        """
        history = self.get_dataset_history()
        cloudinary_records = []
        for item in history:
            cld_pub = item.get("cloudinary_public_id")
            cld_asset = item.get("cloudinary_asset_id")

            # Must have non-empty, non-local Cloudinary IDs
            if not cld_pub or not cld_asset:
                continue
            if str(cld_pub).startswith("local_") or str(cld_asset).startswith("local_"):
                continue
            if not str(cld_pub).startswith("data_center_datasets/"):
                continue

            record = dict(item)
            record["cloudinary_storage_status"] = "STORED"
            record["storage_provider"] = "Cloudinary"
            # Ensure row_count and file_size are present
            record["row_count"] = record.get("row_count") or record.get("total_records", 0)
            record["original_filename"] = record.get("original_filename") or record.get("filename", "dataset")
            cloudinary_records.append(record)

        return cloudinary_records

    def get_active_dataset_path(self) -> Optional[str]:
        if os.path.exists(ACTIVE_DATASET_FILE):
            try:
                with open(ACTIVE_DATASET_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    p = data.get("file_path")
                    if p and os.path.exists(p):
                        return p
            except Exception:
                pass
        return None

    def set_active_dataset_path(self, file_path: str):
        os.makedirs(Config.STORAGE_DIR, exist_ok=True)
        filename = os.path.basename(file_path)
        payload = {
            "file_path": file_path,
            "filename": filename,
            "activated_at": datetime.now(timezone.utc).isoformat()
        }
        try:
            with open(ACTIVE_DATASET_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

        # Invalidate previous forecast cache
        cache_file = os.path.join(Config.PREDICTIONS_DIR, "latest_forecast.json")
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
            except Exception:
                pass

        # Update is_active in history
        history = self.get_dataset_history()
        for item in history:
            item["is_active"] = (item.get("file_path") == file_path or item.get("filename") == filename)
        try:
            with open(DATASET_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        except Exception:
            pass

_storage_service = None

def get_storage_service() -> StorageService:
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
