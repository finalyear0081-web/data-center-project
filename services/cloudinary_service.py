"""
Cloudinary Storage Service Module
Handles dataset upload to Cloudinary with local storage fallback.
"""
import os
import shutil
from datetime import datetime, timezone
from typing import Dict, Any, Optional

CLOUDINARY_AVAILABLE = False
try:
    import cloudinary
    import cloudinary.uploader
    CLOUDINARY_AVAILABLE = True
except ImportError:
    CLOUDINARY_AVAILABLE = False

LOCAL_UPLOADS_DIR = "dataset/uploads"

class CloudinaryService:
    def __init__(self):
        self.is_configured = False
        self._init_cloudinary()
        os.makedirs(LOCAL_UPLOADS_DIR, exist_ok=True)
        
    def _init_cloudinary(self):
        cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
        api_key = os.getenv("CLOUDINARY_API_KEY")
        api_secret = os.getenv("CLOUDINARY_API_SECRET")
        
        if CLOUDINARY_AVAILABLE and cloud_name and api_key and api_secret:
            try:
                cloudinary.config(
                    cloud_name=cloud_name,
                    api_key=api_key,
                    api_secret=api_secret,
                    secure=True
                )
                self.is_configured = True
                print("Cloudinary service configured.")
            except Exception as e:
                print(f"Cloudinary configuration warning: {e}")

    def upload_dataset(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Uploads a CSV file to Cloudinary or falls back to local uploads directory.
        """
        if self.is_configured:
            try:
                result = cloudinary.uploader.upload(
                    file_path,
                    resource_type="raw",
                    folder="datacenter_datasets",
                    public_id=f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{filename}"
                )
                return {
                    "success": True,
                    "storage": "cloudinary",
                    "url": result.get("secure_url"),
                    "public_id": result.get("public_id"),
                    "filename": filename,
                    "size_bytes": result.get("bytes", os.path.getsize(file_path))
                }
            except Exception as e:
                print(f"Cloudinary upload failed: {e}. Defaulting to local storage.")
                
        # Local fallback
        dest_path = os.path.join(LOCAL_UPLOADS_DIR, filename)
        shutil.copyfile(file_path, dest_path)
        return {
            "success": True,
            "storage": "local",
            "url": f"/dataset/uploads/{filename}",
            "local_path": dest_path,
            "filename": filename,
            "size_bytes": os.path.getsize(dest_path)
        }

_cloudinary_service = None

def get_cloudinary_service() -> CloudinaryService:
    global _cloudinary_service
    if _cloudinary_service is None:
        _cloudinary_service = CloudinaryService()
    return _cloudinary_service
