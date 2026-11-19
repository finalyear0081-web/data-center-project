"""
Dataset Management Routes Blueprint
"""
import os
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
import pandas as pd

from config import Config
from preprocessing.data_loader import load_dataset, get_dataset_metadata, EXPECTED_COLUMNS
from preprocessing.cleaning import validate_and_clean_dataset
from services.cloudinary_service import get_cloudinary_service
from services.firebase_service import get_firebase_service
from routes.auth_middleware import require_auth

dataset_bp = Blueprint("dataset", __name__, url_prefix="/api/dataset")

@dataset_bp.route("/info", methods=["GET"])
def dataset_info():
    try:
        df = load_dataset(Config.DATASET_PATH)
        meta = get_dataset_metadata(df)
        return jsonify({"success": True, "data": meta})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@dataset_bp.route("/preview", methods=["GET"])
def dataset_preview():
    try:
        limit = min(int(request.args.get("limit", 20)), 100)
        df = load_dataset(Config.DATASET_PATH)
        preview_records = df.head(limit).to_dict(orient="records")
        for r in preview_records:
            r["timestamp"] = str(r["timestamp"])
        return jsonify({"success": True, "count": len(preview_records), "records": preview_records})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@dataset_bp.route("/statistics", methods=["GET"])
def dataset_statistics():
    try:
        df = load_dataset(Config.DATASET_PATH)
        meta = get_dataset_metadata(df)
        return jsonify({"success": True, "statistics": meta["statistics"]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@dataset_bp.route("/upload", methods=["POST"])
@require_auth(admin_only=True)
def dataset_upload():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded in form field 'file'."}), 400
        
    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"success": False, "error": "Only CSV files are supported."}), 400
        
    filename = secure_filename(file.filename)
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    temp_path = os.path.join(Config.UPLOAD_FOLDER, f"temp_{filename}")
    file.save(temp_path)
    
    try:
        # Validate uploaded CSV
        uploaded_df = load_dataset(temp_path)
        clean_df, report = validate_and_clean_dataset(uploaded_df)
        
        # Save to Cloudinary / storage
        cloud_svc = get_cloudinary_service()
        upload_res = cloud_svc.upload_dataset(temp_path, filename)
        
        # Save metadata to Firestore
        meta = get_dataset_metadata(clean_df)
        fb = get_firebase_service()
        record_id = fb.save_record("datasets", {
            "filename": filename,
            "storage_url": upload_res.get("url"),
            "storage_type": upload_res.get("storage"),
            "total_records": meta["total_records"],
            "uploaded_by": getattr(request, "user", {}).get("email", "admin"),
            "validation_report": report
        })
        
        return jsonify({
            "success": True,
            "message": "Dataset uploaded and validated successfully.",
            "dataset_id": record_id,
            "metadata": meta,
            "validation": report,
            "storage": upload_res
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Dataset validation failed: {str(e)}"}), 400
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@dataset_bp.route("/download", methods=["GET"])
def download_dataset():
    if not os.path.exists(Config.DATASET_PATH):
        return jsonify({"error": "Dataset not found."}), 404
    return send_file(Config.DATASET_PATH, as_attachment=True, download_name="data_center_temperature_dataset.csv")
