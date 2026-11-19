"""
Dataset Upload & Processing Status API Blueprint
Supports CSV, XLSX, XLS, and JSON files with universal schema detection and suitability analysis.
"""
import os
from flask import Blueprint, request
from backend.services.dataset_service import get_dataset_service
from backend.services.prediction_service import get_prediction_service
from backend.utils.response import success_response, error_response

upload_bp = Blueprint("upload_api", __name__, url_prefix="/api/dataset")

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}

@upload_bp.route("/upload", methods=["POST"])
def upload_dataset_endpoint():
    if "file" not in request.files:
        return error_response("No file uploaded. Please upload a file in field 'file'.", status_code=400)
        
    file = request.files["file"]
    if not file.filename:
        return error_response("No selected file.", status_code=400)

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return error_response(
            f"Invalid file format '{ext}'. Supported formats: CSV (.csv), Excel (.xlsx, .xls), JSON (.json).",
            status_code=400
        )
        
    dataset_svc = get_dataset_service()
    pred_svc = get_prediction_service()
    
    upload_result = dataset_svc.handle_upload(file)
    val_report = upload_result.get("validation", {})
    suitability = upload_result.get("suitability", {})
    
    is_acceptable = (val_report.get("is_valid", False) or suitability.get("verdict") in ["SUITABLE", "PARTIALLY_SUITABLE"])
    
    if not is_acceptable:
        reasons = suitability.get("reasons") or val_report.get("errors", ["Dataset is not suitable for temperature forecasting."])
        return error_response(
            message="Dataset validation and suitability evaluation failed.",
            errors=reasons,
            validation_report=val_report,
            suitability=suitability,
            status_code=422
        )
        
    # Trigger automated background pipeline with forced retrain for newly uploaded dataset
    job_info = pred_svc.start_pipeline_job(upload_result["file_path"], force_retrain=True)
    
    return success_response(
        data={
            "filename": os.path.basename(upload_result["file_path"]),
            "dataset_id": upload_result.get("dataset_id"),
            "cloudinary_public_id": upload_result.get("cloudinary_public_id"),
            "cloudinary_asset_id": upload_result.get("cloudinary_asset_id"),
            "total_records": suitability.get("data_quality", {}).get("total_records", val_report.get("total_records", 0)),
            "sampling_interval_minutes": suitability.get("time_series_analysis", {}).get("inferred_interval_minutes", val_report.get("sampling_interval_minutes", 5)),
            "forecast_horizon_steps": suitability.get("recommended_horizon_steps", val_report.get("forecast_horizon_steps", 144)),
            "suitability_verdict": suitability.get("verdict", "SUITABLE"),
            "suitability": suitability,
            "validation_report": val_report,
            "pipeline_job": job_info
        },
        message="Dataset uploaded and validated successfully. Automated forecasting pipeline initiated.",
        status_code=202
    )

@upload_bp.route("/status", methods=["GET"])
def dataset_status_endpoint():
    pred_svc = get_prediction_service()
    status = pred_svc.get_job_status()
    return success_response(data=status)
