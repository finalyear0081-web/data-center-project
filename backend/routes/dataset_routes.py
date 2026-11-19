"""
Dataset Preview, Statistics, Info, History, Analysis, and Remapping API Blueprint
"""
from flask import Blueprint, request
from backend.services.dataset_service import get_dataset_service
from backend.services.storage_service import get_storage_service
from backend.preprocessing.schema_detector import CANONICAL_CATALOG
from backend.utils.response import success_response, error_response

dataset_bp = Blueprint("dataset_api", __name__, url_prefix="/api/dataset")

@dataset_bp.route("/preview", methods=["GET"])
def dataset_preview_endpoint():
    limit = min(int(request.args.get("limit", 20)), 100)
    svc = get_dataset_service()
    preview = svc.get_preview(limit=limit)
    return success_response(data=preview)

@dataset_bp.route("/info", methods=["GET"])
def dataset_info_endpoint():
    svc = get_dataset_service()
    info = svc.get_info()
    return success_response(data=info)

@dataset_bp.route("/statistics", methods=["GET"])
def dataset_statistics_endpoint():
    svc = get_dataset_service()
    stats = svc.get_statistics()
    return success_response(data=stats)

@dataset_bp.route("/history", methods=["GET"])
def dataset_history_endpoint():
    storage = get_storage_service()
    cloudinary_only = request.args.get("cloudinary_only", "false").lower() in ["true", "1", "yes"]
    if cloudinary_only:
        from backend.services.cloudinary_service import get_cloudinary_service
        cld = get_cloudinary_service()
        if not cld.is_configured():
            return error_response("Cloudinary is not configured on this server.", status_code=503)
        history = storage.get_cloudinary_dataset_history()
    else:
        history = storage.get_dataset_history()
    return success_response(data=history)

@dataset_bp.route("/cloudinary-history", methods=["GET"])
def dataset_cloudinary_history_endpoint():
    from backend.services.cloudinary_service import get_cloudinary_service
    cld = get_cloudinary_service()
    if not cld.is_configured():
        return error_response("Cloudinary is not configured on this server.", status_code=503)
    storage = get_storage_service()
    history = storage.get_cloudinary_dataset_history()
    return success_response(data=history)

@dataset_bp.route("/analysis", methods=["GET"])
def dataset_analysis_get_endpoint():
    svc = get_dataset_service()
    analysis = svc.get_suitability_analysis()
    return success_response(data=analysis)

@dataset_bp.route("/analyze", methods=["POST"])
def dataset_analyze_post_endpoint():
    data = request.get_json(silent=True) or {}
    overrides = data.get("mappings") or data.get("manual_overrides")
    svc = get_dataset_service()
    analysis = svc.get_suitability_analysis(force_refresh=True, overrides=overrides)
    return success_response(data=analysis, message="Dataset suitability analysis completed.")

@dataset_bp.route("/remap", methods=["POST"])
def dataset_remap_endpoint():
    data = request.get_json(silent=True) or {}
    mappings = data.get("mappings", {})
    if not isinstance(mappings, dict):
        return error_response("Payload must contain a 'mappings' object.", status_code=400)
        
    svc = get_dataset_service()
    updated_analysis = svc.remap_dataset(mappings)
    return success_response(
        data=updated_analysis,
        message="Column mappings updated and dataset re-evaluated successfully."
    )

@dataset_bp.route("/activate", methods=["POST"])
def dataset_activate_endpoint():
    data = request.get_json(silent=True) or {}
    dataset_id = data.get("dataset_id") or data.get("file_path") or data.get("filename")
    if not dataset_id:
        return error_response("Must provide 'dataset_id', 'filename', or 'file_path'.", status_code=400)
    svc = get_dataset_service()
    try:
        result = svc.activate_dataset(dataset_id)
        return success_response(data=result, message=f"Dataset '{result['filename']}' is now active.")
    except Exception as e:
        return error_response(f"Activation failed: {str(e)}", status_code=404)

@dataset_bp.route("/catalog", methods=["GET"])
def dataset_catalog_endpoint():
    return success_response(data=CANONICAL_CATALOG)
