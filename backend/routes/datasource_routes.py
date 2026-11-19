"""
External Telemetry REST API Connection & Ingestion API Blueprint
Handles authorized endpoint setup, secure credentials, latency testing, and telemetry sync.
"""
from flask import Blueprint, request
from backend.services.datasource_service import get_datasource_service
from backend.services.storage_service import get_storage_service
from backend.services.prediction_service import get_prediction_service
from backend.utils.response import success_response, error_response

datasource_bp = Blueprint("datasource_api", __name__, url_prefix="/api/datasource")

@datasource_bp.route("/status", methods=["GET"])
def datasource_status_endpoint():
    svc = get_datasource_service()
    status = svc.get_status()
    return success_response(data=status)

@datasource_bp.route("/connect", methods=["POST"])
def datasource_connect_endpoint():
    data = request.get_json(silent=True) or {}
    endpoint_url = data.get("endpoint_url")
    if not endpoint_url:
        return error_response("Field 'endpoint_url' is required.", status_code=400)

    auth_type = data.get("auth_type", "NONE")
    api_key = data.get("api_key")
    header_name = data.get("header_name", "Authorization")
    custom_headers = data.get("custom_headers", {})
    json_path = data.get("json_path", "")
    polling_interval = data.get("polling_interval_minutes", 15)
    sync_mode = data.get("sync_mode", "HISTORICAL")

    svc = get_datasource_service()
    # Test connection first
    test_result = svc.test_connection(
        endpoint_url=endpoint_url,
        auth_type=auth_type,
        api_key=api_key,
        header_name=header_name,
        custom_headers=custom_headers
    )

    if not test_result.get("success"):
        return error_response(
            message=f"Failed to verify API endpoint: {test_result.get('error')}",
            errors=[test_result.get("error")],
            status_code=400
        )

    # Save connection configuration securely
    updated_status = svc.save_connection(
        endpoint_url=endpoint_url,
        auth_type=auth_type,
        api_key=api_key,
        header_name=header_name,
        custom_headers=custom_headers,
        json_path=json_path,
        polling_interval_minutes=polling_interval,
        sync_mode=sync_mode
    )

    return success_response(
        data=updated_status,
        message=f"External API data source connected successfully ({test_result.get('records_detected')} records detected).",
        status_code=200
    )

@datasource_bp.route("/test", methods=["POST"])
def datasource_test_endpoint():
    data = request.get_json(silent=True) or {}
    svc = get_datasource_service()
    result = svc.test_connection(
        endpoint_url=data.get("endpoint_url"),
        auth_type=data.get("auth_type"),
        api_key=data.get("api_key"),
        header_name=data.get("header_name"),
        custom_headers=data.get("custom_headers")
    )
    if not result.get("success"):
        return error_response(
            message=result.get("error", "API test connection failed."),
            errors=[result.get("error")],
            status_code=400
        )
    return success_response(data=result, message=result.get("message", "API connection test successful."))

@datasource_bp.route("/sync", methods=["POST"])
def datasource_sync_endpoint():
    svc = get_datasource_service()
    sync_result = svc.sync_data()

    if not sync_result.get("success"):
        return error_response(
            message=sync_result.get("error", "Failed to sync telemetry from external API."),
            errors=[sync_result.get("error")],
            status_code=400
        )

    # Launch pipeline job with retrain on newly ingested dataset snapshot
    pred_svc = get_prediction_service()
    pred_svc.start_pipeline_job(sync_result["file_path"], force_retrain=True)

    return success_response(
        data=sync_result,
        message=f"Telemetry data synced successfully ({sync_result.get('total_records')} records ingested).",
        status_code=202
    )

@datasource_bp.route("/sources", methods=["GET"])
def datasource_sources_endpoint():
    storage = get_storage_service()
    history = storage.get_dataset_history()
    api_sources = [h for h in history if h.get("source_type") == "EXTERNAL_REST_API"]
    return success_response(data=api_sources)
