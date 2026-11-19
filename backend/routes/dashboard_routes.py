"""
Dashboard Telemetry and Real-Time State API Blueprint
"""
from flask import Blueprint, request
from backend.services.prediction_service import get_prediction_service
from backend.cooling.alerts import get_alert_manager
from backend.utils.response import success_response

dashboard_bp = Blueprint("dashboard_api", __name__, url_prefix="/api/dashboard")

@dashboard_bp.route("/summary", methods=["GET"])
def dashboard_summary_endpoint():
    pred_svc = get_prediction_service()
    summary = pred_svc.get_dashboard_summary()
    return success_response(data=summary)

@dashboard_bp.route("/temperature", methods=["GET"])
def dashboard_temperature_endpoint():
    hours = int(request.args.get("hours", 24))
    pred_svc = get_prediction_service()
    telemetry = pred_svc.get_telemetry_history(hours=hours)
    return success_response(data=telemetry)

@dashboard_bp.route("/forecast", methods=["GET"])
def dashboard_forecast_endpoint():
    pred_svc = get_prediction_service()
    fc = pred_svc.get_latest_forecast()
    return success_response(data=fc)

@dashboard_bp.route("/alerts", methods=["GET"])
def dashboard_alerts_endpoint():
    limit = int(request.args.get("limit", 30))
    alert_mgr = get_alert_manager()
    alerts = alert_mgr.get_recent(limit=limit)
    return success_response(data=alerts)
