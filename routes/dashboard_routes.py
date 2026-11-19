"""
Dashboard Overview & Telemetry Routes Blueprint
"""
from flask import Blueprint, request, jsonify

from services.prediction_service import get_prediction_service
from cooling.alerts import get_alert_manager

dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/api/dashboard")

@dashboard_bp.route("/summary", methods=["GET"])
def dashboard_summary():
    pred_svc = get_prediction_service()
    data = pred_svc.get_dashboard_summary()
    return jsonify({"success": True, "data": data})

@dashboard_bp.route("/temperature", methods=["GET"])
def dashboard_temperature():
    hours = int(request.args.get("hours", 24))
    pred_svc = get_prediction_service()
    data = pred_svc.get_historical_telemetry(hours=hours)
    return jsonify({"success": True, "data": data})

@dashboard_bp.route("/forecast", methods=["GET"])
def dashboard_forecast():
    pred_svc = get_prediction_service()
    data = pred_svc.get_latest_forecast()
    return jsonify({
        "success": True,
        "forecast": data["prediction"],
        "cooling_evaluation": data["cooling"],
        "thresholds": data["thresholds"]
    })

@dashboard_bp.route("/alerts", methods=["GET"])
def dashboard_alerts():
    alert_mgr = get_alert_manager()
    limit = int(request.args.get("limit", 20))
    alerts = alert_mgr.get_recent_alerts(limit=limit)
    return jsonify({"success": True, "alerts": alerts})
