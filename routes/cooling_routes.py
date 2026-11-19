"""
Cooling Control System Routes Blueprint
"""
from flask import Blueprint, request, jsonify

from cooling.controller import get_cooling_controller
from cooling.threshold import get_threshold_manager
from routes.auth_middleware import require_auth

cooling_bp = Blueprint("cooling", __name__, url_prefix="/api/cooling")

@cooling_bp.route("/status", methods=["GET"])
def cooling_status():
    controller = get_cooling_controller()
    return jsonify({"success": True, "data": controller.get_status()})

@cooling_bp.route("/history", methods=["GET"])
def cooling_history():
    limit = int(request.args.get("limit", 50))
    controller = get_cooling_controller()
    return jsonify({"success": True, "events": controller.get_event_history(limit=limit)})

@cooling_bp.route("/evaluate", methods=["POST"])
def evaluate_custom_cooling():
    data = request.get_json(silent=True) or {}
    temps = data.get("temperatures", [])
    timestamps = data.get("timestamps", [])
    
    if not temps:
        return jsonify({"success": False, "error": "Array of temperatures required."}), 400
        
    controller = get_cooling_controller()
    result = controller.evaluate_forecast(temps, timestamps or [f"T+{i*5}m" for i in range(len(temps))])
    return jsonify({"success": True, "evaluation": result})

@cooling_bp.route("/thresholds", methods=["PUT"])
@require_auth(admin_only=True)
def update_thresholds():
    data = request.get_json(silent=True) or {}
    on_th = data.get("on_threshold")
    off_th = data.get("off_threshold")
    
    if on_th is None or off_th is None:
        return jsonify({"success": False, "error": "Both 'on_threshold' and 'off_threshold' are required."}), 400
        
    th_mgr = get_threshold_manager()
    success, msg = th_mgr.update_thresholds(float(on_th), float(off_th))
    
    if not success:
        return jsonify({"success": False, "error": msg}), 400
        
    return jsonify({
        "success": True,
        "message": msg,
        "thresholds": {
            "on_threshold": float(on_th),
            "off_threshold": float(off_th)
        }
    })
