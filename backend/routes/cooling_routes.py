"""
Predictive Cooling and Chiller Control API Blueprint
"""
from flask import Blueprint, request
from backend.cooling.controller import get_cooling_controller
from backend.cooling.threshold import get_threshold_manager
from backend.services.prediction_service import get_prediction_service
from backend.utils.response import success_response, error_response

cooling_bp = Blueprint("cooling_api", __name__, url_prefix="/api/cooling")

@cooling_bp.route("/status", methods=["GET"])
def cooling_status_endpoint():
    ctrl = get_cooling_controller()
    return success_response(data=ctrl.get_status())

@cooling_bp.route("/mode", methods=["POST"])
def set_cooling_mode_endpoint():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode")
    if not mode:
        return error_response("Missing required field 'mode'. Must be 'AUTOMATIC' or 'MANUAL'.", status_code=400)
        
    ctrl = get_cooling_controller()
    pred_svc = get_prediction_service()
    fc = pred_svc.get_latest_forecast()
    
    temps = fc.get("prediction", {}).get("forecast_temperatures", [])
    times = fc.get("prediction", {}).get("forecast_timestamps", [])
    
    try:
        updated_status = ctrl.set_operating_mode(mode, forecast_temps=temps, forecast_timestamps=times)
        return success_response(data=updated_status, message=f"Operating mode set to {mode}.")
    except ValueError as e:
        return error_response(str(e), status_code=400)

@cooling_bp.route("/manual", methods=["POST"])
def manual_chiller_action_endpoint():
    data = request.get_json(silent=True) or {}
    action = data.get("action")  # "ON" or "OFF"
    if not action:
        return error_response("Missing required field 'action'. Must be 'ON' or 'OFF'.", status_code=400)
        
    ctrl = get_cooling_controller()
    updated = ctrl.manual_override(action)
    return success_response(data=updated, message=f"Chiller manually commanded to {updated['chiller_state']}.")

@cooling_bp.route("/history", methods=["GET"])
def cooling_history_endpoint():
    limit = int(request.args.get("limit", 50))
    ctrl = get_cooling_controller()
    return success_response(data=ctrl.get_event_history(limit=limit))

@cooling_bp.route("/thresholds", methods=["GET", "PUT", "POST"])
def thresholds_endpoint():
    th_mgr = get_threshold_manager()

    if request.method == "GET":
        th = th_mgr.get_thresholds()
        return success_response(data={
            "on_threshold": th.on_threshold,
            "off_threshold": th.off_threshold,
            "min_allowable": th.min_allowable,
            "max_allowable": th.max_allowable,
            "deadband": round(th.on_threshold - th.off_threshold, 2)
        })

    data = request.get_json(silent=True) or {}
    on_th = data.get("on_threshold")
    off_th = data.get("off_threshold")
    
    if on_th is None or off_th is None:
        return error_response("Both 'on_threshold' and 'off_threshold' are required.", status_code=400)
    
    try:
        on_flt = float(on_th)
        off_flt = float(off_th)
    except (ValueError, TypeError):
        return error_response("Threshold setpoints must be valid numeric temperatures in Celsius.", status_code=400)

    success, msg = th_mgr.update_thresholds(on_flt, off_flt)
    if not success:
        return error_response(msg, status_code=422)
        
    # Re-evaluate forecast, controller state, and 15-minute presentation records
    pred_svc = get_prediction_service()
    pred_svc.recalculate_cooling_with_thresholds(on_flt, off_flt)
        
    return success_response(
        data={
            "on_threshold": on_flt,
            "off_threshold": off_flt,
            "deadband": round(on_flt - off_flt, 2)
        },
        message="Cooling thresholds updated and verified."
    )
