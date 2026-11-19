"""
Temperature Prediction and Export API Blueprint
"""
from flask import Blueprint, Response, request
from backend.services.prediction_service import get_prediction_service
from backend.services.dataset_service import get_dataset_service
from backend.utils.response import success_response, error_response

prediction_bp = Blueprint("prediction_api", __name__, url_prefix="/api/prediction")

@prediction_bp.route("/forecast", methods=["POST"])
def trigger_forecast_endpoint():
    pred_svc = get_prediction_service()
    dataset_svc = get_dataset_service()
    active_path = dataset_svc.get_current_dataset_path()
    
    if not active_path:
        return error_response("No dataset uploaded or available for forecasting.", status_code=400)
        
    result = pred_svc.get_latest_forecast()
    return success_response(data=result, message="24-Hour forecast generated successfully.")

@prediction_bp.route("/results", methods=["GET"])
@prediction_bp.route("/latest", methods=["GET"])
def get_prediction_results_endpoint():
    pred_svc = get_prediction_service()
    result = pred_svc.get_latest_forecast()
    return success_response(data=result)

@prediction_bp.route("/export", methods=["GET"])
def export_predictions_csv_endpoint():
    pred_svc = get_prediction_service()
    resolution = request.args.get("resolution", "15min").lower()
    if resolution not in ["15min", "full", "720"]:
        resolution = "15min"
        
    csv_data = pred_svc.export_predictions_csv(resolution=resolution)
    filename = "datacenter_temperature_forecast_24h_full.csv" if resolution in ["full", "720"] else "datacenter_temperature_forecast_24h_15min.csv"
    
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
