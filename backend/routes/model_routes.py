"""
Model Training, Performance Metrics, and Summary API Blueprint
"""
import os
import json
from flask import Blueprint, request
from backend.config import Config
from backend.preprocessing.data_loader import load_and_parse_csv
from backend.services.prediction_service import get_prediction_service
from backend.services.dataset_service import get_dataset_service
from backend.utils.response import success_response, error_response

model_bp = Blueprint("model_api", __name__, url_prefix="/api/model")

@model_bp.route("/status", methods=["GET"])
def model_status_endpoint():
    status_path = Config.TRAINING_STATUS_PATH
    if os.path.exists(status_path):
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                return success_response(data=json.load(f))
        except Exception:
            pass
    return success_response(data={"status": "READY", "progress": 1.0, "message": "Model is ready for inference."})

@model_bp.route("/metrics", methods=["GET"])
def model_metrics_endpoint():
    metrics = {}
    history = {}
    
    if os.path.exists(Config.METRICS_PATH):
        try:
            with open(Config.METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            pass
            
    # Prefer current loss_history.json for genuine traceable training run
    hist_path = os.path.join(Config.MODELS_DIR, "loss_history.json")
    if not os.path.exists(hist_path):
        hist_path = os.path.join(Config.MODELS_DIR, "training_history.json")
    if os.path.exists(hist_path):
        try:
            with open(hist_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass

    # Check compatibility with active dataset
    dataset_svc = get_dataset_service()
    active_path = dataset_svc.get_current_dataset_path()
    active_filename = os.path.basename(active_path) if active_path else "none"
    trained_source = history.get("dataset_source") or metrics.get("dataset_source")
    
    is_compatible = (trained_source == active_filename) if trained_source else True
    
    return success_response(data={
        "evaluation": metrics,
        "training_history": history,
        "is_model_compatible": is_compatible,
        "active_dataset": active_filename,
        "trained_dataset": trained_source
    })

@model_bp.route("/history", methods=["GET"])
def model_history_endpoint():
    loss_history_path = os.path.join(Config.MODELS_DIR, "loss_history.json")
    if os.path.exists(loss_history_path):
        try:
            with open(loss_history_path, "r", encoding="utf-8") as f:
                return success_response(data=json.load(f))
        except Exception:
            pass
    return success_response(data={"loss": [], "val_loss": [], "epochs": 0, "message": "No training history recorded yet."})

@model_bp.route("/train", methods=["POST"])
def trigger_train_endpoint():
    data = request.get_json(silent=True) or {}
    epochs = int(data.get("epochs", 10))
    architecture = str(data.get("architecture", "RESIDUAL_LSTM")).strip().upper()
    
    dataset_svc = get_dataset_service()
    active_path = dataset_svc.get_current_dataset_path()
    
    if not active_path:
        return error_response("No dataset found to train model.", status_code=400)
        
    pred_svc = get_prediction_service()
    res = pred_svc.start_pipeline_job(
        active_path,
        force_retrain=True,
        architecture=architecture,
        epochs=epochs
    )
    return success_response(
        data=res,
        message=f"Model retraining initiated with {architecture} for {epochs} epochs.",
        status_code=202
    )

@model_bp.route("/summary", methods=["GET"])
def model_summary_endpoint():
    model_exists = os.path.exists(Config.MODEL_PATH)
    scaler_exists = os.path.exists(Config.SCALER_PATH)
    
    loss_history_path = os.path.join(Config.MODELS_DIR, "loss_history.json")
    arch_info = "Residual LSTM with Persistence Anchor"
    horizon_steps = 144
    trained_source = None
    trained_schema = None
    input_window = None

    if os.path.exists(loss_history_path):
        try:
            with open(loss_history_path, "r", encoding="utf-8") as f:
                h = json.load(f)
                arch = h.get("architecture", "RESIDUAL_LSTM")
                horizon_steps = h.get("horizon_steps", 144)
                input_window = h.get("input_window")
                trained_source = h.get("dataset_source")
                trained_schema = h.get("schema_type")
                if arch == "STACKED_LSTM":
                    arch_info = f"Stacked Multi-Layer LSTM ({horizon_steps} Output Steps)"
                else:
                    arch_info = f"Residual LSTM with Persistence Anchor ({horizon_steps} Output Steps)"
        except Exception:
            pass

    dataset_svc = get_dataset_service()
    active_path = dataset_svc.get_current_dataset_path()
    active_filename = os.path.basename(active_path) if active_path else "unknown"

    is_compatible = False
    compatibility_msg = "No trained model found."
    if model_exists and scaler_exists and active_path and os.path.exists(active_path):
        if trained_source == active_filename:
            is_compatible = True
            compatibility_msg = f"Model is directly trained on active dataset '{active_filename}' ({horizon_steps}-step horizon)."
        else:
            try:
                df_test = load_and_parse_csv(active_path)
                if len(df_test) >= (input_window or 100):
                    is_compatible = True
                    compatibility_msg = f"Model ({arch_info}) is compatible with active dataset '{active_filename}'."
                else:
                    is_compatible = False
                    compatibility_msg = f"Active dataset has {len(df_test)} records, below model's required input window of {input_window}."
            except Exception:
                is_compatible = False
                compatibility_msg = f"Model trained on '{trained_source}' may not be compatible with '{active_filename}'."

    return success_response(data={
        "model_exists": model_exists,
        "scaler_exists": scaler_exists,
        "model_architecture": arch_info,
        "framework": "TensorFlow / Keras 2.x/3.x",
        "target": "inlet_temperature (°C)",
        "forecast_horizon_steps": horizon_steps,
        "trained_dataset_source": trained_source,
        "active_dataset_source": active_filename,
        "is_model_compatible": is_compatible,
        "compatibility_message": compatibility_msg
    })
