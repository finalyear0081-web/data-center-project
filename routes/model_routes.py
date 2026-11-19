"""
Model Management & Inference Routes Blueprint
"""
import os
import json
import threading
from flask import Blueprint, request, jsonify

from config import Config
from models.train import train_pipeline, STATUS_PATH, HISTORY_PATH
from models.evaluate import EVAL_PATH
from services.prediction_service import get_prediction_service
from routes.auth_middleware import require_auth

model_bp = Blueprint("model", __name__, url_prefix="/api/model")

@model_bp.route("/status", methods=["GET"])
def model_status():
    if os.path.exists(STATUS_PATH):
        try:
            with open(STATUS_PATH, "r", encoding="utf-8") as f:
                return jsonify({"success": True, "status": json.load(f)})
        except Exception:
            pass
    return jsonify({
        "success": True,
        "status": {
            "status": "READY",
            "progress": 1.0,
            "message": "Model is trained and ready for inference."
        }
    })

@model_bp.route("/metrics", methods=["GET"])
def model_metrics():
    history = {}
    evaluation = {}
    
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass
            
    if os.path.exists(EVAL_PATH):
        try:
            with open(EVAL_PATH, "r", encoding="utf-8") as f:
                evaluation = json.load(f)
        except Exception:
            pass
            
    return jsonify({
        "success": True,
        "evaluation": evaluation,
        "training_history": history
    })

@model_bp.route("/train", methods=["POST"])
@require_auth(admin_only=True)
def trigger_training():
    data = request.get_json(silent=True) or {}
    epochs = int(data.get("epochs", 10))
    batch_size = int(data.get("batch_size", 64))
    
    # Run in daemon thread to avoid blocking HTTP request
    def run_training_job():
        try:
            train_pipeline(epochs=epochs, batch_size=batch_size, sample_stride=12)
        except Exception as e:
            print(f"Background training failed: {e}")
            
    thread = threading.Thread(target=run_training_job, daemon=True)
    thread.start()
    
    return jsonify({
        "success": True,
        "message": f"Model retraining initiated in background for {epochs} epochs.",
        "status": "STARTING"
    })

@model_bp.route("/predict", methods=["POST"])
def run_prediction():
    pred_service = get_prediction_service()
    result = pred_service.run_prediction_cycle()
    return jsonify({"success": True, "data": result})
