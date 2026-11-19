import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# MUST set KERAS_HOME and TF_ENABLE_ONEDNN_OPTS before importing tensorflow
keras_dir = os.path.abspath(os.path.join(BASE_DIR, ".keras"))
os.makedirs(keras_dir, exist_ok=True)
os.environ["KERAS_HOME"] = keras_dir
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

keras_json = os.path.join(keras_dir, "keras.json")
if not os.path.exists(keras_json):
    with open(keras_json, "w") as f:
        f.write('{"backend": "tensorflow"}')

import tensorflow as tf

from backend.models.train import train_lstm_pipeline
from backend.models.evaluation import run_model_evaluation
from backend.services.prediction_service import get_prediction_service

dataset_path = os.path.abspath(os.path.join(BASE_DIR, "dataset/hazardnet_cleaned_aggregate_data.csv"))
print("=== STARTING MODEL TRAINING ===")
history = train_lstm_pipeline(dataset_path, epochs=10, batch_size=64, stride=12)
print("Training completed.")
print(f"Final Train Loss: {history['loss'][-1]:.6f}, Final Val Loss: {history['val_loss'][-1]:.6f}")

print("\n=== STARTING MODEL EVALUATION ===")
metrics = run_model_evaluation(dataset_path, stride=24)
print("Evaluation Metrics:", metrics)

print("\n=== REFRESHING FORECAST ===")
svc = get_prediction_service()
forecast = svc.get_latest_forecast(force_refresh=True)
print("Forecast computed successfully!")
fc_list = forecast.get("forecast", [])
print(f"Forecast points: {len(fc_list)}")
print(f"Min temp: {forecast.get('metrics', {}).get('min_temp')}")
print(f"Max temp: {forecast.get('metrics', {}).get('max_temp')}")
print(f"First timestamp: {fc_list[0].get('timestamp') if fc_list else None}")
print(f"Last timestamp: {fc_list[-1].get('timestamp') if fc_list else None}")
