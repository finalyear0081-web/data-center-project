"""
End-to-end AAU Training, Evaluation, and Forecast Generation Script
"""
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

keras_dir = os.path.abspath(os.path.join(BASE_DIR, ".keras"))
os.makedirs(keras_dir, exist_ok=True)
os.environ["KERAS_HOME"] = keras_dir
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from backend.config import Config
from backend.preprocessing.validation import validate_csv_dataset
from backend.models.train import train_lstm_pipeline
from backend.models.evaluation import run_model_evaluation
from backend.services.prediction_service import get_prediction_service

aau_csv = os.path.abspath("storage/uploads/20261116_090649_aau_real_measurements_lstm.csv")

print("=== 1. VALIDATING AAU DATASET ===")
report = validate_csv_dataset(aau_csv)
print(f"Schema Name: {report['schema_name']}")
print(f"Total Experiments: {report['total_experiments']}")
print(f"Total Records: {report['total_records']}")
print(f"Sampling Interval: {report['sampling_interval_minutes']} minutes")
print(f"Forecast Horizon: {report['forecast_horizon_hours']}h ({report['forecast_horizon_steps']} steps)")
print(f"Available Features: {report['available_features']}")
print(f"Unavailable Features: {report['unavailable_features']}")

print("\n=== 2. TRAINING RESIDUAL LSTM ON AAU EXPERIMENTS ===")
history = train_lstm_pipeline(aau_csv, epochs=10, batch_size=64, stride=4)
print(f"Training Complete! Final Loss: {history['loss'][-1]:.6f}, Final Val Loss: {history['val_loss'][-1]:.6f}")

print("\n=== 3. EVALUATING HELD-OUT TEST PERFORMANCE & BASELINES ===")
metrics = run_model_evaluation(aau_csv, stride=6)
print("Models Evaluation:")
for m_name, m_data in metrics["models"].items():
    print(f"  {m_name:15s}: MAE={m_data['mae']:.4f}°C, RMSE={m_data['rmse']:.4f}°C, R²={m_data['r2']:.4f}")

print("\n=== 4. REFRESHING INFERENCE AND FORECAST ===")
pred_svc = get_prediction_service()
# Update active dataset path to AAU
pred_svc.dataset_svc.active_dataset_path = aau_csv
forecast_data = pred_svc.get_latest_forecast(force_refresh=True, experiment_id="BaselineTest_100percent")
print("Forecast generated successfully!")
pred = forecast_data.get("prediction", {})
cooling = forecast_data.get("cooling", {})
print(f"Experiment ID: {pred.get('experiment_id')}")
print(f"Forecast Steps: {pred.get('prediction_count')}")
print(f"Forecast Range: {pred.get('min_temperature')}°C to {pred.get('max_temperature')}°C (Avg: {pred.get('avg_temperature')}°C)")
print(f"First 5 steps: {pred.get('forecast_temperatures', [])[:5]}")
print(f"Timestamps: {pred.get('forecast_timestamps', [])[:3]}")
print(f"Chiller State: {cooling.get('current_state')}, Predicted running steps: {cooling.get('predicted_running_steps')}")
print("ALL PIPELINE STAGES COMPLETED!")
