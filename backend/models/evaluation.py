"""
Model Benchmark Evaluation Module
Calculates MAE, RMSE, and R2 against Random Forest and Persistence Baselines
with multi-schema support and experimental boundary isolation.
"""
import os
import sys
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ.setdefault("KERAS_HOME", os.path.join(BASE_DIR, ".keras"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf

from backend.config import Config
from backend.preprocessing.data_loader import load_and_parse_csv
from backend.preprocessing.cleaning import clean_and_repair_dataset
from backend.preprocessing.normalization import DataCenterScaler
from backend.preprocessing.sequence_generator import split_chronologically, create_experiment_isolated_sequences
from backend.preprocessing.validation import detect_schema, detect_sampling_interval_and_horizon

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true.flatten(), y_pred.flatten()))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4)
    }

def run_model_evaluation(dataset_path: str, stride: Optional[int] = None) -> dict:
    """
    Evaluates LSTM against benchmark baselines on strictly held-out test split
    while strictly isolating experiment sequences.
    """
    if not os.path.exists(Config.MODEL_PATH) or not os.path.exists(Config.SCALER_PATH):
        raise FileNotFoundError("Model or Scaler not found. Train the model first.")
        
    df = load_and_parse_csv(dataset_path)
    clean_df, _ = clean_and_repair_dataset(df)
    
    schema_type, schema_name = detect_schema(clean_df)
    interval_mins, default_horizon = detect_sampling_interval_and_horizon(clean_df, schema=schema_type)
    
    # 1. Load LSTM model and inspect actual trained dimensions
    lstm_model = tf.keras.models.load_model(Config.MODEL_PATH, compile=False)
    input_window = int(lstm_model.input_shape[1]) if lstm_model.input_shape else default_horizon
    horizon_steps = int(lstm_model.output_shape[1]) if lstm_model.output_shape else default_horizon
    test_stride = stride or max(4, min(24, len(clean_df) // 1000))

    train_df, val_df, test_df = split_chronologically(clean_df, 0.70, 0.15, 0.15)
    
    scaler = DataCenterScaler.load(Config.SCALER_PATH)
    X_test, y_test_scaled = create_experiment_isolated_sequences(
        test_df, scaler, input_window, horizon_steps, stride=test_stride
    )
    y_test_true = scaler.inverse_transform_target(y_test_scaled)
    
    # Evaluate LSTM
    y_pred_lstm_scaled = lstm_model.predict(X_test, batch_size=64, verbose=0)
    y_pred_lstm = scaler.inverse_transform_target(y_pred_lstm_scaled)
    lstm_metrics = calculate_metrics(y_test_true, y_pred_lstm)
    
    # High-temperature event evaluation (>= 28.0°C)
    mask_high = (y_test_true >= 28.0)
    high_count = int(np.sum(mask_high))
    if high_count > 0:
        high_act_mean = float(np.mean(y_test_true[mask_high]))
        high_pred_mean = float(np.mean(y_pred_lstm[mask_high]))
        high_mae = float(mean_absolute_error(y_test_true[mask_high], y_pred_lstm[mask_high]))
        high_rmse = float(np.sqrt(mean_squared_error(y_test_true[mask_high], y_pred_lstm[mask_high])))
        high_act_peak = float(np.max(y_test_true[mask_high]))
        high_pred_peak = float(np.max(y_pred_lstm[mask_high]))
    else:
        high_act_mean, high_pred_mean, high_mae, high_rmse = 0.0, 0.0, 0.0, 0.0
        high_act_peak, high_pred_peak = 0.0, 0.0

    mask_normal = (y_test_true < 28.0)
    normal_mae = float(mean_absolute_error(y_test_true[mask_normal], y_pred_lstm[mask_normal])) if np.sum(mask_normal) > 0 else 0.0

    # 2. Evaluate Baseline: Persistence
    last_known_scaled = X_test[:, -1, 0:1]
    y_pred_persist_scaled = np.repeat(last_known_scaled, horizon_steps, axis=1)
    y_pred_persist = scaler.inverse_transform_target(y_pred_persist_scaled)
    persist_metrics = calculate_metrics(y_test_true, y_pred_persist)
    
    # 3. Evaluate Baseline: Random Forest
    rf_stride = max(test_stride * 2, 8)
    X_train_sub, y_train_sub = create_experiment_isolated_sequences(
        train_df, scaler, input_window, horizon_steps, stride=rf_stride
    )
    X_rf_train = X_train_sub.reshape(len(X_train_sub), -1)
    X_rf_test = X_test.reshape(len(X_test), -1)
    
    rf = RandomForestRegressor(n_estimators=25, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_rf_train, y_train_sub)
    y_pred_rf_scaled = rf.predict(X_rf_test)
    y_pred_rf = scaler.inverse_transform_target(y_pred_rf_scaled)
    rf_metrics = calculate_metrics(y_test_true, y_pred_rf)

    # Sample forecast slice for plotting
    sample_idx = 0
    sample_forecast = {
        "actual": [round(float(v), 2) for v in y_test_true[sample_idx]],
        "predicted": [round(float(v), 2) for v in y_pred_lstm[sample_idx]],
        "lstm_pred": [round(float(v), 2) for v in y_pred_lstm[sample_idx]],
        "rf_pred": [round(float(v), 2) for v in y_pred_rf[sample_idx]],
        "persistence_pred": [round(float(v), 2) for v in y_pred_persist[sample_idx]],
        "error_lstm": [round(float(abs(y_test_true[sample_idx][i] - y_pred_lstm[sample_idx][i])), 2) for i in range(horizon_steps)]
    }

    metrics_payload = {
        "dataset_source": os.path.basename(dataset_path),
        "dataset_records": len(clean_df),
        "model_version": f"LSTM-{horizon_steps}step-v1",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_configuration": list(scaler.feature_columns),
        "target_column": scaler.target_column,
        "schema_type": schema_type,
        "schema_name": schema_name,
        "forecast_horizon_steps": horizon_steps,
        "forecast_horizon_hours": round((horizon_steps * interval_mins) / 60.0, 1),
        "input_window_steps": input_window,
        "sampling_interval_minutes": interval_mins,
        "test_samples_count": len(X_test),
        "models": {
            "LSTM": {
                "name": "Residual Stacked LSTM Neural Network",
                "mae": lstm_metrics["mae"],
                "rmse": lstm_metrics["rmse"],
                "r2": lstm_metrics["r2"],
                "description": f"Residual LSTM with persistence anchor ({horizon_steps}-step horizon)"
            },
            "Random_Forest": {
                "name": "Random Forest Regressor",
                "mae": rf_metrics["mae"],
                "rmse": rf_metrics["rmse"],
                "r2": rf_metrics["r2"],
                "description": "Multi-output tree ensemble regressor"
            },
            "Persistence": {
                "name": "Persistence Baseline",
                "mae": persist_metrics["mae"],
                "rmse": persist_metrics["rmse"],
                "r2": persist_metrics["r2"],
                "description": "Propagates last observed temperature reading across horizon"
            }
        },
        "high_temperature_evaluation": {
            "high_temp_threshold_celsius": 28.0,
            "high_temp_steps_count": high_count,
            "actual_peak_celsius": round(high_act_peak, 2),
            "predicted_peak_celsius": round(high_pred_peak, 2),
            "actual_mean_celsius": round(high_act_mean, 2),
            "predicted_mean_celsius": round(high_pred_mean, 2),
            "high_temp_mae": round(high_mae, 4),
            "high_temp_rmse": round(high_rmse, 4),
            "normal_temp_mae": round(normal_mae, 4)
        },
        "sample_forecast": sample_forecast
    }

    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    with open(Config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    return metrics_payload
