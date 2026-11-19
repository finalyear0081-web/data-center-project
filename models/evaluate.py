"""
Model Evaluation and Benchmark Comparison Module
Evaluates trained LSTM, Random Forest, and Persistence baseline models on the exact chronological test split.
Calculates MAE, RMSE, and R2 without inventing values.
"""
import os
import sys
import json
import time

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("KERAS_HOME", os.path.abspath(".keras"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from preprocessing.data_loader import load_dataset
from preprocessing.cleaning import validate_and_clean_dataset
from preprocessing.normalization import DataCenterScaler
from preprocessing.sequence_generator import split_chronological, create_sliding_sequences, DEFAULT_INPUT_WINDOW, DEFAULT_HORIZON

MODEL_PATH = "models/saved/lstm_temperature_model.keras"
SCALER_PATH = "models/saved/scaler.pkl"
EVAL_PATH = "models/saved/evaluation_metrics.json"

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes MAE, RMSE, and R2 on unscaled Celsius values."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true.flatten(), y_pred.flatten()))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4)
    }

def evaluate_all_models(dataset_path: str = "dataset/data_center_temperature_dataset.csv",
                        input_window: int = DEFAULT_INPUT_WINDOW,
                        forecast_horizon: int = DEFAULT_HORIZON,
                        test_stride: int = 24) -> dict:
    """
    Evaluates LSTM against benchmark baselines on the strictly held-out chronological test split.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Train the model first.")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}.")
        
    print("Loading test data and models...")
    df = load_dataset(dataset_path)
    clean_df, _ = validate_and_clean_dataset(df)
    train_df, val_df, test_df = split_chronological(clean_df, 0.70, 0.15, 0.15)
    
    scaler = DataCenterScaler.load(SCALER_PATH)
    
    test_feat = scaler.transform_features(test_df)
    test_targ = scaler.transform_target(test_df)
    
    # Generate test sequences
    X_test, y_test_scaled = create_sliding_sequences(test_feat, test_targ, input_window, forecast_horizon, stride=test_stride)
    y_test_true = scaler.inverse_transform_target(y_test_scaled)
    
    print(f"Test samples generated: {len(X_test)} sequences.")
    
    # 1. Evaluate LSTM
    print("Evaluating LSTM...")
    lstm_model = tf.keras.models.load_model(MODEL_PATH)
    y_pred_lstm_scaled = lstm_model.predict(X_test, batch_size=64, verbose=0)
    y_pred_lstm = scaler.inverse_transform_target(y_pred_lstm_scaled)
    lstm_metrics = calculate_metrics(y_test_true, y_pred_lstm)
    
    # 2. Benchmark 1: Persistence Baseline (repeats the last observed temperature across horizon)
    print("Evaluating Baseline Persistence...")
    # Last step of target in input window
    # In features, inlet_temperature is column 0 in NUMERIC_COLUMNS
    last_known_inlet_scaled = X_test[:, -1, 0:1] # shape (samples, 1)
    y_pred_persist_scaled = np.repeat(last_known_inlet_scaled, forecast_horizon, axis=1)
    y_pred_persist = scaler.inverse_transform_target(y_pred_persist_scaled)
    persist_metrics = calculate_metrics(y_test_true, y_pred_persist)
    
    # 3. Benchmark 2: Random Forest Regressor (flattened summary statistics / multioutput)
    print("Evaluating Random Forest Regressor...")
    # To keep training fast and memory-safe, fit RF on a subset of training data
    train_feat = scaler.transform_features(train_df)
    train_targ = scaler.transform_target(train_df)
    X_train_sub, y_train_sub = create_sliding_sequences(train_feat, train_targ, input_window, forecast_horizon, stride=48)
    # Flatten input window (samples, 288*9) -> target (samples, 288)
    X_rf_train = X_train_sub.reshape(len(X_train_sub), -1)
    X_rf_test = X_test.reshape(len(X_test), -1)
    
    rf = RandomForestRegressor(n_estimators=30, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_rf_train, y_train_sub)
    y_pred_rf_scaled = rf.predict(X_rf_test)
    y_pred_rf = scaler.inverse_transform_target(y_pred_rf_scaled)
    rf_metrics = calculate_metrics(y_test_true, y_pred_rf)
    
    # Extract representative sample for visualization (e.g. sample #10)
    sample_idx = min(10, len(X_test) - 1)
    sample_actual = y_test_true[sample_idx].tolist()
    sample_lstm = y_pred_lstm[sample_idx].tolist()
    sample_rf = y_pred_rf[sample_idx].tolist()
    sample_persist = y_pred_persist[sample_idx].tolist()
    
    results = {
        "models": {
            "LSTM": {
                "name": "LSTM Neural Network",
                "mae": lstm_metrics["mae"],
                "rmse": lstm_metrics["rmse"],
                "r2": lstm_metrics["r2"],
                "description": "2-layer stacked LSTM with Dropout & Dense multi-step head"
            },
            "Random_Forest": {
                "name": "Random Forest Regressor",
                "mae": rf_metrics["mae"],
                "rmse": rf_metrics["rmse"],
                "r2": rf_metrics["r2"],
                "description": "Multi-output Random Forest (30 estimators, max depth 12)"
            },
            "Persistence": {
                "name": "Persistence Baseline (Last Known)",
                "mae": persist_metrics["mae"],
                "rmse": persist_metrics["rmse"],
                "r2": persist_metrics["r2"],
                "description": "Naive persistence benchmark propagating last observed value"
            }
        },
        "test_samples_count": len(X_test),
        "forecast_horizon_steps": forecast_horizon,
        "sample_forecast": {
            "actual": [round(x, 2) for x in sample_actual],
            "lstm_pred": [round(x, 2) for x in sample_lstm],
            "rf_pred": [round(x, 2) for x in sample_rf],
            "persistence_pred": [round(x, 2) for x in sample_persist],
            "error_lstm": [round(abs(a - p), 2) for a, p in zip(sample_actual, sample_lstm)]
        }
    }
    
    with open(EVAL_PATH, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n--- Model Benchmark Results ---")
    for m_key, m_info in results["models"].items():
        print(f"{m_info['name']}: MAE={m_info['mae']} °C, RMSE={m_info['rmse']} °C, R²={m_info['r2']}")
        
    return results

if __name__ == "__main__":
    evaluate_all_models()
