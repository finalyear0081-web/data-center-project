"""
Model Training Script for Data Center Temperature Prediction
Executes full preprocessing, train-only normalization, sliding sequence creation,
and model fitting with EarlyStopping and ModelCheckpoint.
"""
import os
import sys
import json
import time
from datetime import datetime

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Sandbox environment variable configuration
os.environ.setdefault("KERAS_HOME", os.path.abspath(".keras"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from preprocessing.data_loader import load_dataset
from preprocessing.cleaning import validate_and_clean_dataset
from preprocessing.normalization import DataCenterScaler
from preprocessing.sequence_generator import split_chronological, create_sliding_sequences, DEFAULT_INPUT_WINDOW, DEFAULT_HORIZON
from models.lstm_model import build_lstm_model

MODEL_DIR = "models/saved"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_temperature_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")
STATUS_PATH = os.path.join(MODEL_DIR, "training_status.json")

def update_training_status(status: str, progress: float = 0.0, message: str = "", extra: dict = None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    payload = {
        "status": status,
        "progress": progress,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
        **(extra or {})
    }
    with open(STATUS_PATH, "w") as f:
        json.dump(payload, f, indent=2)

def train_pipeline(dataset_path: str = "dataset/data_center_temperature_dataset.csv",
                   epochs: int = 15,
                   batch_size: int = 64,
                   input_window: int = DEFAULT_INPUT_WINDOW,
                   forecast_horizon: int = DEFAULT_HORIZON,
                   sample_stride: int = 12) -> dict:
    """
    Executes the complete training workflow.
    
    Args:
        dataset_path: Path to dataset CSV.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        input_window: Number of historical steps (288 = 24h).
        forecast_horizon: Number of forecast steps (288 = 24h).
        sample_stride: Stride in timesteps for sequence generation (12 = 1 hr stride).
        
    Returns:
        dict containing training summary metrics and status.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    update_training_status("RUNNING", 0.05, "Loading and cleaning dataset...")
    
    start_time = time.time()
    raw_df = load_dataset(dataset_path)
    clean_df, clean_report = validate_and_clean_dataset(raw_df)
    
    update_training_status("RUNNING", 0.15, "Splitting dataset chronologically...")
    train_df, val_df, test_df = split_chronological(clean_df, 0.70, 0.15, 0.15)
    
    update_training_status("RUNNING", 0.25, "Fitting scaler strictly on training split...")
    scaler = DataCenterScaler()
    scaler.fit(train_df)
    scaler.save(SCALER_PATH)
    
    update_training_status("RUNNING", 0.35, "Generating normalized sliding window sequences...")
    train_feat = scaler.transform_features(train_df)
    train_targ = scaler.transform_target(train_df)
    val_feat = scaler.transform_features(val_df)
    val_targ = scaler.transform_target(val_df)
    
    X_train, y_train = create_sliding_sequences(train_feat, train_targ, input_window, forecast_horizon, stride=sample_stride)
    X_val, y_val = create_sliding_sequences(val_feat, val_targ, input_window, forecast_horizon, stride=sample_stride)
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    
    update_training_status("RUNNING", 0.50, "Compiling LSTM Neural Network architecture...")
    model = build_lstm_model(input_shape=(input_window, X_train.shape[2]), forecast_horizon=forecast_horizon)
    
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    ]
    
    update_training_status("RUNNING", 0.60, f"Training LSTM model for up to {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    elapsed = time.time() - start_time
    
    # Save training history
    history_dict = {
        "loss": [float(x) for x in history.history.get("loss", [])],
        "val_loss": [float(x) for x in history.history.get("val_loss", [])],
        "mae": [float(x) for x in history.history.get("mae", [])],
        "val_mae": [float(x) for x in history.history.get("val_mae", [])],
        "epochs_completed": len(history.history.get("loss", [])),
        "training_duration_seconds": round(elapsed, 2),
        "best_val_loss": float(min(history.history.get("val_loss", [0.0]))),
        "final_loss": float(history.history.get("loss", [0.0])[-1]),
        "timestamp": datetime.utcnow().isoformat()
    }
    with open(HISTORY_PATH, "w") as f:
        json.dump(history_dict, f, indent=2)
        
    update_training_status("COMPLETED", 1.0, "Model training finished successfully.", extra=history_dict)
    print(f"Model successfully saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")
    print(f"Training completed in {elapsed:.1f} seconds. Best val_loss: {history_dict['best_val_loss']:.5f}")
    return history_dict

if __name__ == "__main__":
    train_pipeline(epochs=10, batch_size=64, sample_stride=12)
