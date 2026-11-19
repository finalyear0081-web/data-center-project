"""
Model Training Script and Background Pipeline
Supports multi-schema dataset training:
- HazardNet Operational Telemetry (24-hour horizon)
- AAU Experimental Data (1-hour horizon, 16 isolated experiments)
"""
import os
import sys
import json
import time
from datetime import datetime, timezone

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ.setdefault("KERAS_HOME", os.path.join(BASE_DIR, ".keras"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from backend.config import Config
from backend.preprocessing.data_loader import load_and_parse_csv
from backend.preprocessing.cleaning import clean_and_repair_dataset
from backend.preprocessing.normalization import DataCenterScaler
from backend.preprocessing.sequence_generator import split_chronologically, create_experiment_isolated_sequences
from backend.preprocessing.validation import detect_schema, detect_sampling_interval_and_horizon
from backend.preprocessing.suitability_engine import DatasetSuitabilityEngine
from backend.models.lstm_model import build_model_by_architecture

def update_training_status(status: str, progress: float = 0.0, message: str = "", extra: dict = None):
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    payload = {
        "status": status,
        "progress": progress,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **(extra or {})
    }
    with open(Config.TRAINING_STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def train_lstm_pipeline(csv_path: str,
                        epochs: int = 10,
                        batch_size: int = 64,
                        stride: int = None,
                        architecture: str = "RESIDUAL_LSTM",
                        manual_overrides: dict = None) -> dict:
    """
    Executes end-to-end dataset preprocessing, train-only normalization,
    experiment-isolated sliding-window sequence generation, and configurable LSTM training.
    """
    start_time = time.time()
    update_training_status("RUNNING", 0.05, f"Loading and validating dataset for training ({architecture})...")
    
    raw_df = load_and_parse_csv(csv_path, manual_overrides=manual_overrides)
    clean_df, clean_report = clean_and_repair_dataset(raw_df)
    
    # Run suitability analysis to get validated parameters
    suitability = DatasetSuitabilityEngine.evaluate_dataset(clean_df, manual_overrides=manual_overrides)
    schema_type, schema_name = detect_schema(clean_df)
    
    interval_mins, horizon_steps = detect_sampling_interval_and_horizon(clean_df, schema=schema_type)
    
    # Check if dataset contains multiple short experiments (e.g. genuine AAU benchmark)
    is_short_experiment_benchmark = False
    if "experiment_id" in clean_df.columns and clean_df["experiment_id"].nunique() > 1:
        min_exp_len = clean_df.groupby("experiment_id").size().min()
        if min_exp_len < 1000:
            is_short_experiment_benchmark = True

    if is_short_experiment_benchmark:
        input_window = 60   # 2 hours input history for 720-step experiment
        horizon_steps = 30  # Validated 1-hour horizon
        stride = stride or 4
    else:
        # Full 24-hour forecasting horizon
        if interval_mins == 2:
            horizon_steps = 720   # 720 steps * 2 min = 1440 min = 24.0 hours
            input_window = 360    # 12 hours of historical context
            stride = stride or 12
        elif interval_mins == 10:
            horizon_steps = 144   # 144 steps * 10 min = 1440 min = 24.0 hours
            input_window = 144    # 24 hours of context
            stride = stride or 12
        elif interval_mins == 5:
            horizon_steps = 288   # 288 steps * 5 min = 1440 min = 24.0 hours
            input_window = 288
            stride = stride or 12
        else:
            horizon_steps = max(1, int(round(1440.0 / interval_mins)))
            input_window = min(horizon_steps, 288)
            stride = stride or max(2, min(12, len(clean_df) // 2000))

    min_needed = (input_window + horizon_steps) * 2
    if len(clean_df) < min_needed:
        raise ValueError(
            f"Dataset has {len(clean_df)} records, but at least {min_needed} records are required for sequence training."
        )

    num_exps = clean_df["experiment_id"].nunique() if "experiment_id" in clean_df.columns else 1
    update_training_status(
        "RUNNING", 0.15,
        f"Detected {schema_name} ({num_exps} exps, {interval_mins}m interval). "
        f"Arch: {architecture}, Input: {input_window} steps, Horizon: {horizon_steps} steps. Splitting chronologically..."
    )
    
    train_df, val_df, test_df = split_chronologically(clean_df, 0.70, 0.15, 0.15)
    
    update_training_status("RUNNING", 0.25, "Fitting MinMax scaler strictly on training data...")
    scaler = DataCenterScaler()
    scaler.fit(train_df)
    
    update_training_status("RUNNING", 0.35, "Generating experiment-isolated sliding-window sequences...")
    X_train, y_train = create_experiment_isolated_sequences(train_df, scaler, input_window, horizon_steps, stride=stride)
    X_val, y_val = create_experiment_isolated_sequences(val_df, scaler, input_window, horizon_steps, stride=stride)
    
    update_training_status("RUNNING", 0.50, f"Building {architecture} model for {horizon_steps} output horizon...")
    model = build_model_by_architecture(
        architecture=architecture,
        input_shape=(input_window, X_train.shape[2]),
        forecast_horizon=horizon_steps
    )
    
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=Config.MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    ]
    
    update_training_status("RUNNING", 0.60, f"Training {architecture} for up to {epochs} epochs on {len(X_train)} sequences...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Save fitted scaler only after successful model convergence
    scaler.save(Config.SCALER_PATH)

    # Save training loss history and update training_history.json
    loss_payload = {
        "loss": [float(x) for x in history.history.get("loss", [])],
        "val_loss": [float(x) for x in history.history.get("val_loss", [])],
        "epochs": len(history.history.get("loss", [])),
        "architecture": architecture,
        "schema_type": schema_type,
        "horizon_steps": horizon_steps,
        "forecast_horizon_hours": round((horizon_steps * interval_mins) / 60.0, 1),
        "input_window": input_window,
        "dataset_source": os.path.basename(csv_path),
        "feature_configuration": scaler.feature_columns,
        "target_column": scaler.target_column,
        "trained_at": datetime.now(timezone.utc).isoformat()
    }
    loss_history_path = os.path.join(Config.MODELS_DIR, "loss_history.json")
    with open(loss_history_path, "w", encoding="utf-8") as f:
        json.dump(loss_payload, f, indent=2)

    # Synchronize training_history.json to eliminate stale mock/old results
    training_history_path = os.path.join(Config.MODELS_DIR, "training_history.json")
    with open(training_history_path, "w", encoding="utf-8") as f:
        json.dump(loss_payload, f, indent=2)

    elapsed_time = round(time.time() - start_time, 2)
    update_training_status(
        "COMPLETED", 1.0,
        f"Training completed successfully in {elapsed_time}s. Best val_loss: {min(history.history['val_loss']):.6f}",
        extra={
            "epochs_trained": len(history.history["loss"]),
            "final_train_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "duration_seconds": elapsed_time,
            "architecture": architecture,
            "schema_type": schema_type,
            "total_sequences": len(X_train)
        }
    )

    return history.history
