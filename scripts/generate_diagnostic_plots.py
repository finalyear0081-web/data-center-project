"""
HazardNet LSTM Forecast Diagnostic Plot Generator
Generates a 4-panel publication-grade diagnostic visualization:
1. Historical Actual Temperature (Full Year 2019)
2. Actual vs Predicted on Unseen Chronological Test Data
3. Forecasted Next 24 Hours with Chiller Thresholds
4. Training and Validation Loss Convergence
"""
import os
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.config import Config
from backend.preprocessing.data_loader import load_and_parse_csv

def generate_diagnostic_plots(output_png_path: str = None) -> str:
    if output_png_path is None:
        output_png_path = os.path.join(Config.STORAGE_DIR, "hazardnet_forecast_diagnostic.png")
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)

    # 1. Load Data
    df = load_and_parse_csv(Config.DEFAULT_DATASET_PATH)
    inlet_temps = df["inlet_temperature"].values
    timestamps = pd.to_datetime(df["timestamp"])

    # 2. Load Evaluation Metrics and History
    with open(Config.EVALUATION_METRICS_PATH, "r", encoding="utf-8") as f:
        eval_metrics = json.load(f)

    with open(Config.TRAINING_HISTORY_PATH, "r", encoding="utf-8") as f:
        train_hist = json.load(f)

    with open(os.path.join(Config.STORAGE_DIR, "latest_forecast.json"), "r", encoding="utf-8") as f:
        latest_fc = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=150)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # -------------------------------------------------------------
    # Panel 1: Historical Actual Temperature (Full Year 2019)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot(timestamps, inlet_temps, color="#2563eb", alpha=0.6, linewidth=0.8, label="Marconi A2 Cluster Mean Inlet")
    # 7-day rolling mean
    rolling_7d = df["inlet_temperature"].rolling(144 * 7, min_periods=1).mean()
    ax1.plot(timestamps, rolling_7d, color="#1e40af", linewidth=1.8, label="7-Day Rolling Trend")
    ax1.axhline(28.0, color="#dc2626", linestyle="--", linewidth=1.2, label="Chiller ON Threshold (28°C)")
    ax1.axhline(26.0, color="#16a34a", linestyle="--", linewidth=1.2, label="Chiller OFF Threshold (26°C)")
    ax1.set_title("Panel 1: Marconi A2 Cluster Mean Inlet Temperature (2019 Full Year)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Inlet Temperature (°C)", fontsize=10)
    ax1.set_xlabel("Date", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # Panel 2: Actual vs Predicted on Unseen Chronological Test Data
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    sample = eval_metrics.get("sample_forecast", {})
    act_sample = sample.get("actual", [])
    lstm_sample = sample.get("lstm", [])
    persist_sample = sample.get("persistence", [])
    x_steps = np.arange(len(act_sample)) * 10 / 60.0  # Hours

    ax2.plot(x_steps, act_sample, color="#0f172a", linewidth=2.2, label="Ground Truth (Unseen Test)")
    ax2.plot(x_steps, lstm_sample, color="#2563eb", linewidth=2.0, linestyle="-", label="Residual Stacked LSTM")
    ax2.plot(x_steps, persist_sample, color="#94a3b8", linewidth=1.4, linestyle=":", label="Persistence Baseline")
    ax2.set_title("Panel 2: Actual vs Predicted Trajectory (Held-Out Test Sample)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Forecast Horizon (Hours into Future)", fontsize=10)
    ax2.set_ylabel("Temperature (°C)", fontsize=10)
    lstm_m = eval_metrics["models"]["LSTM"]
    pers_m = eval_metrics["models"]["Persistence"]
    txt = f"LSTM MAE: {lstm_m['mae']:.3f}°C | R²: {lstm_m['r2']:.3f}\nPersistence MAE: {pers_m['mae']:.3f}°C | R²: {pers_m['r2']:.3f}"
    ax2.text(0.03, 0.08, txt, transform=ax2.transAxes, fontsize=8,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cbd5e1", alpha=0.9))
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # Panel 3: Forecasted Next 24 Hours with Thresholds
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    pred = latest_fc["prediction"]
    cooling = latest_fc["cooling"]
    fc_temps = pred["forecast_temperatures"]
    fc_steps = np.arange(len(fc_temps)) * 10 / 60.0  # Hours

    ax3.plot(fc_steps, fc_temps, color="#2563eb", linewidth=2.5, label="24-Hour LSTM Forecast")
    ax3.axhline(cooling["on_threshold"], color="#dc2626", linestyle="--", linewidth=1.5, label=f"Chiller ON Limit ({cooling['on_threshold']}°C)")
    ax3.axhline(cooling["off_threshold"], color="#16a34a", linestyle="--", linewidth=1.5, label=f"Chiller OFF Limit ({cooling['off_threshold']}°C)")
    ax3.fill_between(fc_steps, cooling["off_threshold"], cooling["on_threshold"], color="#e2e8f0", alpha=0.4, label="Hysteresis Band")
    ax3.set_title("Panel 3: Corrected Next 24-Hour Forecast (Dec 31 23:00 to Jan 1 22:50)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Hours Ahead", fontsize=10)
    ax3.set_ylabel("Server Inlet Temperature (°C)", fontsize=10)
    
    fc_info = f"Min: {np.min(fc_temps):.2f}°C | Max: {np.max(fc_temps):.2f}°C\nSpan: {np.max(fc_temps)-np.min(fc_temps):.2f}°C | Std: {np.std(fc_temps):.3f}°C\nChiller Status: {cooling['current_state']}"
    ax3.text(0.03, 0.08, fc_info, transform=ax3.transAxes, fontsize=8,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cbd5e1", alpha=0.9))
    ax3.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # Panel 4: Training and Validation Loss Curve
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    epochs_range = range(1, len(train_hist["loss"]) + 1)
    ax4.plot(epochs_range, train_hist["loss"], marker="o", color="#2563eb", linewidth=1.8, label="Training Loss (Thermal Weighted MSE)")
    ax4.plot(epochs_range, train_hist["val_loss"], marker="s", color="#f59e0b", linewidth=1.8, label="Validation Loss")
    ax4.set_title("Panel 4: Neural Network Learning Curve (MSE vs Epoch)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Epoch", fontsize=10)
    ax4.set_ylabel("Loss", fontsize=10)
    ax4.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax4.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("HazardNet Marconi A2 Supercomputer — LSTM Forecast & Benchmark Diagnostic", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(output_png_path, bbox_inches="tight")
    plt.close()
    print(f"Diagnostic plot successfully generated at: {output_png_path}")
    return output_png_path

if __name__ == "__main__":
    generate_diagnostic_plots()
