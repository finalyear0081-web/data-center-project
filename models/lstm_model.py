"""
LSTM Neural Network Architecture Definition for Temperature Forecasting
Builds a multi-layer stacked LSTM network for multi-step data center inlet temperature forecasting.
"""
import os
# Ensure sandbox compatibility for Keras cache directory
os.environ.setdefault("KERAS_HOME", os.path.abspath(".keras"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from typing import Tuple

def build_lstm_model(input_shape: Tuple[int, int] = (288, 9), 
                     forecast_horizon: int = 288, 
                     learning_rate: float = 0.001) -> Sequential:
    """
    Constructs and compiles the LSTM multi-step temperature forecasting model.
    
    Architecture:
      Input (288 time-steps, 9 sensor features)
      -> LSTM(64 units, return_sequences=True)
      -> Dropout(0.2)
      -> LSTM(32 units)
      -> Dropout(0.2)
      -> Dense(32, activation='relu')
      -> Dense(288, activation='linear') [Forecast next 24 hours]
      
    Args:
        input_shape: Tuple of (input_timesteps, feature_count), default (288, 9)
        forecast_horizon: Number of output prediction steps, default 288
        learning_rate: Adam optimizer learning rate, default 0.001
        
    Returns:
        Compiled tf.keras.Sequential model.
    """
    model = Sequential([
        Input(shape=input_shape, name="sensor_sequence_input"),
        LSTM(64, return_sequences=True, name="lstm_layer_1"),
        Dropout(0.2, name="dropout_1"),
        LSTM(32, return_sequences=False, name="lstm_layer_2"),
        Dropout(0.2, name="dropout_2"),
        Dense(32, activation="relu", name="dense_intermediate"),
        Dense(forecast_horizon, activation="linear", name="temperature_forecast_output")
    ], name="DataCenter_LSTM_Predictor")
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae", "mse"]
    )
    return model

if __name__ == "__main__":
    test_model = build_lstm_model()
    test_model.summary()
