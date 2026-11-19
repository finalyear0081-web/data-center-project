"""
LSTM Neural Network Architecture Builder with Thermal-Aware Multi-Step Forecasting
"""
import os
os.environ.setdefault("KERAS_HOME", os.path.abspath(".keras"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, RepeatVector, Reshape, Add
from tensorflow.keras.optimizers import Adam
from typing import Tuple

@tf.keras.utils.register_keras_serializable(package="CustomLoss")
def thermal_weighted_loss(y_true, y_pred):
    """
    Thermal-Weighted Loss Function.
    Penalizes underprediction of high temperatures (upper half of normalized range)
    to prevent regression-to-the-mean from smoothing out critical peak thermal spikes.
    """
    diff = y_true - y_pred
    squared_err = tf.square(diff)
    # Critical threshold in normalized [0, 1] space: upper 45% (typically >= 27.5-28°C)
    high_temp_mask = tf.cast(tf.logical_and(y_true >= 0.50, diff > 0.0), tf.float32)
    weights = 1.0 + 1.5 * high_temp_mask
    return tf.reduce_mean(weights * squared_err)

def build_lstm_forecasting_model(input_shape: Tuple[int, int] = (144, 3), 
                                 forecast_horizon: int = 144, 
                                 learning_rate: float = 0.001,
                                 use_thermal_loss: bool = True) -> Model:
    """
    Constructs and compiles the stacked Residual LSTM temperature forecasting model.
    Incorporates a persistent baseline skip connection anchored to the last observed 
    inlet temperature, with the stacked LSTM learning the dynamic thermal delta trajectory.
    """
    inp = Input(shape=input_shape, name="sensor_features_input")
    
    # Anchor to the last observed inlet temperature (feature column 0)
    last_inlet = inp[:, -1, 0:1]
    base_line = Reshape((forecast_horizon,), name="persistence_anchor")(RepeatVector(forecast_horizon)(last_inlet))
    
    # Stacked LSTM layers to extract multi-step thermal inertia and trends
    x = LSTM(64, return_sequences=True, name="lstm_layer_1")(inp)
    x = Dropout(0.1, name="dropout_1")(x)
    x = LSTM(64, return_sequences=False, name="lstm_layer_2")(x)
    x = Dropout(0.1, name="dropout_2")(x)
    x = Dense(64, activation="relu", name="dense_intermediate")(x)
    delta = Dense(forecast_horizon, activation="linear", name="learned_thermal_delta")(x)
    
    # Output: current temperature baseline + dynamic thermal delta
    out = Add(name="temperature_forecast_output")([base_line, delta])
    
    model = Model(inputs=inp, outputs=out, name="DataCenter_Residual_LSTM_Model")
    optimizer = Adam(learning_rate=learning_rate)
    loss_fn = thermal_weighted_loss if use_thermal_loss else "mse"
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["mae", "mse"])
    return model

def build_stacked_lstm_model(input_shape: Tuple[int, int] = (144, 3), 
                             forecast_horizon: int = 144, 
                             learning_rate: float = 0.001,
                             use_thermal_loss: bool = True) -> Model:
    """
    Constructs standard Stacked Multi-Layer LSTM without persistence skip connection.
    """
    inp = Input(shape=input_shape, name="sensor_features_input")
    x = LSTM(64, return_sequences=True, name="lstm_layer_1")(inp)
    x = Dropout(0.1, name="dropout_1")(x)
    x = LSTM(64, return_sequences=False, name="lstm_layer_2")(x)
    x = Dropout(0.1, name="dropout_2")(x)
    x = Dense(64, activation="relu", name="dense_intermediate")(x)
    out = Dense(forecast_horizon, activation="linear", name="temperature_forecast_output")(x)
    
    model = Model(inputs=inp, outputs=out, name="DataCenter_Stacked_LSTM_Model")
    optimizer = Adam(learning_rate=learning_rate)
    loss_fn = thermal_weighted_loss if use_thermal_loss else "mse"
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["mae", "mse"])
    return model

def build_model_by_architecture(architecture: str = "RESIDUAL_LSTM",
                                input_shape: Tuple[int, int] = (144, 3),
                                forecast_horizon: int = 144,
                                learning_rate: float = 0.001,
                                use_thermal_loss: bool = True) -> Model:
    arch = str(architecture).strip().upper()
    if arch in ["STACKED_LSTM", "STACKED", "STANDARD_LSTM"]:
        return build_stacked_lstm_model(input_shape, forecast_horizon, learning_rate, use_thermal_loss)
    return build_lstm_forecasting_model(input_shape, forecast_horizon, learning_rate, use_thermal_loss)


