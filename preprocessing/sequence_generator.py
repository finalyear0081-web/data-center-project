"""
Sequence Generator Module for Sliding Window Time-Series
Prepares chronological train/val/test splits and sliding-window sequences for LSTM training.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

DEFAULT_INPUT_WINDOW = 288   # 24 hours * 12 (5-min intervals)
DEFAULT_HORIZON = 288        # 24 hours * 12 (5-min intervals)

def split_chronological(df: pd.DataFrame, 
                        train_ratio: float = 0.70, 
                        val_ratio: float = 0.15, 
                        test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time-series data chronologically to strictly prevent temporal leakage.
    
    Args:
        df: Chronologically sorted DataFrame.
        train_ratio: Fraction for training (default 0.70).
        val_ratio: Fraction for validation (default 0.15).
        test_ratio: Fraction for testing (default 0.15).
        
    Returns:
        (train_df, val_df, test_df)
    """
    total = len(df)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df

def create_sliding_sequences(features: np.ndarray, 
                             target: np.ndarray, 
                             input_window: int = DEFAULT_INPUT_WINDOW, 
                             forecast_horizon: int = DEFAULT_HORIZON,
                             stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sliding-window input sequences and corresponding forecast targets.
    
    Input shape X: (num_samples, input_window, num_features)
    Target shape y: (num_samples, forecast_horizon)
    
    Args:
        features: Scaled feature array of shape (N, num_features).
        target: Scaled target 1D or (N, 1) array.
        input_window: Length of input history (default 288 steps).
        forecast_horizon: Length of output forecast (default 288 steps).
        stride: Step stride between consecutive sliding windows.
        
    Returns:
        X: numpy array of shape (samples, input_window, features)
        y: numpy array of shape (samples, forecast_horizon)
    """
    target = target.flatten()
    total_len = len(features)
    window_total = input_window + forecast_horizon
    
    if total_len < window_total:
        raise ValueError(f"Data length ({total_len}) is smaller than input_window + horizon ({window_total}).")
        
    indices = range(0, total_len - window_total + 1, stride)
    
    X_list = []
    y_list = []
    
    for i in indices:
        X_list.append(features[i : i + input_window])
        y_list.append(target[i + input_window : i + window_total])
        
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    return X, y
