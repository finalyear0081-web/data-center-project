"""
Sequence Generator Module
Generates chronological train/val/test splits and sliding window sequences.
Strictly preserves experimental boundaries: never concatenates or crosses
different experiment_id time series when generating sliding window sequences.
"""
from typing import Tuple, List
import numpy as np
import pandas as pd

def split_chronologically(df: pd.DataFrame, 
                          train_ratio: float = 0.70, 
                          val_ratio: float = 0.15, 
                          test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits time-series chronologically into Train, Validation, and Test sets.
    If multiple experiments exist (AAU), splits chronologically strictly WITHIN each experiment,
    then combines the respective splits to preserve chronological integrity across all experiments.
    """
    if "experiment_id" in df.columns and df["experiment_id"].nunique() > 1:
        train_parts = []
        val_parts = []
        test_parts = []
        
        for _, exp_group in df.groupby("experiment_id", sort=False):
            n = len(exp_group)
            t_end = int(n * train_ratio)
            v_end = int(n * (train_ratio + val_ratio))
            
            train_parts.append(exp_group.iloc[:t_end])
            val_parts.append(exp_group.iloc[t_end:v_end])
            test_parts.append(exp_group.iloc[v_end:])
            
        train_df = pd.concat(train_parts, ignore_index=True)
        val_df = pd.concat(val_parts, ignore_index=True)
        test_df = pd.concat(test_parts, ignore_index=True)
        return train_df, val_df, test_df

    # Single continuous dataset (HazardNet)
    total = len(df)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df

def create_sliding_sequences(features: np.ndarray, 
                             target: np.ndarray, 
                             input_window: int = 144, 
                             forecast_horizon: int = 144,
                             stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates input sequences X (samples, input_window, features) and multi-step targets y (samples, forecast_horizon)
    for a contiguous time series block.
    """
    target = target.flatten()
    total_len = len(features)
    window_total = input_window + forecast_horizon
    
    if total_len < window_total:
        return np.empty((0, input_window, features.shape[1]), dtype=np.float32), np.empty((0, forecast_horizon), dtype=np.float32)
        
    indices = range(0, total_len - window_total + 1, stride)
    
    X_list = []
    y_list = []
    for i in indices:
        X_list.append(features[i : i + input_window])
        y_list.append(target[i + input_window : i + window_total])
        
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

def create_experiment_isolated_sequences(df: pd.DataFrame,
                                         scaler,
                                         input_window: int,
                                         forecast_horizon: int,
                                         stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates training sequences independently within each experiment_id.
    Guarantees that NO input sequence ever starts in one experiment and ends in another.
    """
    if "experiment_id" in df.columns and df["experiment_id"].nunique() > 1:
        X_all = []
        y_all = []
        for exp_id, exp_group in df.groupby("experiment_id", sort=False):
            if len(exp_group) < (input_window + forecast_horizon):
                continue
            feat = scaler.transform_features(exp_group)
            targ = scaler.transform_target(exp_group)
            X_exp, y_exp = create_sliding_sequences(feat, targ, input_window, forecast_horizon, stride=stride)
            if len(X_exp) > 0:
                X_all.append(X_exp)
                y_all.append(y_exp)
                
        if not X_all:
            raise ValueError("No valid sequences could be generated from experiments. Check window dimensions.")
        return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)

    # Single contiguous dataset
    feat = scaler.transform_features(df)
    targ = scaler.transform_target(df)
    return create_sliding_sequences(feat, targ, input_window, forecast_horizon, stride=stride)
