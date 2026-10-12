"""Data loading, feature engineering, and preprocessing."""
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from backend.config import DATA_PATH, N_HOURS, N_QUBITS


def load_series(path: Path = DATA_PATH, n_hours: int = N_HOURS) -> np.ndarray:
    """Load first n_hours electricity demand from CSV, sorted chronologically."""
    if path.exists():
        df = pd.read_csv(path, parse_dates=["Datetime"]).sort_values("Datetime")
        return df["AEP_MW"].values[:n_hours].astype(float)

    # Fallback synthetic periodic series if CSV is unavailable
    t = np.arange(n_hours)
    return (
        500.0
        + 0.05 * t
        + 80.0 * np.sin(2 * np.pi * t / 24.0 - 1.5)
        + 40.0 * np.sin(2 * np.pi * t / (24.0 * 7.0))
        + np.random.normal(0, 8, n_hours)
    ).astype(float)


def make_lags(series: np.ndarray, n_lags: int = N_QUBITS) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window lag features: X[i] = series[i-n_lags:i], y[i] = series[i]."""
    X = np.array([series[i - n_lags : i] for i in range(n_lags, len(series))], dtype=float)
    y = series[n_lags:].astype(float)
    return X, y


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
) -> Tuple[MinMaxScaler, MinMaxScaler, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronologically split and scale data. Scalers are fit exclusively on train set."""
    split_idx = int(train_ratio * len(X))
    X_tr, X_te = X[:split_idx], X[split_idx:]
    y_tr, y_te = y[:split_idx], y[split_idx:]

    x_scaler = MinMaxScaler(feature_range=(0, 1)).fit(X_tr)
    y_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(y_tr.reshape(-1, 1))

    X_tr_s = x_scaler.transform(X_tr)
    X_te_s = np.clip(x_scaler.transform(X_te), 0.0, 1.0)
    y_tr_s = y_scaler.transform(y_tr.reshape(-1, 1)).ravel()

    return x_scaler, y_scaler, X_tr_s, y_tr_s, X_te_s, y_te
