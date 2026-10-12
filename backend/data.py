"""Data loading, feature engineering, and preprocessing."""
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from backend.config import DATA_PATH, LAGS, N_ROWS


def load_series(path: Path = DATA_PATH, n_rows: int = N_ROWS) -> np.ndarray:
    """Load electricity demand with cleaning: parse, sort, drop duplicates, strict hourly reindex, interpolate, drop <= 0."""
    if path.exists():
        df = pd.read_csv(path, parse_dates=["Datetime"]).sort_values("Datetime")
        df = df.drop_duplicates(subset=["Datetime"], keep="first")
        df = df.set_index("Datetime")
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="1h")
        df = df.reindex(full_idx)
        df["AEP_MW"] = df["AEP_MW"].interpolate(method="linear")
        df = df[df["AEP_MW"] > 0]
        return df["AEP_MW"].values[:n_rows].astype(float)

    # Fallback synthetic periodic series if CSV is unavailable
    t = np.arange(n_rows)
    return (
        500.0
        + 0.05 * t
        + 80.0 * np.sin(2 * np.pi * t / 24.0 - 1.5)
        + 40.0 * np.sin(2 * np.pi * t / (24.0 * 7.0))
        + np.random.normal(0, 8, n_rows)
    ).astype(float)


def make_lags(series: np.ndarray, lags: List[int] = LAGS) -> Tuple[np.ndarray, np.ndarray]:
    """Create lag features: X[t] = [d(t-1), d(t-2), d(t-3), d(t-24)] for each t >= max(lags), y[t] = d(t)."""
    max_lag = max(lags)
    X = np.array([[series[t - lag] for lag in lags] for t in range(max_lag, len(series))], dtype=float)
    y = series[max_lag:].astype(float)
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
