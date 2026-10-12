"""Configuration constants and paths for Quantum Electricity Demand Forecasting."""
from pathlib import Path
from typing import List

# Quantum circuit and training hyperparameters
LAGS: List[int] = [1, 2, 3, 24]
N_QUBITS: int = len(LAGS)
N_LAYERS: int = 3
N_ROWS: int = 2000
N_HOURS: int = N_ROWS  # alias for backwards compatibility
EPOCHS: int = 120
LR: float = 0.05
SEED: int = 42

# Project paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = BASE_DIR / "data" / "AEP_hourly.csv"
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
ARTIFACTS_PATH: Path = ARTIFACTS_DIR / "artifacts.joblib"
