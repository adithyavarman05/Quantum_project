"""Configuration constants and paths for Quantum Electricity Demand Forecasting."""
from pathlib import Path

# Quantum circuit and training hyperparameters
N_QUBITS: int = 4
N_LAYERS: int = 2
N_HOURS: int = 1500
EPOCHS: int = 80
LR: float = 0.05
SEED: int = 42

# Project paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = BASE_DIR / "data" / "AEP_hourly.csv"
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
ARTIFACTS_PATH: Path = ARTIFACTS_DIR / "artifacts.joblib"
