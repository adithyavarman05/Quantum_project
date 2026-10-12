"""Model inference and data service loading artifacts once at startup."""
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
from pennylane import numpy as pnp
from backend.config import ARTIFACTS_PATH
from backend.quantum import bloch_vectors, predict as vqc_predict

_artifacts: Optional[Dict[str, Any]] = None


def load_artifacts() -> bool:
    """Load model artifacts from disk if available."""
    global _artifacts
    if ARTIFACTS_PATH.exists():
        _artifacts = joblib.load(ARTIFACTS_PATH)
        return True
    return False


# Attempt loading on import
load_artifacts()


def is_model_loaded() -> bool:
    """Check if model artifacts are loaded."""
    return _artifacts is not None


def get_sample() -> Dict[str, Any]:
    """Return the last 4 known hourly demand values and labels from the series."""
    if _artifacts is None:
        raise RuntimeError("Artifacts not loaded.")
    values = [float(x) for x in _artifacts.get("last_values", [])]
    labels = _artifacts.get("labels", ["t-1h", "t-2h", "t-3h", "t-24h"])
    return {"values": values, "labels": labels}


def predict(values: List[float]) -> Dict[str, Any]:
    """Scale inputs, execute VQC and Linear models, generate Bloch vectors, inverse transform."""
    if _artifacts is None:
        raise RuntimeError("Artifacts not loaded.")

    x_scaler = _artifacts["x_scaler"]
    y_scaler = _artifacts["y_scaler"]
    weights = _artifacts["weights"]
    linear_model = _artifacts["linear_model"]

    # Reshape and scale input features
    raw_x = np.array(values, dtype=float).reshape(1, -1)
    scaled_x = np.clip(x_scaler.transform(raw_x), 0.0, 1.0)

    # 1. Classical linear regression prediction
    lin_scaled = linear_model.predict(scaled_x)
    linear_mw = float(y_scaler.inverse_transform(np.asarray(lin_scaled).reshape(-1, 1))[0, 0])

    # 2. VQC quantum prediction
    vqc_scaled = vqc_predict(pnp.array(scaled_x, requires_grad=False), weights)
    vqc_mw = float(y_scaler.inverse_transform(np.asarray(vqc_scaled).reshape(-1, 1))[0, 0])

    # 3. Bloch sphere coordinates
    bloch_enc, bloch_fin = bloch_vectors(scaled_x[0], weights)

    return {
        "vqc_mw": vqc_mw,
        "linear_mw": linear_mw,
        "bloch_encoded": bloch_enc,
        "bloch_final": bloch_fin,
    }


def get_results() -> Dict[str, Any]:
    """Retrieve full training and evaluation metrics and comparison series."""
    if _artifacts is None:
        raise RuntimeError("Artifacts not loaded.")

    return {
        "metrics": _artifacts["metrics"],
        "loss_curve": _artifacts["loss_curve"],
        "actual": _artifacts["actual"],
        "vqc_pred": _artifacts["vqc_pred"],
        "linear_pred": _artifacts["linear_pred"],
    }
