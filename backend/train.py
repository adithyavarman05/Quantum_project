"""Training module for VQC regressor and Linear Regression baseline."""
import time
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backend.config import (
    ARTIFACTS_DIR,
    ARTIFACTS_PATH,
    EPOCHS,
    LR,
    N_LAYERS,
    N_QUBITS,
    SEED,
)
from backend.data import load_series, make_lags, split_and_scale
from backend.quantum import mse_loss, predict


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """Calculate MAE, RMSE, and R2 evaluation metrics in physical units (MW)."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return mae, rmse, r2


def train_and_evaluate() -> Dict[str, Any]:
    """Train VQC and Linear Regression models, evaluate on test split, and save artifacts."""
    np.random.seed(SEED)

    # 1. Prepare data
    series = load_series()
    X, y = make_lags(series)
    x_scaler, y_scaler, X_tr_s, y_tr_s, X_te_s, y_te = split_and_scale(X, y)

    # 2. Train Variational Quantum Circuit (VQC)
    shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)
    n_params_vqc = int(np.prod(shape))
    weights = pnp.array(np.random.uniform(0, 2 * np.pi, shape), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=LR)
    Xtr_p = pnp.array(X_tr_s, requires_grad=False)
    ytr_p = pnp.array(y_tr_s, requires_grad=False)

    losses: list[float] = []
    t_vqc_start = time.time()
    for ep in range(1, EPOCHS + 1):
        weights, loss_val = opt.step_and_cost(lambda w: mse_loss(w, Xtr_p, ytr_p), weights)
        losses.append(float(loss_val))
    vqc_train_time = time.time() - t_vqc_start

    # 3. Train Classical Linear Regression Baseline
    t_lr_start = time.time()
    lr = LinearRegression().fit(X_tr_s, y_tr_s)
    lr_train_time = time.time() - t_lr_start
    n_params_lr = int(X_tr_s.shape[1] + 1)

    # 4. Predict and evaluate in MW
    inv = lambda arr: y_scaler.inverse_transform(np.asarray(arr).reshape(-1, 1)).ravel()
    p_vqc = inv(predict(pnp.array(X_te_s, requires_grad=False), weights))
    p_lr = inv(lr.predict(X_te_s))

    vqc_mae, vqc_rmse, vqc_r2 = compute_metrics(y_te, p_vqc)
    lr_mae, lr_rmse, lr_r2 = compute_metrics(y_te, p_lr)

    # 5. Format metrics summary
    metrics = {
        "vqc": {
            "mae": vqc_mae,
            "rmse": vqc_rmse,
            "r2": vqc_r2,
            "train_time_s": float(vqc_train_time),
            "params": n_params_vqc,
        },
        "linear": {
            "mae": lr_mae,
            "rmse": lr_rmse,
            "r2": lr_r2,
            "train_time_s": float(lr_train_time),
            "params": n_params_lr,
        },
    }

    # 6. Save artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "weights": np.array(weights),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "linear_model": lr,
        "metrics": metrics,
        "loss_curve": losses,
        "actual": [float(val) for val in y_te],
        "vqc_pred": [float(val) for val in p_vqc],
        "linear_pred": [float(val) for val in p_lr],
        "last_values": [float(val) for val in series[-N_QUBITS:]],
    }
    joblib.dump(artifacts, ARTIFACTS_PATH)

    # Print summary table
    print("=" * 62)
    print(f"{'Metric':<18}{'VQC (Quantum)':>20}{'Linear Regression':>22}")
    print("-" * 62)
    print(f"{'MAE (MW)':<18}{vqc_mae:>20.4f}{lr_mae:>22.4f}")
    print(f"{'RMSE (MW)':<18}{vqc_rmse:>20.4f}{lr_rmse:>22.4f}")
    print(f"{'R2':<18}{vqc_r2:>20.4f}{lr_r2:>22.4f}")
    print(f"{'Train time (s)':<18}{vqc_train_time:>20.2f}{lr_train_time:>22.4f}")
    print(f"{'Parameters':<18}{n_params_vqc:>20d}{n_params_lr:>22d}")
    print("=" * 62)

    return artifacts


if __name__ == "__main__":
    train_and_evaluate()
