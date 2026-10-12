"""
Quantum AI-based Electricity Demand Forecasting
Hybrid Variational Quantum Circuit (VQC) regressor vs Linear Regression baseline.
Student: Adithya Varman Nandhivarman
Run:  python quantum_demand_forecasting.py
"""
import os, time, sys
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- Config ----------------
N_QUBITS = 4          # = number of lag features
N_LAYERS = 2
N_HOURS = 1500        # rows of data to use
EPOCHS = 80
LR = 0.05
SEED = 42
np.random.seed(SEED)

# Locate CSV — check common paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_CANDIDATES = [
    os.path.join(BASE_DIR, "archive", "AEP_hourly.csv"),
    os.path.join(BASE_DIR, "AEP_hourly.csv"),
]


# ---------------- 1. Data ----------------
def load_demand(n_hours=N_HOURS):
    for path in CSV_CANDIDATES:
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["Datetime"]).sort_values("Datetime")
            print(f"Loaded real dataset: {path}")
            return df["AEP_MW"].values[:n_hours].astype(float)
    print("CSV not found -> using synthetic hourly demand data")
    t = np.arange(n_hours)
    demand = (500 + 0.05 * t
              + 80 * np.sin(2 * np.pi * t / 24 - 1.5)      # daily cycle
              + 40 * np.sin(2 * np.pi * t / (24 * 7))       # weekly cycle
              + np.random.normal(0, 8, n_hours))             # noise
    return demand


def make_lags(series, n_lags=N_QUBITS):
    """Create sliding-window lag features: X[i] = series[i-n_lags:i], y[i] = series[i]."""
    X = np.array([series[i - n_lags:i] for i in range(n_lags, len(series))])
    y = series[n_lags:]
    return X, y


# ---------------- 2. Quantum circuit ----------------
dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="autograd", diff_method="backprop")
def vqc(x, weights):
    """4-qubit VQC: AngleEmbedding (RX, angle=pi*x) + StronglyEntanglingLayers, measure Z0."""
    qml.AngleEmbedding(np.pi * x, wires=range(N_QUBITS), rotation="X")   # encoding
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))          # ansatz
    return qml.expval(qml.PauliZ(0))                                      # measurement


def predict(X, weights):
    """Batch prediction via PennyLane parameter broadcasting."""
    return vqc(X, weights)


def mse_loss(weights, X, y):
    """Mean squared error loss for the VQC."""
    return pnp.mean((predict(X, weights) - y) ** 2)


# ---------------- 3. Main ----------------
def main():
    print("=" * 60)
    print("Quantum AI Electricity Demand Forecasting")
    print("=" * 60)

    # --- Data ---
    series = load_demand()
    X, y = make_lags(series)
    print(f"Dataset: {len(series)} hours -> {len(X)} samples (4 lag features each)")

    split = int(0.8 * len(X))                     # chronological split
    X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]
    print(f"Train: {len(X_tr)}, Test: {len(X_te)} (80/20 chronological split)")

    # --- Scale ---
    x_scaler = MinMaxScaler((0, 1)).fit(X_tr)
    y_scaler = MinMaxScaler((-1, 1)).fit(y_tr.reshape(-1, 1))
    Xtr_s = x_scaler.transform(X_tr)
    Xte_s = np.clip(x_scaler.transform(X_te), 0, 1)
    ytr_s = y_scaler.transform(y_tr.reshape(-1, 1)).ravel()

    # ---- Train VQC ----
    print("\n--- Training Variational Quantum Circuit ---")
    shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)
    n_params = int(np.prod(shape))
    print(f"Circuit: {N_QUBITS} qubits, {N_LAYERS} layers, {n_params} trainable parameters")
    print(f"Optimizer: Adam (lr={LR}), Epochs: {EPOCHS}")

    weights = pnp.array(np.random.uniform(0, 2 * np.pi, shape), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=LR)
    Xtr_p = pnp.array(Xtr_s, requires_grad=False)
    ytr_p = pnp.array(ytr_s, requires_grad=False)

    losses = []
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        weights, loss = opt.step_and_cost(lambda w: mse_loss(w, Xtr_p, ytr_p), weights)
        losses.append(float(loss))
        if ep % 10 == 0 or ep == 1:
            print(f"  Epoch {ep:3d}/{EPOCHS}  loss = {loss:.6f}")
    vqc_time = time.time() - t0
    print(f"VQC training completed in {vqc_time:.2f}s")

    # ---- Baseline: Linear Regression ----
    print("\n--- Training Linear Regression Baseline ---")
    t0 = time.time()
    lr = LinearRegression().fit(Xtr_s, ytr_s)
    lr_time = time.time() - t0
    lr_params = X_tr.shape[1] + 1  # coefficients + intercept
    print(f"Linear Regression trained in {lr_time:.4f}s ({lr_params} parameters)")

    # ---- Evaluate (back to MW) ----
    inv = lambda a: y_scaler.inverse_transform(np.asarray(a).reshape(-1, 1)).ravel()
    p_vqc = inv(predict(pnp.array(Xte_s, requires_grad=False), weights))
    p_lr = inv(lr.predict(Xte_s))

    def metrics(p):
        return (mean_absolute_error(y_te, p),
                np.sqrt(mean_squared_error(y_te, p)),
                r2_score(y_te, p))

    m_v, m_l = metrics(p_vqc), metrics(p_lr)

    print("\n" + "=" * 62)
    print(f"{'Metric':<18}{'VQC (Quantum)':>20}{'Linear Regression':>22}")
    print("-" * 62)
    for name, a, b in zip(["MAE (MW)", "RMSE (MW)", "R²"], m_v, m_l):
        print(f"{name:<18}{a:>20.4f}{b:>22.4f}")
    print(f"{'Train time (s)':<18}{vqc_time:>20.2f}{lr_time:>22.4f}")
    print(f"{'Parameters':<18}{n_params:>20d}{lr_params:>22d}")
    print("=" * 62)

    # ---- Plots ----
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: Actual vs Predicted (VQC)
    k = min(150, len(y_te))
    ax[0].plot(y_te[:k], label="Actual", linewidth=1.2, color="#2196F3")
    ax[0].plot(p_vqc[:k], "--", label="VQC Predicted", linewidth=1.2, color="#FF5722")
    ax[0].set_title("Actual vs Predicted (VQC)", fontsize=12, fontweight="bold")
    ax[0].set_xlabel("Hour"); ax[0].set_ylabel("Demand (MW)"); ax[0].legend()
    ax[0].grid(alpha=0.3)

    # Subplot 2: Loss curve
    ax[1].plot(losses, color="#4CAF50", linewidth=1.2)
    ax[1].set_title("VQC Training Loss (MSE)", fontsize=12, fontweight="bold")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("MSE Loss")
    ax[1].grid(alpha=0.3)

    # Subplot 3: Model comparison
    ax[2].plot(y_te[:k], label="Actual", linewidth=1.2, color="#2196F3")
    ax[2].plot(p_vqc[:k], "--", label="VQC", linewidth=1.2, color="#FF5722")
    ax[2].plot(p_lr[:k], ":", label="Linear Regression", linewidth=1.2, color="#9C27B0")
    ax[2].set_title("Model Comparison", fontsize=12, fontweight="bold")
    ax[2].set_xlabel("Hour"); ax[2].set_ylabel("Demand (MW)"); ax[2].legend()
    ax[2].grid(alpha=0.3)

    plt.tight_layout()
    results_path = os.path.join(BASE_DIR, "results_plots.png")
    fig.savefig(results_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {results_path}")

    # ---- Circuit diagram ----
    circuit_path = os.path.join(BASE_DIR, "quantum_circuit.png")
    fig_c, _ = qml.draw_mpl(vqc)(np.zeros(N_QUBITS), np.array(weights))
    fig_c.savefig(circuit_path, dpi=150, bbox_inches="tight")
    plt.close(fig_c)
    print(f"Saved: {circuit_path}")

    # ---- Save artifacts for Streamlit & report generation ----
    artifacts = {
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "weights": np.array(weights),
        "losses": losses,
        "metrics_vqc": m_v,
        "metrics_lr": m_l,
        "vqc_time": vqc_time,
        "lr_time": lr_time,
        "n_params_vqc": n_params,
        "n_params_lr": lr_params,
        "last_values": series[-N_QUBITS:],
        "lr_model": lr,
        "y_te": y_te,
        "p_vqc": p_vqc,
        "p_lr": p_lr,
    }
    artifacts_path = os.path.join(BASE_DIR, "artifacts.joblib")
    joblib.dump(artifacts, artifacts_path)
    print(f"Saved: {artifacts_path}")
    print("\nAll done! Run `streamlit run streamlit_app.py` to launch the web app.")

    return artifacts


if __name__ == "__main__":
    main()
