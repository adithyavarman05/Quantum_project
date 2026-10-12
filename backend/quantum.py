"""PennyLane quantum circuit definitions and execution functions."""
from typing import List, Tuple
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from backend.config import N_QUBITS, N_LAYERS

# Default simulator device
dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="autograd", diff_method="backprop")
def vqc(x, weights):
    """4-qubit VQC: AngleEmbedding (RX, angle=pi*x) + StronglyEntanglingLayers, measure Z0."""
    qml.AngleEmbedding(np.pi * x, wires=range(N_QUBITS), rotation="X")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))


def predict(X, weights):
    """Batch prediction via PennyLane parameter broadcasting."""
    return vqc(X, weights)


def mse_loss(weights, X, y):
    """Mean squared error loss for training the VQC."""
    return pnp.mean((predict(X, weights) - y) ** 2)


@qml.qnode(dev)
def _circuit_encoded(x):
    """Expectation values after AngleEmbedding encoding only."""
    qml.AngleEmbedding(np.pi * x, wires=range(N_QUBITS), rotation="X")
    return [
        qml.expval(op(w))
        for w in range(N_QUBITS)
        for op in (qml.PauliX, qml.PauliY, qml.PauliZ)
    ]


@qml.qnode(dev)
def _circuit_final(x, weights):
    """Expectation values after encoding and StronglyEntanglingLayers."""
    qml.AngleEmbedding(np.pi * x, wires=range(N_QUBITS), rotation="X")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [
        qml.expval(op(w))
        for w in range(N_QUBITS)
        for op in (qml.PauliX, qml.PauliY, qml.PauliZ)
    ]


def bloch_vectors(x: np.ndarray, weights: np.ndarray) -> Tuple[List[List[float]], List[List[float]]]:
    """Return per-qubit <X>, <Y>, <Z> Bloch vector components after encoding and after full circuit."""
    raw_enc = _circuit_encoded(x)
    raw_fin = _circuit_final(x, weights)

    bloch_encoded = [
        [float(raw_enc[w * 3]), float(raw_enc[w * 3 + 1]), float(raw_enc[w * 3 + 2])]
        for w in range(N_QUBITS)
    ]
    bloch_final = [
        [float(raw_fin[w * 3]), float(raw_fin[w * 3 + 1]), float(raw_fin[w * 3 + 2])]
        for w in range(N_QUBITS)
    ]
    return bloch_encoded, bloch_final
