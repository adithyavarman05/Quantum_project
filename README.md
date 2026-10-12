# Quantum AI-based Electricity Demand Forecasting

**Student:** Adithya Varman Nandhivarman  
**Capstone Project** — Hybrid Variational Quantum Circuit (VQC) regressor for next-hour electricity demand prediction.

---

## Problem

Predict the next hour's regional grid electricity demand (MW) from the previous 4 consecutive hourly load values, using a parameterised quantum circuit compared against a classical linear regression baseline.

## Method

| Component | Detail |
|:--|:--|
| **Framework** | PennyLane `default.qubit` simulator |
| **Qubits** | 4 (one per lag feature) |
| **Encoding** | `AngleEmbedding` with RX rotations (angle = π × x) |
| **Ansatz** | `StronglyEntanglingLayers` (2 layers, 24 trainable parameters) |
| **Measurement** | `expval(PauliZ(0))` ∈ [-1, 1] |
| **Optimiser** | Adam (lr = 0.05), 80 epochs, MSE loss |
| **Differentiation** | `diff_method="backprop"` with parameter broadcasting |
| **Baseline** | scikit-learn `LinearRegression` (5 parameters) |
| **Data** | PJM AEP hourly demand (`data/AEP_hourly.csv`), first 1,500 rows |
| **Split** | 80/20 chronological (1,196 train / 300 test) |
| **Scaling** | Features [0, 1], target [-1, 1], scalers fit on train only |

### Complexity

| | Expression |
|:--|:--|
| **Time** | O(E · n · L · N · 2ⁿ) |
| **Space** | O(2ⁿ + N · n + n · L) |

Where E = epochs, n = qubits, L = layers, N = training samples.

## Project Structure

```
capstone/
├── backend/
│   ├── __init__.py          # Package marker
│   ├── config.py            # Hyperparameters and paths
│   ├── data.py              # load_series(), make_lags(), split_and_scale()
│   ├── quantum.py           # PennyLane qnode, predict(), bloch_vectors()
│   ├── train.py             # Training pipeline (VQC + Linear), saves artifacts
│   ├── service.py           # Artifact loading, inference, results retrieval
│   ├── schemas.py           # Pydantic request/response models
│   └── main.py              # FastAPI application and routing
├── frontend/
│   ├── index.html           # Single-page application layout
│   ├── styles.css           # "Solar Flare Plasma" design system
│   ├── app.js               # API calls, Chart.js rendering, count-up animations
│   └── scene.js             # Three.js r160 3D plasma core and Bloch arrows
├── data/
│   └── AEP_hourly.csv       # PJM Appalachian Power hourly demand dataset
├── artifacts/
│   └── artifacts.joblib     # Trained model weights, scalers, metrics, predictions
├── tests/
│   └── test_api.py          # 6 pytest tests for all API endpoints
├── quantum_demand_forecasting.py  # Original standalone training script
├── requirements.txt         # Pinned Python dependencies
├── pytest.ini               # Pytest configuration
└── README.md                # This file
```

## Setup and Run

### Prerequisites

- Python 3.10+ with `venv`
- No Node.js required — frontend is served directly by FastAPI

### 1. Create virtual environment and install dependencies

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the models

```powershell
python -m backend.train
```

This trains the VQC (≈80 epochs) and Linear Regression, evaluates both on the test set, and saves all artifacts to `artifacts/artifacts.joblib`. A metrics summary table is printed to the console.

### 3. Start the server

```powershell
uvicorn backend.main:app --reload
```

### 4. Open the application

Navigate to **http://127.0.0.1:8000** in your browser.

### 5. Run tests

```powershell
pytest tests/ -v
```

## API Contract

| Method | Endpoint | Description |
|:--|:--|:--|
| `GET` | `/api/health` | `{"status":"ok","model_loaded":true}` |
| `GET` | `/api/sample` | `{"values":[4 floats]}` — latest 4 hourly demand values |
| `POST` | `/api/predict` | Body: `{"values":[4 floats]}` → `{"vqc_mw":float,"linear_mw":float,"bloch_encoded":[[x,y,z]×4],"bloch_final":[[x,y,z]×4]}` |
| `GET` | `/api/results` | Full training metrics, loss curve, and test predictions for both models |

### Validation

- `POST /api/predict` requires exactly 4 finite floats between 0 and 100,000 MW.
- Returns HTTP 422 on validation failure.
- Returns HTTP 503 if artifacts have not been generated (run training first).

## Design: Solar Flare Plasma

The frontend uses a warm, fiery aesthetic — the opposite of the typical navy/cyan quantum computing look.

| Token | Value | Use |
|:--|:--|:--|
| `--bg` | `#0E0508` | Near-black burgundy background |
| `--amber` | `#FFB703` | Primary accent |
| `--coral` | `#FF5D3A` | Gradient middle |
| `--magenta` | `#E01E8C` | Gradient end |
| `--lime` | `#C8FF2E` | Success highlights |

**Features:**
- Animated gradient blobs and film-grain overlay background
- Rotating conic-gradient card borders
- Three.js 3D plasma core with 4 orbiting qubit spheres and Bloch arrows
- Chart.js benchmark visualisations (test fit, loss curve, error comparison)
- Count-up number animations and scroll-reveal transitions
- Fully responsive at 375px, 768px, and 1440px
- `prefers-reduced-motion` support

## Acknowledgements

- **Dataset:** PJM Interconnection LLC — AEP hourly electricity demand
- **Quantum framework:** [PennyLane](https://pennylane.ai/) by Xanadu
- **3D rendering:** [Three.js](https://threejs.org/) r160
- **Charts:** [Chart.js](https://www.chartjs.org/) 4.4.1
