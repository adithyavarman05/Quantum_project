"""Tests for FastAPI endpoints."""
import math
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health():
    """Verify GET /api/health returns status ok and model_loaded true."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_results_shape():
    """Verify GET /api/results contains complete metrics, curves, and evaluation series."""
    response = client.get("/api/results")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "vqc" in data["metrics"]
    assert "linear" in data["metrics"]
    for key in ("mae", "rmse", "r2", "train_time_s", "params"):
        assert key in data["metrics"]["vqc"]
        assert key in data["metrics"]["linear"]
    assert isinstance(data["loss_curve"], list)
    assert len(data["loss_curve"]) > 0
    assert isinstance(data["actual"], list)
    assert isinstance(data["vqc_pred"], list)
    assert isinstance(data["linear_pred"], list)
    assert len(data["actual"]) == len(data["vqc_pred"]) == len(data["linear_pred"])


def test_sample():
    """Verify GET /api/sample returns 4 numbers and correct labels."""
    response = client.get("/api/sample")
    assert response.status_code == 200
    data = response.json()
    assert "values" in data
    assert len(data["values"]) == 4
    assert "labels" in data
    assert data["labels"] == ["t-1h", "t-2h", "t-3h", "t-24h"]


def test_valid_predict():
    """Verify POST /api/predict produces finite MW values and 4x3 Bloch coordinates."""
    response = client.post("/api/predict", json={"values": [12000.0, 12500.0, 13000.0, 12800.0]})
    assert response.status_code == 200
    data = response.json()
    assert math.isfinite(data["vqc_mw"])
    assert math.isfinite(data["linear_mw"])
    assert len(data["bloch_encoded"]) == 4
    assert len(data["bloch_final"]) == 4
    for b in data["bloch_encoded"]:
        assert len(b) == 3
        for coord in b:
            assert math.isfinite(coord)
    for b in data["bloch_final"]:
        assert len(b) == 3
        for coord in b:
            assert math.isfinite(coord)


def test_invalid_predict_length():
    """Verify POST /api/predict rejects requests with fewer than 4 values with 422."""
    response = client.post("/api/predict", json={"values": [12000.0, 12500.0, 13000.0]})
    assert response.status_code == 422


def test_invalid_predict_out_of_range():
    """Verify POST /api/predict rejects out-of-range values with 422."""
    response = client.post("/api/predict", json={"values": [-10.0, 12000.0, 13000.0, 12500.0]})
    assert response.status_code == 422
