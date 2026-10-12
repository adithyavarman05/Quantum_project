"""Pydantic schemas and validation for API request and response models."""
import math
from typing import List
from pydantic import BaseModel, field_validator


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class SampleResponse(BaseModel):
    values: List[float]
    labels: List[str] = ["t-1h", "t-2h", "t-3h", "t-24h"]


class PredictRequest(BaseModel):
    values: List[float]

    @field_validator("values")
    @classmethod
    def validate_values(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError("Values must contain exactly 4 numbers.")
        for val in v:
            if not math.isfinite(val):
                raise ValueError("Values must be finite.")
            if val < 0.0 or val > 100000.0:
                raise ValueError("Values must be between 0 and 100000.")
        return v


class PredictResponse(BaseModel):
    vqc_mw: float
    linear_mw: float
    bloch_encoded: List[List[float]]
    bloch_final: List[List[float]]


class ModelMetrics(BaseModel):
    mae: float
    rmse: float
    r2: float
    train_time_s: float
    params: int


class MetricsSummary(BaseModel):
    vqc: ModelMetrics
    linear: ModelMetrics


class ResultsResponse(BaseModel):
    metrics: MetricsSummary
    loss_curve: List[float]
    actual: List[float]
    vqc_pred: List[float]
    linear_pred: List[float]
