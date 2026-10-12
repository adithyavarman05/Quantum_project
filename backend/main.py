"""FastAPI application entrypoint for Quantum Demand Forecasting."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from backend.config import BASE_DIR
from backend import service
from backend.schemas import (
    HealthResponse,
    SampleResponse,
    PredictRequest,
    PredictResponse,
    ResultsResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load_artifacts()
    yield


app = FastAPI(title="Quantum Electricity Demand Forecasting", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ensure_model_loaded():
    """Ensure artifacts are loaded before handling model requests."""
    if not service.is_model_loaded():
        if not service.load_artifacts():
            raise HTTPException(
                status_code=503,
                detail="Run: python -m backend.train",
            )


@app.get("/api/health", response_model=HealthResponse)
def get_health():
    """Health check endpoint indicating service and model readiness."""
    return HealthResponse(
        status="ok",
        model_loaded=service.is_model_loaded(),
    )


@app.get("/api/sample", response_model=SampleResponse)
def get_sample():
    """Return latest 4 hourly demand sample values."""
    ensure_model_loaded()
    return SampleResponse(values=service.get_sample())


@app.post("/api/predict", response_model=PredictResponse)
def post_predict(request: PredictRequest):
    """Run VQC and Linear model predictions on 4 lag features."""
    ensure_model_loaded()
    result = service.predict(request.values)
    return PredictResponse(**result)


@app.get("/api/results", response_model=ResultsResponse)
def get_results():
    """Return historical training curves, evaluation metrics, and comparison series."""
    ensure_model_loaded()
    res = service.get_results()
    return ResultsResponse(**res)


# Mount frontend directory for static assets
frontend_dir = BASE_DIR / "frontend"
frontend_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
