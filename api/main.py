"""FastAPI application for churn and lifetime-value scoring."""
from __future__ import annotations

import json
import logging
import math

import joblib
from fastapi import FastAPI

from api.database import initialize_database
from api.routers.customer import router as customer_router
from api.routers.prediction import router as prediction_router
from api.settings import BASE_DIR, CHURN_MODEL_PATH, CLV_MODEL_PATH, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOG = logging.getLogger(__name__)

if not CHURN_MODEL_PATH.is_absolute():
    CHURN_MODEL_PATH = BASE_DIR / CHURN_MODEL_PATH
if not CLV_MODEL_PATH.is_absolute():
    CLV_MODEL_PATH = BASE_DIR / CLV_MODEL_PATH
if not CHURN_MODEL_PATH.exists() or not CLV_MODEL_PATH.exists():
    raise FileNotFoundError(f"Model artifacts missing: churn={CHURN_MODEL_PATH.exists()} CLV={CLV_MODEL_PATH.exists()}")

churn_model = joblib.load(CHURN_MODEL_PATH)
clv_model = joblib.load(CLV_MODEL_PATH)
# Batch parallelism is counterproductive for one-row web requests and can spawn
# worker processes per call. Keep inference predictable for the scoring API.
for _inference_model in (churn_model, clv_model):
    if hasattr(_inference_model, "n_jobs"):
        _inference_model.n_jobs = 1
CALIBRATION_PATH = BASE_DIR / "models_artifacts/churn_probability_calibration.json"
calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8")) if CALIBRATION_PATH.exists() else None
app = FastAPI(title="Vantara Customer Intelligence API", description="Customer churn and CLV prediction API", version="1.1.0")
app.include_router(prediction_router)
app.include_router(customer_router)


@app.on_event("startup")
def startup() -> None:
    """Initialize storage and load precomputed customer segments."""
    initialize_database()


@app.get("/health")
def health_check() -> dict[str, str]:
    """Report API and model readiness."""
    return {"status": "healthy", "churn_model": "loaded", "clv_model": "loaded"}


def calibrated_churn_probability(probability: float) -> float:
    """Apply validation-fitted Platt scaling when calibration parameters exist."""
    if calibration is None:
        return probability
    clipped = min(max(probability, 1e-6), 1 - 1e-6)
    log_odds = math.log(clipped / (1 - clipped))
    scaled = calibration["slope"] * log_odds + calibration["intercept"]
    return 1 / (1 + math.exp(-max(min(scaled, 35), -35)))


@app.get("/metadata")
def metadata() -> dict[str, object]:
    """Return model metadata and the required inference columns."""
    return {
        "application": "Vantara Customer Intelligence Platform",
        "version": app.version,
        "next_purchase_horizon_days": 90,
        "churn_probability_calibrated": calibration is not None,
        "models": {"churn": {"name": type(churn_model).__name__, "features": len(churn_model.feature_names_in_)}, "clv": {"name": type(clv_model).__name__, "features": len(clv_model.feature_names_in_)}},
        "churn_features": list(churn_model.feature_names_in_),
        "clv_features": list(clv_model.feature_names_in_),
    }
