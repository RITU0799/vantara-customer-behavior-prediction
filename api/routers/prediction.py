"""Single, JSON batch, and CSV batch scoring endpoints."""
from __future__ import annotations

from io import BytesIO, StringIO
import logging
from typing import Any

import pandas as pd
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from api.database import store_prediction
from api.schemas.request import BatchPredictionRequest, PredictionRequest
from api.schemas.response import BatchPredictionResponse, PredictionResponse
from src.utils.config import load_config

router = APIRouter(prefix="/prediction", tags=["Prediction"])
LOG = logging.getLogger(__name__)


def score_features(features: dict[str, float], customer_id: str | None = None) -> dict[str, Any]:
    """Validate required inputs, score both models, and persist the result."""
    from api.main import calibrated_churn_probability, churn_model, clv_model

    churn_names = list(churn_model.feature_names_in_)
    clv_names = list(clv_model.feature_names_in_)
    missing_churn = [name for name in churn_names if name not in features]
    missing_clv = [name for name in clv_names if name not in features]
    if missing_churn or missing_clv:
        raise HTTPException(status_code=422, detail={"message": "Required model features are missing", "missing_churn_features": missing_churn, "missing_clv_features": missing_clv})
    import math
    if any(not math.isfinite(float(value)) for value in features.values()):
        raise HTTPException(status_code=422, detail="Feature values must be finite numbers")
    churn_input = pd.DataFrame([{name: features[name] for name in churn_names}], columns=churn_names)
    clv_input = pd.DataFrame([{name: features[name] for name in clv_names}], columns=clv_names)
    raw_probability = float(churn_model.predict_proba(churn_input)[0][1])
    probability = calibrated_churn_probability(raw_probability)
    threshold = float(load_config()["models"].get("churn_threshold", 0.5))
    result = {
        "customer_id": customer_id,
        "churn": {"prediction": int(probability >= threshold), "probability": round(probability, 4)},
        "next_purchase_probability": round(1 - probability, 4),
        "prediction_horizon_days": 90,
        "clv": {"predicted_value": round(float(np.expm1(clv_model.predict(clv_input)[0])), 2)},
    }
    store_prediction(customer_id, features, result["churn"], result["clv"])
    return result


@router.post("/predict", response_model=PredictionResponse)
def predict_customer(request: PredictionRequest) -> dict[str, Any]:
    """Return churn probability and CLV for a customer."""
    return score_features(request.features, request.customer_id)


@router.post("/explain")
def explain_customer(request: PredictionRequest, method: str = Query(default="shap", pattern="^(shap|lime)$")) -> dict[str, Any]:
    """Return local SHAP or LIME contributions for the churn prediction."""
    import numpy as np
    import shap
    from api.main import churn_model

    names = list(churn_model.feature_names_in_)
    missing = [name for name in names if name not in request.features]
    if missing:
        raise HTTPException(status_code=422, detail={"missing_churn_features": missing})
    row = pd.DataFrame([{name: request.features[name] for name in names}], columns=names)
    try:
        if method == "lime":
            from lime.lime_tabular import LimeTabularExplainer
            from api.main import BASE_DIR
            training_path = BASE_DIR / "data/processed/model_data/X_train.csv"
            if not training_path.exists():
                raise HTTPException(status_code=503, detail="LIME training reference data is not installed")
            training = pd.read_csv(training_path)[names]
            explainer = LimeTabularExplainer(training.values, feature_names=names, class_names=["Active", "Churn"], mode="classification", random_state=42)
            explanation = explainer.explain_instance(row.iloc[0].values, churn_model.predict_proba, num_features=10)
            return {"customer_id": request.customer_id, "method": "LIME", "contributions": [{"feature": feature, "impact": float(impact)} for feature, impact in explanation.as_list()]}
        values = shap.TreeExplainer(churn_model).shap_values(row)
        if isinstance(values, list):
            contributions = np.asarray(values[1])[0]
        else:
            values = np.asarray(values)
            contributions = values[0, :, 1] if values.ndim == 3 else values[0]
        order = np.argsort(np.abs(contributions))[::-1][:10]
        return {"customer_id": request.customer_id, "method": "SHAP", "contributions": [{"feature": names[int(i)], "value": float(row.iloc[0, int(i)]), "impact": float(contributions[int(i)])} for i in order]}
    except HTTPException:
        raise
    except Exception as exc:
        LOG.exception("Unable to calculate local SHAP values")
        raise HTTPException(status_code=500, detail=f"Could not compute local SHAP explanation: {exc}") from exc


@router.post("/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> dict[str, Any]:
    """Score a batch of feature dictionaries."""
    results = []
    for index, item in enumerate(request.customers):
        try:
            scored = score_features(item.features, item.customer_id)
        except HTTPException as exc:
            raise HTTPException(status_code=exc.status_code, detail={"customer_index": index, "detail": exc.detail}) from exc
        results.append({
            "customer_index": index,
            "customer_id": item.customer_id,
            "next_purchase_probability": scored["next_purchase_probability"],
            "prediction_horizon_days": scored["prediction_horizon_days"],
            "churn": scored["churn"],
            "clv": scored["clv"],
        })
    return {"count": len(results), "predictions": results}


@router.post("/batch-csv")
async def predict_csv(file: UploadFile = File(...)) -> StreamingResponse:
    """Score a CSV upload and return a downloadable CSV report."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file")
    try:
        source = pd.read_csv(BytesIO(await file.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}") from exc
    max_rows = int(load_config()["service"].get("batch_csv_max_rows", 10000))
    if source.empty or len(source) > max_rows:
        raise HTTPException(status_code=400, detail=f"CSV must contain 1 to {max_rows:,} customer rows")
    records = []
    customer_col = "Customer_ID" if "Customer_ID" in source.columns else "customer_id" if "customer_id" in source.columns else None
    for index, row in source.iterrows():
        identifier = str(row[customer_col]) if customer_col and pd.notna(row[customer_col]) else None
        features: dict[str, float] = {}
        for column, value in row.items():
            if column == customer_col:
                continue
            if pd.isna(value):
                raise HTTPException(status_code=422, detail=f"CSV row {index + 2}, column {column}: missing value")
            try:
                features[str(column)] = float(value)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=f"CSV row {index + 2}, column {column}: expected a number") from exc
        try:
            result = score_features(features, identifier)
        except HTTPException as exc:
            raise HTTPException(status_code=exc.status_code, detail={"csv_row": index + 2, "detail": exc.detail}) from exc
        records.append({
            "Customer_ID": identifier or "",
            "Churn_Prediction": result["churn"]["prediction"],
            "Churn_Probability": result["churn"]["probability"],
            "90_Day_Purchase_Probability": result["next_purchase_probability"],
            "Predicted_CLV": result["clv"]["predicted_value"],
        })
    output = StringIO()
    pd.DataFrame(records).to_csv(output, index=False)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=customer_scores.csv"})


