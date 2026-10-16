"""Calibrate saved churn scores on the validation split and report held-out metrics."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "data/processed/model_data"
MODEL_PATH = MODEL_DIR / "random_forest.pkl"
OUTPUT_PATH = ROOT / "models_artifacts/churn_probability_calibration.json"
METRICS_PATH = ROOT / "data/processed/model_results/churn_calibration_metrics.csv"


def _logit(probabilities: np.ndarray) -> np.ndarray:
    """Convert probabilities to finite log odds for logistic calibration."""
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def main() -> None:
    """Fit on validation rows and evaluate once on the saved test split."""
    model = joblib.load(MODEL_PATH)
    names = list(model.feature_names_in_)
    X_val = pd.read_csv(MODEL_DIR / "X_val.csv")[names]
    y_val = pd.read_csv(MODEL_DIR / "y_val.csv").squeeze()
    X_test = pd.read_csv(MODEL_DIR / "X_test.csv")[names]
    y_test = pd.read_csv(MODEL_DIR / "y_test.csv").squeeze()

    validation_scores = model.predict_proba(X_val)[:, 1]
    calibrator = LogisticRegression(random_state=42, max_iter=1000)
    calibrator.fit(_logit(validation_scores), y_val)

    raw_test = model.predict_proba(X_test)[:, 1]
    calibrated_test = calibrator.predict_proba(_logit(raw_test))[:, 1]
    raw_brier = brier_score_loss(y_test, raw_test)
    calibrated_brier = brier_score_loss(y_test, calibrated_test)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "method": "platt_scaling",
                "input": "random_forest.pkl probability",
                "validation_rows": len(y_val),
                "slope": float(calibrator.coef_[0, 0]),
                "intercept": float(calibrator.intercept_[0]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    metrics = pd.DataFrame(
        [
            {"split": "test", "probability": "raw", "roc_auc": roc_auc_score(y_test, raw_test), "brier_score": raw_brier, "log_loss": log_loss(y_test, raw_test)},
            {"split": "test", "probability": "calibrated", "roc_auc": roc_auc_score(y_test, calibrated_test), "brier_score": calibrated_brier, "log_loss": log_loss(y_test, calibrated_test)},
        ]
    )
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    print(metrics.to_string(index=False))
    print(f"Saved calibration parameters to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
