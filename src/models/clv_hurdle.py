import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.utils.model_logger import save_model_results

TRAIN_DIR = BASE_DIR / "data" / "processed" / "clv_regression"
OUTPUT_PATH = BASE_DIR / "models_artifacts" / "clv_hurdle.pkl"


def load_data():
    """Load preprocessed CLV train, validation and test data."""

    print("\nLoading CLV Regression Data...")

    X_train = pd.read_csv(TRAIN_DIR / "X_train.csv")
    X_val = pd.read_csv(TRAIN_DIR / "X_val.csv")
    X_test = pd.read_csv(TRAIN_DIR / "X_test.csv")

    y_train = pd.read_csv(TRAIN_DIR / "y_train.csv").squeeze()
    y_val = pd.read_csv(TRAIN_DIR / "y_val.csv").squeeze()
    y_test = pd.read_csv(TRAIN_DIR / "y_test.csv").squeeze()

    return X_train, X_val, X_test, y_train, y_val, y_test


def calculate_bias_correction(regressor, X_positive, y_positive):
    """Calculate log-normal bias correction using positive training data only.

    The regressor predicts log1p(CLV). Directly applying expm1()
    estimates a transformed prediction rather than the original-scale
    conditional mean. The residual variance correction helps account
    for this when converting predictions back to the original scale."""

    predicted_log = regressor.predict(X_positive)

    residuals = y_positive.to_numpy(dtype=np.float64) - predicted_log

    residual_variance = np.var(residuals, ddof=1)

    bias_correction = np.exp(0.5 * residual_variance)

    # Prevent an extreme correction factor from destabilizing predictions.
    bias_correction = np.clip(bias_correction, 1.0, 3.0)

    return bias_correction


def train_models(X_train, y_train, classifier_params, regressor_params):
    """Train the two-stage hurdle CLV model."""

    y_train_original = np.expm1(y_train)

    # Stage 1: Probability of positive future CLV
    y_train_positive = (y_train_original > 0).astype(int)

    classifier = RandomForestClassifier(
        n_estimators=classifier_params["n_estimators"],
        max_depth=classifier_params["max_depth"],
        min_samples_leaf=classifier_params["min_samples_leaf"],
        random_state=42,
        n_jobs=-1
    )

    classifier.fit(X_train, y_train_positive)

    # Stage 2: Positive CLV prediction
    positive_mask = y_train_original > 0

    X_positive = X_train.loc[positive_mask]
    y_positive = y_train.loc[positive_mask]

    regressor = RandomForestRegressor(
        n_estimators=regressor_params["n_estimators"],
        max_depth=regressor_params["max_depth"],
        min_samples_leaf=regressor_params["min_samples_leaf"],
        random_state=42,
        n_jobs=-1
    )

    regressor.fit(X_positive, y_positive)

    # Bias correction calculated from training positives only
    bias_correction = calculate_bias_correction(
        regressor,
        X_positive,
        y_positive
    )

    return classifier, regressor, bias_correction


def predict_clv(classifier, regressor, bias_correction, X):
    """Generate final original-scale CLV predictions."""

    # Probability that future CLV is positive.
    positive_probability = classifier.predict_proba(X)[:, 1]

    # Positive CLV prediction on log1p scale.
    positive_clv_log = regressor.predict(X)

    positive_clv_log = np.asarray(
        positive_clv_log,
        dtype=np.float64
    )

    positive_clv_log = np.clip(
        positive_clv_log,
        -20,
        20
    )

    # Convert back to original scale with bias correction.
    positive_clv = (
        np.expm1(positive_clv_log)
        * bias_correction
    )

    predictions = positive_probability * positive_clv

    return predictions


def evaluate_model(
    classifier,
    regressor,
    bias_correction,
    X,
    y
):
    """Evaluate final CLV predictions on original scale."""

    actual = np.expm1(y)

    predictions = predict_clv(
        classifier,
        regressor,
        bias_correction,
        X
    )

    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    r2 = r2_score(actual, predictions)

    return mae, rmse, r2


def tune_hurdle_model(X_train, y_train, X_val, y_val):
    """Tune classifier and regressor using validation R²."""

    print("\nTuning Hurdle CLV Model...")

    classifier_grid = [
        {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 1
        },
        {
            "n_estimators": 300,
            "max_depth": 10,
            "min_samples_leaf": 1
        },
        {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_leaf": 2
        },
        {
            "n_estimators": 300,
            "max_depth": 15,
            "min_samples_leaf": 2
        }
    ]

    regressor_grid = [
        {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 1
        },
        {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_leaf": 2
        },
        {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_leaf": 2
        },
        {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_leaf": 2
        }
    ]

    best_r2 = -np.inf

    best_classifier_params = None
    best_regressor_params = None

    print("\nTesting classifier and regressor combinations...")
    print("-------------------------------------")

    for classifier_params in classifier_grid:

        for regressor_params in regressor_grid:

            classifier, regressor, bias_correction = train_models(
                X_train,
                y_train,
                classifier_params,
                regressor_params
            )

            mae, rmse, r2 = evaluate_model(
                classifier,
                regressor,
                bias_correction,
                X_val,
                y_val
            )

            print(
                f"Classifier: {classifier_params} | "
                f"Regressor: {regressor_params} | "
                f"Validation R²: {r2:.4f}"
            )

            if r2 > best_r2:

                best_r2 = r2
                best_classifier_params = classifier_params
                best_regressor_params = regressor_params

    print("\nBest Validation Configuration")
    print("-------------------------------------")

    print(
        f"Best Classifier Params : "
        f"{best_classifier_params}"
    )

    print(
        f"Best Regressor Params  : "
        f"{best_regressor_params}"
    )

    print(
        f"Best Validation R²     : "
        f"{best_r2:.4f}"
    )

    return best_classifier_params, best_regressor_params


def main():
    """Tune, train, evaluate and save the hurdle CLV model."""

    start_time = time.time()

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    print("\nDataset Split")
    print("-------------------------------------")

    print(f"Train : {X_train.shape}")
    print(f"Val   : {X_val.shape}")
    print(f"Test  : {X_test.shape}")

    print("\nTarget Scale : Log-transformed")

    best_classifier_params, best_regressor_params = tune_hurdle_model(
        X_train,
        y_train,
        X_val,
        y_val
    )


    # Final training on train + validation
    print("\nRetraining Final Hurdle Model...")
    print("-------------------------------------")

    X_train_final = pd.concat(
        [X_train, X_val],
        axis=0
    )

    y_train_final = pd.concat(
        [y_train, y_val],
        axis=0
    )

    print(
        f"Final Training Data : "
        f"{X_train_final.shape}"
    )

    classifier, regressor, bias_correction = train_models(
        X_train_final,
        y_train_final,
        best_classifier_params,
        best_regressor_params
    )


    # Final untouched test evaluation
    print("\nEvaluating Final Hurdle CLV Model...")

    mae, rmse, r2 = evaluate_model(
        classifier,
        regressor,
        bias_correction,
        X_test,
        y_test
    )

    print("\nFinal CLV Hurdle Test Results")
    print("-------------------------------------")

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    print(
        f"\nBias Correction : "
        f"{bias_correction:.4f}"
    )

    training_time = time.time() - start_time

    print(
        f"\nTotal Training Time : "
        f"{training_time:.2f} seconds"
    )

    # Save model artifact
    model_artifact = {
        "classifier": classifier,
        "regressor": regressor,
        "bias_correction": bias_correction,
        "target_transform": "log1p",
        "prediction_transform": "expm1_with_bias_correction",
        "classifier_params": best_classifier_params,
        "regressor_params": best_regressor_params
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model_artifact,
        OUTPUT_PATH
    )

    print(
        f"\nModel Saved to : "
        f"{OUTPUT_PATH}"
    )


    # Save model results
    save_model_results(
        model_name="Hurdle CLV",
        task="CLV Regression",
        training_time=training_time,
        mae=mae,
        rmse=rmse,
        r2=r2
    )

    print("\nCLV Hurdle Model Completed Successfully!")

if __name__ == "__main__":
    main()