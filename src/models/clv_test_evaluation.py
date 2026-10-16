from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Project root
BASE_DIR = Path(__file__).resolve().parents[2]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.model_logger import save_model_results


MODEL_DIR = BASE_DIR / "models_artifacts"
DATA_DIR = BASE_DIR / "data" / "processed" / "clv_regression"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "model_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X_TEST_FILE = DATA_DIR / "X_test.csv"
X_TEST_SCALED_FILE = DATA_DIR / "X_test_scaled.csv"
Y_TEST_FILE = DATA_DIR / "y_test.csv"


MODELS = {
    "CLV Random Forest": "clv_random_forest.pkl",
    "CLV Ridge": "clv_ridge.pkl",
    "CLV XGBoost": "clv_xgboost.pkl",
}


def load_test_data():
    """Load held-out CLV test data."""

    X_test = pd.read_csv(X_TEST_FILE)
    X_test_scaled = pd.read_csv(X_TEST_SCALED_FILE)
    y_test = pd.read_csv(Y_TEST_FILE).squeeze()

    return X_test, X_test_scaled, y_test


def evaluate_standard_model(
    name,
    model_file,
    X_test,
    X_test_scaled,
    y_test
):
    """Evaluate a saved CLV regression model."""

    model_path = MODEL_DIR / model_file

    print(f"\nEvaluating {name}...")

    model = joblib.load(model_path)

    # Ridge was trained on scaled features.
    if name == "CLV Ridge":
        X_eval = X_test_scaled
    else:
        X_eval = X_test

    # Match saved model feature order when available.
    if hasattr(model, "feature_names_in_"):
        X_eval = X_eval[model.feature_names_in_]

    start_time = time.time()

    predictions = model.predict(X_eval)

    evaluation_time = time.time() - start_time

    predictions = np.asarray(
        predictions,
        dtype=np.float64
    )

    # All standard CLV models predict log-transformed CLV.
    predictions = np.clip(
        predictions,
        -20,
        20
    )

    predictions = np.expm1(predictions)

    # Convert actual target from log1p back to original CLV scale.
    actual = np.expm1(y_test)

    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions
        )
    )

    r2 = r2_score(
        actual,
        predictions
    )

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    # Save final held-out result to master model results.
    # Do not save evaluation_time as training_time.
    save_model_results(
        model_name=name.replace("CLV ", ""),
        task="CLV Regression",
        mae=mae,
        rmse=rmse,
        r2=r2
    )

    return {
        "Model": name,
        "Dataset": "Test",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Evaluation Time": evaluation_time
    }


def evaluate_hurdle(X_test, y_test):
    """Evaluate the saved two-stage hurdle CLV model."""

    model_path = MODEL_DIR / "clv_hurdle.pkl"

    print("\nEvaluating Hurdle CLV...")

    model_artifact = joblib.load(model_path)

    classifier = model_artifact["classifier"]
    regressor = model_artifact["regressor"]

    # Load the bias correction saved during Hurdle training.
    bias_correction = model_artifact.get(
        "bias_correction",
        1.0
    )

    print(
        f"Bias Correction : "
        f"{bias_correction:.4f}"
    )

    start_time = time.time()

    # Stage 1:
    # Probability that the customer will have future CLV > 0.
    positive_probability = (
        classifier.predict_proba(X_test)[:, 1]
    )

    # Stage 2:
    # Predict positive CLV on log1p scale.
    positive_clv_log = regressor.predict(X_test)

    positive_clv_log = np.asarray(
        positive_clv_log,
        dtype=np.float64
    )

    positive_clv_log = np.clip(
        positive_clv_log,
        -20,
        20
    )

    # Convert log1p prediction back to original CLV scale.
    positive_clv = np.expm1(
        positive_clv_log
    )

    # Apply the same bias correction used during Hurdle training.
    positive_clv = (
        positive_clv *
        bias_correction
    )

    # Final expected CLV:
    # Probability of positive CLV × predicted positive CLV.
    predictions = (
        positive_probability *
        positive_clv
    )

    evaluation_time = time.time() - start_time

    # Convert actual target from log1p back to original CLV scale.
    actual = np.expm1(y_test)

    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions
        )
    )

    r2 = r2_score(
        actual,
        predictions
    )

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    # Hurdle CLV is already logged by clv_hurdle.py.
    # Do not log it again here.

    return {
        "Model": "Hurdle CLV",
        "Dataset": "Test",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Evaluation Time": evaluation_time
    }


def main():
    """Run final held-out CLV test evaluation."""

    print("\nLoading CLV Test Data...")
    print("-------------------------------------")

    X_test, X_test_scaled, y_test = load_test_data()

    print(f"X_test Shape        : {X_test.shape}")
    print(f"X_test Scaled Shape : {X_test_scaled.shape}")
    print(f"y_test Shape        : {y_test.shape}")

    results = []

    # Evaluate standard CLV models.
    for name, model_file in MODELS.items():

        result = evaluate_standard_model(
            name,
            model_file,
            X_test,
            X_test_scaled,
            y_test
        )

        results.append(result)

    # Evaluate Hurdle CLV.
    hurdle_result = evaluate_hurdle(
        X_test,
        y_test
    )

    results.append(hurdle_result)

    # Create final comparison dataframe.
    results_df = pd.DataFrame(
        results
    )

    output_file = (
        OUTPUT_DIR /
        "final_clv_test_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\nFinal CLV Test Results")
    print("-------------------------------------")

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nResults Saved : "
        f"{output_file}"
    )

    print(
        "\nFinal CLV Test Evaluation "
        "Completed Successfully!"
    )

if __name__ == "__main__":
    main()