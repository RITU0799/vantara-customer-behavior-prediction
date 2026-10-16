from pathlib import Path
import time

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)


# Paths
BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "data" / "processed" / "model_data"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "model_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X_TEST_FILE = MODEL_DIR / "X_test.csv"
X_TEST_SCALED_FILE = MODEL_DIR / "X_test_scaled.csv"
Y_TEST_FILE = MODEL_DIR / "y_test.csv"

MODELS = {
    "Logistic Regression": ("logistic_regression.pkl", "scaled"),
    "Decision Tree": ("decision_tree.pkl", "raw"),
    "Random Forest": ("random_forest.pkl", "raw"),
    "XGBoost": ("xgboost_model.pkl", "raw"),
    "LightGBM": ("lightgbm_model.pkl", "raw"),
    "SVM": ("svm_model.pkl", "scaled"),
    "ANN": ("ann_model.pkl", "scaled"),
    "ANN (Keras)": ("ann_keras_model.keras", "scaled"),
}


def load_test_data():
    """Load held-out test datasets."""

    X_test = pd.read_csv(X_TEST_FILE)
    X_test_scaled = pd.read_csv(X_TEST_SCALED_FILE)
    y_test = pd.read_csv(Y_TEST_FILE).squeeze()

    return X_test, X_test_scaled, y_test


def evaluate_model(name, model_file, data_type, X_test, X_test_scaled, y_test):
    """Evaluate one saved model on the untouched test set."""

    model_path = MODEL_DIR / model_file

    print(f"\nEvaluating {name}...")

    if model_file.endswith(".keras"):
        from tensorflow import keras                # type: ignore
        model = keras.models.load_model(model_path)
    else:
        model = joblib.load(model_path)

    X = X_test_scaled if data_type == "scaled" else X_test

    start_time = time.time()

    if model_file.endswith(".keras"):
        probabilities = model.predict(X, verbose=0).ravel()
        predictions = (probabilities >= 0.5).astype(int)
    elif hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
        predictions = model.predict(X)
    else:
        probabilities = model.decision_function(X)
        predictions = model.predict(X)

    evaluation_time = time.time() - start_time


    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)
    cm = confusion_matrix(y_test, predictions)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    return {
        "Model": name,
        "Dataset": "Test",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "Evaluation Time": evaluation_time,
    }


def main():
    """Run final held-out test evaluation."""

    print("\nLoading Test Data...")

    X_test, X_test_scaled, y_test = load_test_data()

    print(f"X_test Shape        : {X_test.shape}")
    print(f"X_test Scaled Shape : {X_test_scaled.shape}")
    print(f"y_test Shape        : {y_test.shape}")

    results = []

    for name, (model_file, data_type) in MODELS.items():
        result = evaluate_model(
            name,
            model_file,
            data_type,
            X_test,
            X_test_scaled,
            y_test,
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    output_file = OUTPUT_DIR / "final_test_results.csv"

    results_df.to_csv(output_file, index=False)

    print("\nFinal Test Results")
    print("-------------------------------------")
    print(results_df.to_string(index=False))

    print(f"\nResults Saved : {output_file}")
    print("\nFinal Test Evaluation Completed Successfully!")

if __name__ == "__main__":
    main()