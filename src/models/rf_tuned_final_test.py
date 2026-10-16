import time
import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DATA_DIR = BASE_DIR / "data" / "processed" / "model_data"

MODEL_FILE = MODEL_DATA_DIR / "random_forest_tuned.pkl"

X_TEST_FILE = MODEL_DATA_DIR / "X_test.csv"
Y_TEST_FILE = MODEL_DATA_DIR / "y_test.csv"


# ============================================================
# LOAD MODEL AND TEST DATA
# ============================================================

print("Loading tuned Random Forest...")

model = joblib.load(MODEL_FILE)

X_test = pd.read_csv(X_TEST_FILE)
y_test = pd.read_csv(Y_TEST_FILE).squeeze()

print(f"Test shape    : {X_test.shape}")
print(f"Model features: {model.n_features_in_}")


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\nEvaluating tuned Random Forest on untouched test set...")

start_time = time.time()

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

evaluation_time = time.time() - start_time


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TUNED RANDOM FOREST - FINAL TEST RESULTS")
print("=" * 60)

print(f"Accuracy       : {accuracy:.6f}")
print(f"Precision      : {precision:.6f}")
print(f"Recall         : {recall:.6f}")
print(f"F1 Score       : {f1:.6f}")
print(f"ROC-AUC        : {roc_auc:.6f}")
print(f"Evaluation Time: {evaluation_time:.4f} seconds")


# ============================================================
# PRD TARGET CHECK
# ============================================================

print("\n" + "=" * 60)
print("PRD TARGET CHECK")
print("=" * 60)

print(
    f"ROC-AUC >= 0.80 : "
    f"{'PASS' if roc_auc >= 0.80 else 'NOT MET'} "
    f"({roc_auc:.6f})"
)

print(
    f"Recall >= 0.70  : "
    f"{'PASS' if recall >= 0.70 else 'NOT MET'} "
    f"({recall:.6f})"
)

print("\nThis evaluation uses the untouched test set.")
print("No existing model or final_test_results.csv was modified.")