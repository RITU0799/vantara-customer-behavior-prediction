import time
import joblib
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
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

X_TRAIN_FILE = MODEL_DATA_DIR / "X_train.csv"
X_VAL_FILE = MODEL_DATA_DIR / "X_val.csv"
X_TEST_FILE = MODEL_DATA_DIR / "X_test.csv"

Y_TRAIN_FILE = MODEL_DATA_DIR / "y_train.csv"
Y_VAL_FILE = MODEL_DATA_DIR / "y_val.csv"
Y_TEST_FILE = MODEL_DATA_DIR / "y_test.csv"

MODEL_OUTPUT = MODEL_DATA_DIR / "random_forest_23_features.pkl"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading train, validation and test data...")

X_train = pd.read_csv(X_TRAIN_FILE)
X_val = pd.read_csv(X_VAL_FILE)
X_test = pd.read_csv(X_TEST_FILE)

y_train = pd.read_csv(Y_TRAIN_FILE).squeeze()
y_val = pd.read_csv(Y_VAL_FILE).squeeze()
y_test = pd.read_csv(Y_TEST_FILE).squeeze()


# ============================================================
# REMOVE COUNTRY FEATURES
# ============================================================

country_features = [
    column
    for column in X_train.columns
    if column.startswith("Country_")
]

X_train = X_train.drop(columns=country_features)
X_val = X_val.drop(columns=country_features)
X_test = X_test.drop(columns=country_features)

print(f"Country features removed : {len(country_features)}")
print(f"Final feature count      : {X_train.shape[1]}")


# ============================================================
# COMBINE TRAIN + VALIDATION
# ============================================================

print("\nCombining training and validation data...")

X_train_final = pd.concat(
    [X_train, X_val],
    axis=0,
    ignore_index=True,
)

y_train_final = pd.concat(
    [y_train, y_val],
    axis=0,
    ignore_index=True,
)

print(f"Final training rows : {X_train_final.shape[0]}")
print(f"Final training features : {X_train_final.shape[1]}")
print(f"Final test rows : {X_test.shape[0]}")


# ============================================================
# TRAIN FINAL RANDOM FOREST
# ============================================================

print("\nTraining final Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)

start_time = time.time()

model.fit(X_train_final, y_train_final)

training_time = time.time() - start_time


# ============================================================
# SAVE CANDIDATE MODEL
# ============================================================

joblib.dump(model, MODEL_OUTPUT)

print("\nModel saved to:")
print(MODEL_OUTPUT)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\nEvaluating final model on untouched test set...")

start_time = time.time()

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

evaluation_time = time.time() - start_time


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL RANDOM FOREST - 23 FEATURES")
print("=" * 60)

print(f"Accuracy       : {accuracy:.6f}")
print(f"Precision      : {precision:.6f}")
print(f"Recall         : {recall:.6f}")
print(f"F1 Score       : {f1:.6f}")
print(f"ROC-AUC        : {roc_auc:.6f}")
print(f"Training Time  : {training_time:.2f} seconds")
print(f"Evaluation Time: {evaluation_time:.4f} seconds")


# ============================================================
# PRD TARGET CHECK
# ============================================================

print("\n" + "=" * 60)
print("PRD TARGET CHECK")
print("=" * 60)

print(
    f"ROC-AUC target : >= 0.80 | "
    f"Actual: {roc_auc:.6f} | "
    f"{'PASS' if roc_auc >= 0.80 else 'NOT MET'}"
)

print(
    f"Recall target  : >= 0.70 | "
    f"Actual: {recall:.6f} | "
    f"{'PASS' if recall >= 0.70 else 'NOT MET'}"
)

print("\nExisting model and final_test_results.csv were NOT modified.")