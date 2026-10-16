import time
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
Y_TRAIN_FILE = MODEL_DATA_DIR / "y_train.csv"
Y_VAL_FILE = MODEL_DATA_DIR / "y_val.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training and validation data...")

X_train = pd.read_csv(X_TRAIN_FILE)
X_val = pd.read_csv(X_VAL_FILE)

y_train = pd.read_csv(Y_TRAIN_FILE).squeeze()
y_val = pd.read_csv(Y_VAL_FILE).squeeze()

print(f"Original training features : {X_train.shape[1]}")
print(f"Original validation features: {X_val.shape[1]}")


# ============================================================
# REMOVE COUNTRY FEATURES
# ============================================================

country_features = [
    column
    for column in X_train.columns
    if column.startswith("Country_")
]

X_train_reduced = X_train.drop(columns=country_features)
X_val_reduced = X_val.drop(columns=country_features)

print(f"\nCountry features removed   : {len(country_features)}")
print(f"Remaining training features: {X_train_reduced.shape[1]}")
print(f"Remaining validation features: {X_val_reduced.shape[1]}")


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest without country features...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)

start_time = time.time()

model.fit(X_train_reduced, y_train)

training_time = time.time() - start_time


# ============================================================
# VALIDATION EVALUATION
# ============================================================

print("\nEvaluating on validation set...")

start_time = time.time()

y_pred = model.predict(X_val_reduced)
y_prob = model.predict_proba(X_val_reduced)[:, 1]

evaluation_time = time.time() - start_time


accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_prob)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST - WITHOUT COUNTRY FEATURES")
print("VALIDATION RESULTS")
print("=" * 60)

print(f"Accuracy       : {accuracy:.6f}")
print(f"Precision      : {precision:.6f}")
print(f"Recall         : {recall:.6f}")
print(f"F1 Score       : {f1:.6f}")
print(f"ROC-AUC        : {roc_auc:.6f}")
print(f"Training Time  : {training_time:.2f} seconds")
print(f"Evaluation Time: {evaluation_time:.4f} seconds")


# ============================================================
# BASELINE COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION BASELINE COMPARISON")
print("=" * 60)

print("Current full-feature RF:")
print("ROC-AUC = 0.8196")
print("Recall  = 0.7411")

print("\nRF without country features:")
print(f"ROC-AUC = {roc_auc:.6f}")
print(f"Recall  = {recall:.6f}")

print("\nDifference:")
print(f"ROC-AUC change = {roc_auc - 0.8196:+.6f}")
print(f"Recall change  = {recall - 0.7411:+.6f}")


# ============================================================
# TOP FEATURES
# ============================================================

print("\nTop 20 features without country encoding:")

importance = pd.Series(
    model.feature_importances_,
    index=X_train_reduced.columns,
).sort_values(ascending=False)

print(importance.head(20).to_string())


print("\nExperiment completed.")
print("No existing model, test data, or final results were modified.")