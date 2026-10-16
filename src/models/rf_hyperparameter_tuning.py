import time
import joblib
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DATA_DIR = BASE_DIR / "data" / "processed" / "model_data"

X_TRAIN_FILE = MODEL_DATA_DIR / "X_train.csv"
X_VAL_FILE = MODEL_DATA_DIR / "X_val.csv"

Y_TRAIN_FILE = MODEL_DATA_DIR / "y_train.csv"
Y_VAL_FILE = MODEL_DATA_DIR / "y_val.csv"

MODEL_OUTPUT = MODEL_DATA_DIR / "random_forest_tuned.pkl"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training and validation data...")

X_train = pd.read_csv(X_TRAIN_FILE)
X_val = pd.read_csv(X_VAL_FILE)

y_train = pd.read_csv(Y_TRAIN_FILE).squeeze()
y_val = pd.read_csv(Y_VAL_FILE).squeeze()

print(f"Training shape   : {X_train.shape}")
print(f"Validation shape: {X_val.shape}")


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)


# ============================================================
# PARAMETER GRID
# ============================================================

param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [5, 8, 10, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}


# ============================================================
# CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# GRID SEARCH
# ============================================================

print("\nStarting Random Forest hyperparameter tuning...")
print("Scoring: ROC-AUC")
print("This uses training data only.")

start_time = time.time()

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

grid_search.fit(X_train, y_train)

tuning_time = time.time() - start_time


# ============================================================
# BEST PARAMETERS
# ============================================================

print("\n" + "=" * 60)
print("BEST RANDOM FOREST PARAMETERS")
print("=" * 60)

print(grid_search.best_params_)
print(f"Best CV ROC-AUC: {grid_search.best_score_:.6f}")
print(f"Tuning time    : {tuning_time:.2f} seconds")


# ============================================================
# VALIDATION EVALUATION
# ============================================================

best_model = grid_search.best_estimator_

print("\nEvaluating best model on validation set...")

val_probability = best_model.predict_proba(X_val)[:, 1]

val_auc = roc_auc_score(
    y_val,
    val_probability
)

print("\n" + "=" * 60)
print("VALIDATION RESULT")
print("=" * 60)

print(f"Validation ROC-AUC: {val_auc:.6f}")


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(best_model, MODEL_OUTPUT)

print("\nTuned model saved to:")
print(MODEL_OUTPUT)


# ============================================================
# PRD CHECK
# ============================================================

print("\n" + "=" * 60)
print("PRD TARGET CHECK")
print("=" * 60)

if val_auc >= 0.80:
    print("ROC-AUC target >= 0.80 : PASS")
else:
    print("ROC-AUC target >= 0.80 : NOT MET")

print("\nTest set was NOT used.")
print("Existing random_forest.pkl was NOT modified.")
print("Existing final_test_results.csv was NOT modified.")