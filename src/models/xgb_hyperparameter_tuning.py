import time
import joblib
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# Paths
# ============================================================

X_TRAIN_PATH = "data/processed/model_data/X_train.csv"
X_VAL_PATH = "data/processed/model_data/X_val.csv"

Y_TRAIN_PATH = "data/processed/model_data/y_train.csv"
Y_VAL_PATH = "data/processed/model_data/y_val.csv"

MODEL_PATH = "data/processed/model_data/xgboost_tuned.pkl"


# ============================================================
# Load training and validation data
# ============================================================

print("\nLoading training and validation data...")

X_train = pd.read_csv(X_TRAIN_PATH)
X_val = pd.read_csv(X_VAL_PATH)

y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
y_val = pd.read_csv(Y_VAL_PATH).squeeze()

print("Training Shape   :", X_train.shape)
print("Validation Shape :", X_val.shape)


# ============================================================
# XGBoost model
# ============================================================

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)


# ============================================================
# Hyperparameter grid
# ============================================================

param_grid = {
    "n_estimators": [200, 400],
    "learning_rate": [0.03, 0.05],
    "max_depth": [3, 5],
    "min_child_weight": [1, 3],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}


# ============================================================
# Cross-validation
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


print("\nStarting XGBoost hyperparameter tuning...")
print("Scoring      : ROC-AUC")
print("CV folds     : 5")
print("Training set : Used for tuning only")
print("Validation   : Used after tuning")
print()


grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)


# ============================================================
# Train
# ============================================================

start_time = time.time()

grid_search.fit(X_train, y_train)

training_time = time.time() - start_time


# ============================================================
# Best model
# ============================================================

best_model = grid_search.best_estimator_

print("\n" + "=" * 60)
print("BEST XGBOOST PARAMETERS")
print("=" * 60)

print(grid_search.best_params_)

print("\nBest CV ROC-AUC :", round(grid_search.best_score_, 4))
print("Training Time   :", round(training_time, 2), "seconds")


# ============================================================
# Validation evaluation
# ============================================================

print("\nEvaluating on validation set...")

y_val_pred = best_model.predict(X_val)
y_val_prob = best_model.predict_proba(X_val)[:, 1]

accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred, zero_division=0)
recall = recall_score(y_val, y_val_pred, zero_division=0)
f1 = f1_score(y_val, y_val_pred, zero_division=0)
roc_auc = roc_auc_score(y_val, y_val_prob)


print("\n" + "=" * 60)
print("XGBOOST VALIDATION RESULTS")
print("=" * 60)

print("Accuracy  :", round(accuracy, 4))
print("Precision :", round(precision, 4))
print("Recall    :", round(recall, 4))
print("F1 Score  :", round(f1, 4))
print("ROC-AUC   :", round(roc_auc, 4))


# ============================================================
# Save tuned model
# ============================================================

joblib.dump(best_model, MODEL_PATH)

print("\nTuned XGBoost model saved to:")
print(MODEL_PATH)

print("\nXGBoost tuning completed.")