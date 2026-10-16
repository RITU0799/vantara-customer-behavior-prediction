from pathlib import Path
import sys
import time

import joblib
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)


# Project root
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.utils.model_logger import save_model_results


# Data paths
DATA_DIR = BASE_DIR / "data" / "processed" / "model_data"

X_TRAIN_FILE = DATA_DIR / "X_train.csv"
X_VAL_FILE = DATA_DIR / "X_val.csv"
Y_TRAIN_FILE = DATA_DIR / "y_train.csv"
Y_VAL_FILE = DATA_DIR / "y_val.csv"

MODEL_FILE = DATA_DIR / "xgboost_model.pkl"

print("\nLoading Training and Validation Data...")

# Load Data
X_train = pd.read_csv(X_TRAIN_FILE)
X_val = pd.read_csv(X_VAL_FILE)

y_train = pd.read_csv(Y_TRAIN_FILE).squeeze()
y_val = pd.read_csv(Y_VAL_FILE).squeeze()

print("\nData Loaded!")
print("Training   :", X_train.shape)
print("Validation :", X_val.shape)


# XGBoost
model = XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1,
)


# Expanded parameter search
param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [2, 3, 5],
    "min_child_weight": [1, 3],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


# 5-Fold Stratified Cross Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# Grid Search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1,
)


# Training
print("\nStarting XGBoost Hyperparameter Search...")

start_time = time.time()

grid_search.fit(
    X_train,
    y_train,
)

training_time = time.time() - start_time

best_model = grid_search.best_estimator_


# Save Best Model
joblib.dump(
    best_model,
    MODEL_FILE,
)

print(f"\nXGBoost Model Saved : {MODEL_FILE}")


# Best Parameters
print("\n========== XGBOOST ==========")

print(
    "Best Parameters:",
    grid_search.best_params_,
)

print(
    "Best CV ROC-AUC:",
    round(grid_search.best_score_, 4),
)


# Validation Evaluation
y_pred = best_model.predict(X_val)

y_prob = best_model.predict_proba(
    X_val
)[:, 1]


# Metrics
accuracy = accuracy_score(
    y_val,
    y_pred,
)

precision = precision_score(
    y_val,
    y_pred,
)

recall = recall_score(
    y_val,
    y_pred,
)

f1 = f1_score(
    y_val,
    y_pred,
)

roc_auc = roc_auc_score(
    y_val,
    y_prob,
)

cm = confusion_matrix(
    y_val,
    y_pred,
)


# Results
print("\nValidation Results:")
print(
    "Accuracy :",
    round(accuracy, 4),
)

print(
    "Precision:",
    round(precision, 4),
)

print(
    "Recall   :",
    round(recall, 4),
)

print(
    "F1 Score :",
    round(f1, 4),
)

print(
    "ROC-AUC  :",
    round(roc_auc, 4),
)

print(
    "\nConfusion Matrix:\n",
    cm,
)


# Save results
save_model_results(
    model_name="XGBoost",
    accuracy=accuracy,
    precision=precision,
    recall=recall,
    f1=f1,
    roc_auc=roc_auc,
    training_time=training_time,
)


print("\nXGBoost Results Saved Successfully!")

print(
    "\nTraining/Search Time:",
    round(training_time, 3),
    "seconds",
)