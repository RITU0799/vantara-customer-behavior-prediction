from pathlib import Path
import sys
import time
import joblib

# Project root
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.utils.model_logger import save_model_results

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)


# Data path
DATA_DIR = BASE_DIR / "data" / "processed" / "model_data"


# Load Data
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_val = pd.read_csv(DATA_DIR / "X_val.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_val = pd.read_csv(DATA_DIR / "y_val.csv").squeeze()

print("\nData Loaded!")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)


# Random Forest
model = RandomForestClassifier(random_state=42, class_weight="balanced")

# Parameters to tune
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10],
    "min_samples_leaf": [1, 2]
}

# 5-Fold Stratified Cross Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)


# Training
start_time = time.time()
grid_search.fit(X_train, y_train)
training_time = time.time() - start_time

best_model = grid_search.best_estimator_


# Save trained model
MODEL_PATH = DATA_DIR / "random_forest.pkl"

joblib.dump(best_model, MODEL_PATH)

print("\nRandom Forest Model Saved Successfully!")
print("Model Path:", MODEL_PATH)

print("\n========== RANDOM FOREST ==========")
print("Best Parameters:", grid_search.best_params_)
print("Best CV ROC-AUC:", round(grid_search.best_score_, 4))


# Validation Evaluation
y_pred = best_model.predict(X_val)
y_prob = best_model.predict_proba(X_val)[:, 1]


# Metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_prob)
cm = confusion_matrix(y_val, y_pred)


# Results
print("\nValidation Results:")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))

print("\nConfusion Matrix:\n", cm)


# Save results
save_model_results(
    model_name="Random Forest",
    accuracy=accuracy,
    precision=precision,
    recall=recall,
    f1=f1,
    roc_auc=roc_auc,
    training_time=training_time
)

print("\nRandom Forest Results Saved Successfully!")

print("\nTraining/Search Time:", round(training_time, 3), "seconds")