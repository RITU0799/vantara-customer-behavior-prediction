import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


# Paths
MODEL_PATH = "models_artifacts/clv_random_forest.pkl"


def train_model(X_train, y_train):
    """Train Random Forest on log-transformed CLV target."""

    print("\nTraining Random Forest Regressor...")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    start_time = time.time()

    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"Training Time : {training_time:.2f} seconds")

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate predictions after converting back to original CLV scale."""

    print("\nEvaluating CLV Model...")

    # Predictions are on log scale
    predictions_log = model.predict(X_test)

    # Convert predictions and actual values back to original CLV scale
    predictions = np.expm1(predictions_log)
    actual = np.expm1(y_test)

    mae = mean_absolute_error(actual, predictions)
    rmse = mean_squared_error(actual, predictions) ** 0.5
    r2 = r2_score(actual, predictions)

    print("\nCLV Regression Results")
    print("-------------------------------------")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return mae, rmse, r2


def cross_validate_model(model, X_train, y_train):
    """Run 5-fold CV and evaluate each fold on original CLV scale."""

    print("\nRunning 5-Fold Cross-Validation...")

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = []

    for fold, (train_index, val_index) in enumerate(
        kfold.split(X_train),
        start=1
    ):
        X_fold_train = X_train.iloc[train_index]
        X_fold_val = X_train.iloc[val_index]

        y_fold_train = y_train.iloc[train_index]
        y_fold_val = y_train.iloc[val_index]

        fold_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        fold_model.fit(X_fold_train, y_fold_train)

        predictions_log = fold_model.predict(X_fold_val)

        predictions = np.expm1(predictions_log)
        actual = np.expm1(y_fold_val)

        score = r2_score(actual, predictions)

        scores.append(score)

    scores = np.array(scores)

    print("\n5-Fold CV R² Scores")
    print("-------------------------------------")

    for i, score in enumerate(scores, start=1):
        print(f"Fold {i}: {score:.4f}")

    print(f"\nMean CV R² : {scores.mean():.4f}")
    print(f"Std CV R²  : {scores.std():.4f}")

    return scores


def main():
    print("\nLoading CLV regression data...")

    # Load preprocessing split
    X_train = pd.read_csv("data/processed/clv_regression/X_train.csv")
    X_val = pd.read_csv("data/processed/clv_regression/X_val.csv")
    X_test = pd.read_csv("data/processed/clv_regression/X_test.csv")

    y_train = pd.read_csv("data/processed/clv_regression/y_train.csv").squeeze()
    pd.read_csv("data/processed/clv_regression/y_val.csv").squeeze()
    y_test = pd.read_csv("data/processed/clv_regression/y_test.csv").squeeze()

    print(f"Feature Shape : {X_train.shape}")
    print(f"Target Shape  : {y_train.shape}")

    print("\nDataset Split")
    print("-------------------------------------")
    print(f"Train : {X_train.shape}")
    print(f"Val   : {X_val.shape}")
    print(f"Test  : {X_test.shape}")

    print("\nTarget Scale : Log-transformed")

    model = train_model(X_train, y_train)

    cross_validate_model(model, X_train, y_train)

    evaluate_model(model, X_test, y_test)

    # Save model
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel Saved to : {MODEL_PATH}")

    print("\nCLV Random Forest Completed Successfully!")

if __name__ == "__main__":
    main()