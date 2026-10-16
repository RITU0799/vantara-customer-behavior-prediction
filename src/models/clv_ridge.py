import time
import joblib
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score


# Path
MODEL_PATH = "models_artifacts/clv_ridge.pkl"


def train_model(X_train, y_train):
    print("\nTraining Ridge Regression...")

    model = Ridge(alpha=10.0)

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"Training Time : {training_time:.2f} seconds")

    return model


def cross_validate_model(model, X_train, y_train):
    print("\nRunning 5-Fold Cross-Validation...")

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=kfold,
        scoring="r2",
        n_jobs=-1
    )

    print("\n5-Fold CV R² Scores")
    print("-------------------------------------")

    for i, score in enumerate(scores, start=1):
        print(f"Fold {i}: {score:.4f}")

    print(f"\nMean CV R² : {scores.mean():.4f}")
    print(f"Std CV R²  : {scores.std():.4f}")


def evaluate_model(model, X_test, y_test):
    print("\nEvaluating CLV Model...")

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\nCLV Regression Results")
    print("-------------------------------------")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return r2


def main():
    base_path = "data/processed/clv_regression"

    print("\nLoading scaled CLV data...")

    X_train = pd.read_csv(f"{base_path}/X_train_scaled.csv")
    X_test = pd.read_csv(f"{base_path}/X_test_scaled.csv")

    y_train = pd.read_csv(f"{base_path}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{base_path}/y_test.csv").squeeze()

    print(f"Train : {X_train.shape}")
    print(f"Test  : {X_test.shape}")

    model = train_model(X_train, y_train)
    cross_validate_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel Saved to : {MODEL_PATH}")
    print("\nCLV Ridge Regression Completed Successfully!")

if __name__ == "__main__":
    main()