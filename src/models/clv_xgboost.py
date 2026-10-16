import time
import joblib
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score


# Paths
DATA_PATH = "data/processed/clv_regression_data.csv"
MODEL_PATH = "models_artifacts/clv_xgboost.pkl"


def load_data():
    print("\nLoading CLV regression data...")

    data = pd.read_csv(DATA_PATH)

    X = data.drop(columns=["Customer_ID", "Future_90D_CLV"])
    y = data["Future_90D_CLV"]

    return X, y


def train_model(X_train, y_train):
    print("\nTraining XGBoost Regressor...")

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

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

    return scores


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

    return mae, rmse, r2


def main():
    X, y = load_data()

    print(f"Feature Shape : {X.shape}")
    print(f"Target Shape  : {y.shape}")

    base_path = "data/processed/clv_regression"

    X_train = pd.read_csv(f"{base_path}/X_train.csv")
    X_val = pd.read_csv(f"{base_path}/X_val.csv")
    X_test = pd.read_csv(f"{base_path}/X_test.csv")

    y_train = pd.read_csv(f"{base_path}/y_train.csv").squeeze()
    pd.read_csv(f"{base_path}/y_val.csv").squeeze()
    y_test = pd.read_csv(f"{base_path}/y_test.csv").squeeze()

    print("\nDataset Split")
    print("-------------------------------------")
    print(f"Train : {X_train.shape}")
    print(f"Val   : {X_val.shape}")
    print(f"Test  : {X_test.shape}")

    model = train_model(X_train, y_train)
    cross_validate_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel Saved to : {MODEL_PATH}")
    print("\nCLV XGBoost Completed Successfully!")

if __name__ == "__main__":
    main()