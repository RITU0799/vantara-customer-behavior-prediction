from pathlib import Path
import sys

import joblib
import pandas as pd


# Project root
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))


# Data paths
DATA_DIR = BASE_DIR / "data" / "processed" / "model_data"

MODEL_FILE = DATA_DIR / "xgboost_model.pkl"
X_TRAIN_FILE = DATA_DIR / "X_train.csv"

OUTPUT_FILE = (DATA_DIR / "xgboost_feature_importance.csv")


print("\nLoading XGBoost Model...")

model = joblib.load(MODEL_FILE)

X_train = pd.read_csv(X_TRAIN_FILE)

print("Training Data Shape:", X_train.shape)


# Feature importance
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_,
})


# Sort by importance
importance = importance.sort_values(
    by="Importance",
    ascending=False,
).reset_index(drop=True)


# Save results
importance.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("\nTop 20 XGBoost Features")
print("-------------------------------------")

print(
    importance.head(20).to_string(
        index=False
    )
)

print(f"\nFeature Importance Saved : {OUTPUT_FILE}")