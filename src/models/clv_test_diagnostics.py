import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

MODEL_PATH = "models_artifacts/clv_hurdle.pkl"
DATA_DIR = "data/processed/clv_regression"


model = joblib.load(MODEL_PATH)

classifier = model["classifier"]
regressor = model["regressor"]

X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()

actual = np.expm1(y_test)

positive_probability = classifier.predict_proba(X_test)[:, 1]

positive_clv_log = regressor.predict(X_test)
positive_clv_log = np.clip(positive_clv_log, -20, 20)

positive_clv = np.expm1(positive_clv_log)

predictions = positive_probability * positive_clv

print("\nCLV Test Diagnostics")
print("-------------------------------------")

print(f"Actual Mean      : {actual.mean():.4f}")
print(f"Prediction Mean  : {predictions.mean():.4f}")

print(f"Actual Median    : {np.median(actual):.4f}")
print(f"Prediction Median: {np.median(predictions):.4f}")

print(f"Actual Maximum   : {actual.max():.4f}")
print(f"Prediction Maximum: {predictions.max():.4f}")

print(f"Test R²          : {r2_score(actual, predictions):.4f}")

print("\nTop 10 Actual vs Predicted CLV")
print("-------------------------------------")

diagnostics = pd.DataFrame({
    "Actual_CLV": actual,
    "Predicted_CLV": predictions
})

diagnostics["Absolute_Error"] = (
    diagnostics["Actual_CLV"] -
    diagnostics["Predicted_CLV"]
).abs()

print(
    diagnostics
    .sort_values("Actual_CLV", ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\nTop 10 Prediction Errors")
print("-------------------------------------")

print(
    diagnostics
    .sort_values("Absolute_Error", ascending=False)
    .head(10)
    .to_string(index=False)
)