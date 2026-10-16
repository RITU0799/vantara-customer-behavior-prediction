import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# Project path
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_data"
)

OUTPUT_DIR = INPUT_DIR

# Load data
X_train = pd.read_csv(INPUT_DIR / "X_train.csv")
X_val = pd.read_csv(INPUT_DIR / "X_val.csv")
X_test = pd.read_csv(INPUT_DIR / "X_test.csv")

print("\nData Loaded!")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# Create scaler
scaler = StandardScaler()


# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)

X_test_scaled = scaler.transform(X_test)


# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)

X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# Save scaled data
X_train_scaled.to_csv(OUTPUT_DIR / "X_train_scaled.csv", index=False)

X_val_scaled.to_csv(OUTPUT_DIR / "X_val_scaled.csv", index=False)

X_test_scaled.to_csv(OUTPUT_DIR / "X_test_scaled.csv", index=False)


# Save scaler
joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")

print("\nScaling Completed!")

print("Scaled Train:", X_train_scaled.shape)
print("Scaled Validation:", X_val_scaled.shape)
print("Scaled Test:", X_test_scaled.shape)

print("\nScaler Saved Successfully!")