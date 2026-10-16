import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

# Project path
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_ready_data.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_data"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(INPUT_FILE)
print("\nData Loaded:", df.shape)


# Separate features and target
customer_ids = df["Customer_ID"]
X = df.drop(columns=["Churn", "Customer_ID"])
y = df["Churn"]


# 70% Train, 30% Temporary
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)


# 15% Validation, 15% Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Keep Customer_ID mapping in exactly the same row order as the model splits.
customer_train = customer_ids.loc[X_train.index]
customer_val = customer_ids.loc[X_val.index]
customer_test = customer_ids.loc[X_test.index]


# Save datasets
X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)

X_val.to_csv(OUTPUT_DIR / "X_val.csv", index=False)

X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)

y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)

y_val.to_csv(OUTPUT_DIR / "y_val.csv", index=False)

y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

customer_train.to_csv(OUTPUT_DIR / "customer_ids_train.csv", index=False)
customer_val.to_csv(OUTPUT_DIR / "customer_ids_val.csv", index=False)
customer_test.to_csv(OUTPUT_DIR / "customer_ids_test.csv", index=False)


print("\nTrain:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("\nTrain Churn Rate:", round(y_train.mean(), 3))
print("Validation Churn Rate:", round(y_val.mean(), 3))
print("Test Churn Rate:", round(y_test.mean(), 3))

print("\nData Split Completed!")