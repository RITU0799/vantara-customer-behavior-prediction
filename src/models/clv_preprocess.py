import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# Paths
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data" / "processed" / "clv_regression_data.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "clv_regression"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load Data
print("\nLoading CLV regression data...")

data = pd.read_csv(INPUT_PATH)

print("Input Shape :", data.shape)


# Separate Features and Target
X = data.drop(
    columns=["Customer_ID", "Future_90D_CLV"]
)

y = data["Future_90D_CLV"]

print("Feature Shape :", X.shape)
print("Target Shape  :", y.shape)


# CLV Target Distribution
print("\nCLV Target Distribution")
print("-------------------------------------")

print("Zero CLV     :", (y == 0).sum())
print("Positive CLV :", (y > 0).sum())
print("Mean CLV     :", round(y.mean(), 2))
print("Max CLV      :", round(y.max(), 2))


# Log Transform Target
y_log = np.log1p(y)


# Create CLV Strata
def create_clv_strata(target):
    """
    Create representative CLV groups for stratified splitting.

    0 = Zero CLV
    1 = Low positive CLV
    2 = Medium positive CLV
    3 = High positive CLV
    4 = Very high positive CLV
    """

    target = pd.Series(
        target
    ).reset_index(drop=True)

    strata = pd.Series(
        np.zeros(len(target), dtype=int)
    )

    positive_mask = target > 0

    # Low positive CLV
    strata.loc[
        positive_mask & (target <= 500)
    ] = 1

    # Medium positive CLV
    strata.loc[
        positive_mask
        & (target > 500)
        & (target <= 1500)
    ] = 2

    # High positive CLV
    strata.loc[
        positive_mask
        & (target > 1500)
        & (target <= 5000)
    ] = 3

    # Very high positive CLV
    strata.loc[
        positive_mask
        & (target > 5000)
    ] = 4

    return strata


# Create strata using original CLV target
clv_strata = create_clv_strata(y)


print("\nCLV Strata Distribution")
print("-------------------------------------")

for stratum, count in clv_strata.value_counts().sort_index().items():

    if stratum == 0:
        label = "Zero CLV"
    elif stratum == 1:
        label = "Low Positive CLV"
    elif stratum == 2:
        label = "Medium Positive CLV"
    elif stratum == 3:
        label = "High Positive CLV"
    else:
        label = "Very High Positive CLV"

    print(
        f"Stratum {stratum} ({label}) : {count}"
    )


# Train / Validation / Test Split
print(
    "\nCreating stratified train / validation / test split..."
)


# First split:
# 70% Train
# 30% Temporary
X_train, X_temp, y_train, y_temp, strata_train, strata_temp = train_test_split(
    X,
    y_log,
    clv_strata,
    test_size=0.30,
    random_state=42,
    stratify=clv_strata
)


# Second split:
# 15% Validation
# 15% Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=strata_temp
)


print("\nSplit Shapes")
print("-------------------------------------")

print("Train :", X_train.shape)
print("Val   :", X_val.shape)
print("Test  :", X_test.shape)


# Scale Features
print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)

X_test_scaled = scaler.transform(
    X_test
)


# Save Raw Features
X_train.to_csv(
    OUTPUT_DIR / "X_train.csv",
    index=False
)

X_val.to_csv(
    OUTPUT_DIR / "X_val.csv",
    index=False
)

X_test.to_csv(
    OUTPUT_DIR / "X_test.csv",
    index=False
)


# Save Scaled Features
pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
).to_csv(
    OUTPUT_DIR / "X_train_scaled.csv",
    index=False
)

pd.DataFrame(
    X_val_scaled,
    columns=X_val.columns
).to_csv(
    OUTPUT_DIR / "X_val_scaled.csv",
    index=False
)

pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns
).to_csv(
    OUTPUT_DIR / "X_test_scaled.csv",
    index=False
)


# Save Log-Transformed Targets
pd.Series(
    y_train,
    name="Future_90D_CLV"
).to_csv(
    OUTPUT_DIR / "y_train.csv",
    index=False
)

pd.Series(
    y_val,
    name="Future_90D_CLV"
).to_csv(
    OUTPUT_DIR / "y_val.csv",
    index=False
)

pd.Series(
    y_test,
    name="Future_90D_CLV"
).to_csv(
    OUTPUT_DIR / "y_test.csv",
    index=False
)


# Save Scaler
joblib.dump(
    scaler,
    OUTPUT_DIR / "clv_scaler.pkl"
)


# Validation Checks
print("\nMissing Values")
print("-------------------------------------")

print(
    "Train :",
    X_train.isna().sum().sum()
)

print(
    "Val   :",
    X_val.isna().sum().sum()
)

print(
    "Test  :",
    X_test.isna().sum().sum()
)


# CLV Distribution by Split
print("\nCLV Distribution by Split")
print("-------------------------------------")


for name, target in [
    ("Train", y_train),
    ("Val", y_val),
    ("Test", y_test)
]:

    original_target = np.expm1(
        target
    )

    zero_count = (
        original_target == 0
    ).sum()

    positive_count = (
        original_target > 0
    ).sum()

    print(
        f"{name}: "
        f"Zero={zero_count}, "
        f"Positive={positive_count}, "
        f"Mean={original_target.mean():.2f}, "
        f"Max={original_target.max():.2f}"
    )


print("\nCLV Preprocessing Completed Successfully!")

print(
    "\nFiles saved to:",
    OUTPUT_DIR
)