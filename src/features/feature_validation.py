import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_ready_data.csv"
)

def validate_features(df):

    print("\n" + "=" * 50)
    print("FEATURE QUALITY VALIDATION")
    print("=" * 50)

    # --------------------------------------------------
    # 1. Dataset Shape
    # --------------------------------------------------

    print("\n1. Dataset Shape")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    # --------------------------------------------------
    # 2. Duplicate Customers
    # --------------------------------------------------

    print("\n2. Duplicate Customer Check")

    duplicate_customers = df["Customer_ID"].duplicated().sum()

    print("Duplicate Customers:", duplicate_customers)

    # --------------------------------------------------
    # 3. Missing Values
    # --------------------------------------------------

    print("\n3. Missing Value Check")

    missing_values = df.isnull().sum()

    total_missing = missing_values.sum()

    print("Total Missing Values:", total_missing)

    if total_missing == 0:
        print("Missing Value Check: Passed")
    else:
        print("Missing Value Check: Review Required")

    # --------------------------------------------------
    # 4. Infinite Values
    # --------------------------------------------------

    print("\n4. Infinite Value Check")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    infinite_values = np.isinf(
        df[numeric_columns]
    ).sum().sum()

    print("Infinite Values:", infinite_values)

    if infinite_values == 0:
        print("Infinite Value Check: Passed")
    else:
        print("Infinite Value Check: Review Required")

    # --------------------------------------------------
    # 5. Data Types
    # --------------------------------------------------

    print("\n5. Data Type Check")

    print(df.dtypes.value_counts())

    # --------------------------------------------------
    # 6. Target Distribution
    # --------------------------------------------------

    print("\n6. Churn Target Distribution")

    churn_counts = df["Churn"].value_counts()

    print(churn_counts)

    churn_rate = df["Churn"].mean()

    print(
        "Churn Rate:",
        round(churn_rate * 100, 2),
        "%"
    )

    # --------------------------------------------------
    # 7. Feature Range Check
    # --------------------------------------------------

    print("\n7. Feature Range Check")

    print(
        "Recency Min:",
        df["Recency"].min()
    )

    print(
        "Frequency Min:",
        df["Frequency"].min()
    )

    print(
        "Monetary Min:",
        round(df["Monetary"].min(), 2)
    )

    print(
        "Return Rate Min:",
        df["Return_Rate"].min()
    )

    print(
        "Return Rate Max:",
        df["Return_Rate"].max()
    )

    print(
        "Price Sensitivity Min:",
        df["Price_Sensitivity_Proxy"].min()
    )

    print(
        "Price Sensitivity Max:",
        df["Price_Sensitivity_Proxy"].max()
    )

    # --------------------------------------------------
    # 8. Negative Value Check
    # --------------------------------------------------

    print("\n8. Negative Feature Check")

    negative_values = (
        df[numeric_columns] < 0
    ).sum()

    negative_features = (
        negative_values[
            negative_values > 0
        ]
    )

    if len(negative_features) == 0:
        print(
            "Negative Feature Check: Passed"
        )
    else:
        print(
            "Features With Negative Values:"
        )
        print(negative_features)

    # --------------------------------------------------
    # 9. Leakage Check
    # --------------------------------------------------

    print("\n9. Leakage Check")

    leakage_columns = [
        "Churn"
    ]

    model_features = [
        column
        for column in df.columns
        if column not in leakage_columns
    ]

    print("Target column excluded:", leakage_columns)

    print("Model Features:", len(model_features))

    # --------------------------------------------------
    # 10. Feature List
    # --------------------------------------------------

    print("\n10. Final Feature List")

    for number, column in enumerate(
        df.columns,
        start=1
    ):
        print(
            number,
            "-",
            column
        )

    print("\n" + "=" * 50)
    print("FEATURE VALIDATION COMPLETED")
    print("=" * 50)

if __name__ == "__main__":

    print("\nLoading ML-ready Data...")

    df = pd.read_csv(INPUT_FILE)

    print("Rows Loaded:", len(df))

    validate_features(df)