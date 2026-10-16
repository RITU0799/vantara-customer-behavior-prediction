import pandas as pd
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parents[2]

# Input file
INPUT_FILE = (
    BASE_DIR
    / "data"
    / "interim"
    / "combined_retail_data.csv"
)

# Expected columns
REQUIRED_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer_ID",
    "Country"
]

# Maximum allowed missing percentage
NULL_THRESHOLDS = {
    "Invoice": 0.00,
    "StockCode": 0.00,
    "Description": 0.05,
    "Quantity": 0.00,
    "InvoiceDate": 0.00,
    "Price": 0.00,
    "Customer_ID": 0.30,
    "Country": 0.00
}

def validate_data(df):

    print("\nStarting Data Validation...")

    # -----------------------------
    # 1. Schema check
    # -----------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print("Schema Check: Passed")

    # -----------------------------
    # 2. Null-rate check
    # -----------------------------

    null_rates = df[REQUIRED_COLUMNS].isnull().mean()

    for column, threshold in NULL_THRESHOLDS.items():

        if null_rates[column] > threshold:
            raise ValueError(
                f"Null-rate check failed for {column}: "
                f"{null_rates[column]:.2%}"
            )

    print("Null-Rate Check: Passed")

    print("\nMissing Values:")
    print(df[REQUIRED_COLUMNS].isnull().sum())

    # -----------------------------
    # 3. Date-range check
    # -----------------------------

    min_date = df["InvoiceDate"].min()
    max_date = df["InvoiceDate"].max()

    expected_start = pd.Timestamp("2009-01-01")
    expected_end = pd.Timestamp("2011-12-31")

    if min_date < expected_start:
        raise ValueError(f"Date range starts before expected range: {min_date}")

    if max_date > expected_end:
        raise ValueError(f"Date range ends after expected range: {max_date}")

    print("\nDate-Range Check: Passed")
    print("Date Range:", min_date, "to", max_date)

    # -----------------------------
    # 4. Duplicate check
    # -----------------------------

    duplicate_count = df.duplicated().sum()

    print("\nDuplicate Rows:", duplicate_count)

    # -----------------------------
    # 5. Domain checks
    # -----------------------------

    invalid_quantity = (df["Quantity"] <= 0).sum()

    invalid_price = (df["Price"] <= 0).sum()

    print("Non-positive Quantity:", invalid_quantity)
    print("Non-positive Price:", invalid_price)

    # -----------------------------
    # 6. Cancellation check
    # -----------------------------

    cancellation_count = (
        df["Invoice"]
        .astype(str)
        .str.upper()
        .str.startswith("C")
    ).sum()

    print("Cancellation Transactions:", cancellation_count)

    print("\nData Validation Completed Successfully!")

    return True

def main():

    print("\nLoading Combined Dataset...")

    df = pd.read_csv(INPUT_FILE, parse_dates=["InvoiceDate"])

    validate_data(df)

if __name__ == "__main__":
    main()