import pandas as pd
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parents[2]

# File paths
INPUT_FILE = (
    BASE_DIR
    / "data"
    / "interim"
    / "combined_retail_data.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "interim"
    / "cleaned_retail_data.csv"
)

def clean_data(df):

    print("\nStarting Data Cleaning...")

    # -----------------------------
    # 1. Remove exact duplicates
    # -----------------------------

    duplicate_count = df.duplicated().sum()

    df = df.drop_duplicates().copy()

    print("Duplicate Rows Removed:", duplicate_count)

    # -----------------------------
    # 2. Identify cancellations
    # -----------------------------

    df["IsCancellation"] = (
        df["Invoice"]
        .astype(str)
        .str.upper()
        .str.startswith("C")
    )

    # Identify returns
    df["IsReturn"] = (
        (df["Quantity"] < 0)
        | df["IsCancellation"]
    )

    print("Cancellation Transactions:", df["IsCancellation"].sum())

    # -----------------------------
    # 3. Remove missing Customer ID
    # -----------------------------

    missing_customer = (df["Customer_ID"].isna().sum())

    df = df.dropna(subset=["Customer_ID"]).copy()

    print("Rows Removed - Missing Customer ID:", missing_customer)

    # -----------------------------
    # 4. Remove invalid dates
    # -----------------------------

    missing_date = (df["InvoiceDate"].isna().sum())

    df = df.dropna(subset=["InvoiceDate"]).copy()

    print("Rows Removed - Invalid Date:", missing_date)

    # -----------------------------
    # 5. Domain rules
    # -----------------------------

    # Zero quantity is invalid
    zero_quantity = (df["Quantity"] == 0).sum()

    df = df[df["Quantity"] != 0].copy()

    # Price must be positive
    invalid_price = (df["Price"] <= 0).sum()

    df = df[df["Price"] > 0].copy()

    print("Zero Quantity Rows Removed:", zero_quantity)

    print("Invalid Price Rows Removed:", invalid_price)

    # -----------------------------
    # 6. IQR outlier detection
    # -----------------------------

    quantity_values = df["Quantity"].abs()

    q1_quantity = quantity_values.quantile(0.25)
    q3_quantity = quantity_values.quantile(0.75)

    iqr_quantity = (q3_quantity - q1_quantity)

    quantity_upper = (q3_quantity + 1.5 * iqr_quantity)

    df["Quantity_Outlier"] = (quantity_values > quantity_upper)

    q1_price = df["Price"].quantile(0.25)
    q3_price = df["Price"].quantile(0.75)

    iqr_price = (q3_price - q1_price)

    price_upper = (q3_price + 1.5 * iqr_price)

    df["Price_Outlier"] = (df["Price"] > price_upper)

    print("Quantity Outliers:", df["Quantity_Outlier"].sum())

    print("Price Outliers:", df["Price_Outlier"].sum())

    # -----------------------------
    # 7. Standardize descriptions
    # -----------------------------

    df["Description"] = (
        df["Description"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # StockCode-based description lookup
    description_lookup = (
        df.groupby("StockCode")["Description"]
        .agg(
            lambda x:
            x.mode().iloc[0]
            if not x.mode().empty
            else x.iloc[0]
        )
    )

    df["Description"] = (
        df["StockCode"]
        .map(description_lookup)
        .fillna(df["Description"])
    )

    print("\nProduct Descriptions Standardized.")

    # -----------------------------
    # 8. Transaction amount
    # -----------------------------

    df["TotalAmount"] = (df["Quantity"] * df["Price"])

    # -----------------------------
    # 9. Sort chronologically
    # -----------------------------

    df = (df.sort_values("InvoiceDate").reset_index(drop=True))

    print("\nFinal Rows:", len(df))

    return df

def main():

    print("\nLoading Combined Dataset...")

    df = pd.read_csv(INPUT_FILE, parse_dates=["InvoiceDate"])

    df = clean_data(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nCleaned Dataset Saved to:", OUTPUT_FILE)

if __name__ == "__main__":
    main()