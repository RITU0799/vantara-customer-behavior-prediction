import pandas as pd
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "interim" / "cleaned_retail_data.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "customer_encoding_features.csv"


# Existing churn observation date
OBSERVATION_DATE = pd.Timestamp("2011-09-10 12:50")


def create_country_encoding(data):
    """Create customer-level one-hot encoded Country features."""

    country_data = data.copy()

    # One-hot encode Country
    country_encoded = pd.get_dummies(
        country_data["Country"],
        prefix="Country",
        dtype=int
    )

    country_encoded["Customer_ID"] = country_data["Customer_ID"].values

    # Aggregate to customer level
    country_features = (
        country_encoded
        .groupby("Customer_ID")
        .max()
        .reset_index()
    )

    return country_features


def create_stockcode_frequency(data):
    """Create StockCode frequency encoding and aggregate it per customer."""

    stockcode_data = data.copy()

    # Calculate StockCode frequency using historical transactions only
    stockcode_frequency = (
        stockcode_data["StockCode"]
        .value_counts(normalize=True)
        .rename("StockCode_Frequency")
        .reset_index()
    )

    stockcode_frequency.columns = [
        "StockCode",
        "StockCode_Frequency"
    ]

    # Merge frequency values
    stockcode_data = stockcode_data.merge(
        stockcode_frequency,
        on="StockCode",
        how="left"
    )

    # Aggregate StockCode frequency to customer level
    customer_stockcode = (
        stockcode_data
        .groupby("Customer_ID")["StockCode_Frequency"]
        .mean()
        .reset_index()
    )

    return customer_stockcode


def main():
    print("\nLoading Cleaned Retail Data...")

    data = pd.read_csv(INPUT_FILE, parse_dates=["InvoiceDate"])

    print(f"Input Shape : {data.shape}")

    # Keep only historical transactions available at observation date
    data = data[
        (data["InvoiceDate"] <= OBSERVATION_DATE)
        & (data["Quantity"] > 0)
    ].copy()

    print(f"Historical Data Shape : {data.shape}")

    # Remove rows without customer IDs
    data = data.dropna(subset=["Customer_ID"])

    print("\nCreating Country encoding...")
    country_features = create_country_encoding(data)

    print(f"Country Features Shape : {country_features.shape}")

    print("\nCreating StockCode Frequency Encoding...")
    stockcode_features = create_stockcode_frequency(data)

    print(f"StockCode Features Shape : {stockcode_features.shape}")

    # Merge both customer-level feature sets
    encoding_features = country_features.merge(
        stockcode_features,
        on="Customer_ID",
        how="outer"
    )

    # Fill missing encoding values
    encoding_features = encoding_features.fillna(0)

    # Save output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    encoding_features.to_csv(OUTPUT_FILE, index=False)

    print("\nEncoding Completed Successfully!")
    print(f"Final Shape : {encoding_features.shape}")
    print(f"Saved File : {OUTPUT_FILE}")

if __name__ == "__main__":
    main()