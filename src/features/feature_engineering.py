import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "interim"
    / "cleaned_retail_data.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_features.csv"
)

def create_features(df):

    print("\nStarting Feature Engineering...")

    # --------------------------------------------------
    # STEP 1: BASIC CUSTOMER FEATURES
    # --------------------------------------------------

    # Calculate transaction value
    df["TotalAmount"] = df["Quantity"] * df["Price"]

    # Reference date
    reference_date = df["InvoiceDate"].max()

    # Customer-level features
    customer_features = df.groupby("Customer_ID").agg(
        Recency=(
            "InvoiceDate",
            lambda x: (reference_date - x.max()).days
        ),
        Frequency=(
            "Invoice",
            "nunique"
        ),
        Monetary=(
            "TotalAmount",
            "sum"
        ),
        Total_Quantity=(
            "Quantity",
            "sum"
        ),
        Average_Order_Value=(
            "TotalAmount",
            "mean"
        ),
        First_Purchase_Date=(
            "InvoiceDate",
            "min"
        ),
        Last_Purchase_Date=(
            "InvoiceDate",
            "max"
        )
    ).reset_index()

    # Customer lifetime
    customer_features["Customer_Lifetime_Days"] = (
        customer_features["Last_Purchase_Date"]
        - customer_features["First_Purchase_Date"]
    ).dt.days

    # Avoid zero lifetime
    customer_features["Customer_Lifetime_Days"] = (
        customer_features["Customer_Lifetime_Days"].clip(lower=1)
    )

    # --------------------------------------------------
    # STEP 2: RFM + PURCHASE BEHAVIOR
    # --------------------------------------------------

    # Average basket size
    customer_features["Average_Basket_Size"] = (
        customer_features["Total_Quantity"]
        / customer_features["Frequency"]
    )

    # Recency Score
    customer_features["Recency_Score"] = pd.qcut(
        customer_features["Recency"],
        5,
        labels=[5, 4, 3, 2, 1],
        duplicates="drop"
    )

    # Frequency Score
    customer_features["Frequency_Score"] = pd.qcut(
        customer_features["Frequency"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5],
        duplicates="drop"
    )

    # Monetary Score
    customer_features["Monetary_Score"] = pd.qcut(
        customer_features["Monetary"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5],
        duplicates="drop"
    )

    # Convert scores to integers
    customer_features["Recency_Score"] = (
        customer_features["Recency_Score"].astype(int)
    )

    customer_features["Frequency_Score"] = (
        customer_features["Frequency_Score"].astype(int)
    )

    customer_features["Monetary_Score"] = (
        customer_features["Monetary_Score"].astype(int)
    )

    # Combined RFM Score
    customer_features["RFM_Score"] = (
        customer_features["Recency_Score"]
        + customer_features["Frequency_Score"]
        + customer_features["Monetary_Score"]
    )

    print("\nRFM Features Created!")

    # --------------------------------------------------
    # STEP 3: CLV + PRODUCT + PURCHASE BEHAVIOR
    # --------------------------------------------------

    # Customer Lifetime Value
    customer_features["CLV"] = (
        customer_features["Monetary"]
        / customer_features["Customer_Lifetime_Days"]
    )

    # Number of unique products purchased
    unique_products = (
        df.groupby("Customer_ID")["StockCode"]
        .nunique()
        .reset_index(name="Unique_Products")
    )

    # Number of unique countries
    unique_countries = (
        df.groupby("Customer_ID")["Country"]
        .nunique()
        .reset_index(name="Unique_Countries")
    )

    # Merge unique product information
    customer_features = customer_features.merge(
        unique_products,
        on="Customer_ID",
        how="left"
    )

    # Merge country information
    customer_features = customer_features.merge(
        unique_countries,
        on="Customer_ID",
        how="left"
    )

    # Purchase frequency per month
    customer_features["Purchase_Frequency_Per_Month"] = (
        customer_features["Frequency"]
        / (customer_features["Customer_Lifetime_Days"] / 30)
    )

    print("\nCLV and Purchase Behavior Features Created!")

    return customer_features

def main():

    print("\nLoading Cleaned Dataset...")

    df = pd.read_csv(INPUT_FILE, parse_dates=["InvoiceDate"])

    print(f"Rows Loaded: {len(df)}")

    customer_features = create_features(df)

    # Create processed folder if it does not exist
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Save customer features
    customer_features.to_csv(OUTPUT_FILE, index=False)

    print(f"\nCustomers: {len(customer_features)}")
    print(f"Features: {len(customer_features.columns)}")

    print(f"\nCustomer Features Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()