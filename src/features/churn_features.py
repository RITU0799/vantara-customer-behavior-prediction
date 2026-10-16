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
    / "customer_churn_features.csv"
)

def create_churn_features(df):

    print("\nStarting Churn Feature Engineering...")

    # Last date in the dataset
    max_date = df["InvoiceDate"].max()

    # Observation date
    # We keep the last 90 days as the future period
    observation_date = max_date - pd.Timedelta(days=90)

    # Keep purchases before observation date
    observation_data = df[df["InvoiceDate"] <= observation_date].copy()

    # Future 90-day period
    future_data = df[
        (df["InvoiceDate"] > observation_date)
        & (df["InvoiceDate"] <= max_date)
    ].copy()

    print(f"Observation Date: {observation_date}")
    print(f"Future Period: {observation_date} to {max_date}")

    # Customer features from observation period
    customer_features = observation_data.groupby(
        "Customer_ID"
    ).agg(
        Recency=(
            "InvoiceDate",
            lambda x: (
                observation_date - x.max()
            ).days
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
        )
    ).reset_index()

    # Customers who purchased during future 90 days
    future_customers = (
        future_data["Customer_ID"]
        .dropna()
        .unique()
    )

    # Churn label
    customer_features["Churn"] = (
        ~customer_features["Customer_ID"].isin(
            future_customers
        )
    ).astype(int)

    print("\nChurn Target Created!")

    return customer_features

def main():

    print("\nLoading Cleaned Dataset...")

    df = pd.read_csv(INPUT_FILE, parse_dates=["InvoiceDate"])

    print(f"Rows Loaded: {len(df)}")

    # Calculate transaction value
    df["TotalAmount"] = (df["Quantity"] * df["Price"])

    customer_features = create_churn_features(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    customer_features.to_csv(OUTPUT_FILE, index=False)

    print(f"\nCustomers: {len(customer_features)}")
    print(f"Churned Customers: "f"{customer_features['Churn'].sum()}")
    print(f"Active Customers: "f"{(customer_features['Churn'] == 0).sum()}")

    print(f"\nChurn Features Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()