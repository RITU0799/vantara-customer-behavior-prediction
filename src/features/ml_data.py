import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "interim"
    / "cleaned_retail_data.csv"
)

ADVANCED_FEATURES_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "advanced_customer_features.csv"
)

BEHAVIOR_FEATURES_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "behavior_features.csv"
)

ENCODING_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_encoding_features.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_ready_data.csv"
)


def create_ml_dataset():
    """Create the final customer-level dataset for machine learning."""

    print("\nLoading Cleaned Retail Data...")

    data = pd.read_csv(INPUT_FILE, parse_dates=["InvoiceDate"])
    print(f"Input Shape : {data.shape}")

    # Observation period
    max_date = data["InvoiceDate"].max()

    observation_date = max_date - pd.Timedelta(days=90)

    print(f"\nMaximum Transaction Date : {max_date}")
    print(f"Observation Date : {observation_date}")

    # Historical data used for feature creation
    observation_data = data[
        (data["InvoiceDate"] <= observation_date)
        & (data["Quantity"] > 0)
    ].copy()

    # Future data used only for churn target
    future_data = data[
        (data["InvoiceDate"] > observation_date)
        & (data["InvoiceDate"] <= max_date)
        & (data["Quantity"] > 0)
    ].copy()

    print(f"Observation Data Shape : {observation_data.shape}")

    print(f"Future Data Shape : {future_data.shape}")

    # Customer-level historical features
    print("\nCreating Customer-Level Features...")

    customer_features = (
        observation_data
        .groupby("Customer_ID")
        .agg(
            Recency=(
                "InvoiceDate",
                lambda x: (
                    observation_date - x.max()
                ).days
            ),
            Frequency=("Invoice", "nunique"),
            Monetary=("TotalAmount", "sum"),
            Total_Quantity=("Quantity", "sum"),
            Average_Order_Value=("TotalAmount", "mean"),
            First_Purchase=("InvoiceDate", "min"),
            Last_Purchase=("InvoiceDate", "max"),
            Customer_Lifetime_Days=(
                "InvoiceDate",
                lambda x: max(
                    (x.max() - x.min()).days,
                    1
                )
            ),
            Average_Basket_Size=(
                "Quantity",
                "mean"
            )
        )
        .reset_index()
    )


    # RFM Scores
    print("\nCreating RFM Scores...")

    customer_features["Recency_Score"] = pd.qcut(
        customer_features["Recency"],
        5,
        labels=False,
        duplicates="drop"
    )

    customer_features["Frequency_Score"] = pd.qcut(
        customer_features["Frequency"].rank(method="first"),
        5,
        labels=False,
        duplicates="drop"
    )

    customer_features["Monetary_Score"] = pd.qcut(
        customer_features["Monetary"].rank(method="first"),
        5,
        labels=False,
        duplicates="drop"
    )


    # Convert scores to 1-5
    customer_features["Recency_Score"] = (
        5 - customer_features["Recency_Score"]
    )

    customer_features["Frequency_Score"] = (
        customer_features["Frequency_Score"] + 1
    )

    customer_features["Monetary_Score"] = (
        customer_features["Monetary_Score"] + 1
    )

    customer_features["RFM_Score"] = (
        customer_features["Recency_Score"]
        + customer_features["Frequency_Score"]
        + customer_features["Monetary_Score"]
    )


    # CLV
    print("\nCreating Historical CLV...")

    customer_features["CLV"] = (
        customer_features["Monetary"]
        / customer_features["Customer_Lifetime_Days"]
    )

    # Additional customer features
    print("\nCreating Additional Customer Features...")

    customer_features["Unique_Products"] = (
        observation_data
        .groupby("Customer_ID")["StockCode"]
        .nunique()
        .values
    )

    customer_features["Unique_Countries"] = (
        observation_data
        .groupby("Customer_ID")["Country"]
        .nunique()
        .values
    )

    customer_features["Purchase_Frequency_Per_Month"] = (
        customer_features["Frequency"]
        / (
            customer_features["Customer_Lifetime_Days"] / 30
        ).clip(lower=1)
    )

    
    # Churn Target
    print("\nCreating Churn Target...")

    future_customers = set(
        future_data["Customer_ID"].unique()
    )

    customer_features["Churn"] = (
        ~customer_features["Customer_ID"].isin(
            future_customers
        )
    ).astype(int)

    print(
        f"Churned Customers : "
        f"{customer_features['Churn'].sum()}"
    )

    print(
        f"Active Customers : "
        f"{(customer_features['Churn'] == 0).sum()}"
    )


    # Remove date columns
    customer_features = customer_features.drop(
        columns=[
            "First_Purchase",
            "Last_Purchase"
        ]
    )


    # Load Advanced Features
    print("\nLoading Advanced Customer Features...")

    advanced_features = pd.read_csv(ADVANCED_FEATURES_FILE)

    print(
        f"Advanced Features Shape : "
        f"{advanced_features.shape}"
    )

    customer_features = customer_features.merge(
        advanced_features,
        on="Customer_ID",
        how="left"
    )

    print(
        f"Shape After Advanced Features : "
        f"{customer_features.shape}"
    )


    # Load Behavior Features
    print("\nLoading Behavior Features...")

    behavior_features = pd.read_csv(BEHAVIOR_FEATURES_FILE)

    print(
        f"Behavior Features Shape : "
        f"{behavior_features.shape}"
    )

    customer_features = customer_features.merge(
        behavior_features,
        on="Customer_ID",
        how="left"
    )

    print(
        f"Shape After Behavior Features : "
        f"{customer_features.shape}"
    )


    # Load Country + StockCode Encoding
    print("\nLoading Categorical Encoding Features...")

    encoding_features = pd.read_csv(ENCODING_FILE)

    print(
        f"Encoding Features Shape : "
        f"{encoding_features.shape}"
    )

    print("\nMerging Categorical Encoding Features...")

    customer_features = customer_features.merge(
        encoding_features,
        on="Customer_ID",
        how="left"
    )

    print(
        f"Shape After Encoding Merge : "
        f"{customer_features.shape}"
    )


    # Handle Missing Values
    print("\nHandling Missing Values...")

    customer_features = customer_features.fillna(0)

    
    # Final Validation
    print("\nFinal Dataset Validation...")

    print(
        f"Final Dataset Shape : "
        f"{customer_features.shape}"
    )

    print(
        f"Duplicate Customer IDs : "
        f"{customer_features['Customer_ID'].duplicated().sum()}"
    )

    print(
        f"Missing Values : "
        f"{customer_features.isnull().sum().sum()}"
    )


    # Save ML Dataset
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    customer_features.to_csv(OUTPUT_FILE, index=False)

    print("\nML Dataset Created Successfully!")
    print(f"Final Shape : {customer_features.shape}")
    print(f"Saved File : {OUTPUT_FILE}")

if __name__ == "__main__":
    create_ml_dataset()