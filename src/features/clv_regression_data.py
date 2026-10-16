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

ADVANCED_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "advanced_customer_features.csv"
)

BEHAVIOR_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "behavior_features.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "clv_regression_data.csv"
)


def create_clv_regression_dataset():
    """Create a point-in-time dataset for future 90-day CLV regression."""

    print("\nLoading Cleaned Retail Data...")

    data = pd.read_csv(INPUT_FILE, parse_dates=["InvoiceDate"])

    print(f"Input Shape : {data.shape}")

    # Observation and future periods
    max_date = data["InvoiceDate"].max()
    observation_date = max_date - pd.Timedelta(days=90)

    print(f"\nMaximum Transaction Date : {max_date}")
    print(f"Observation Date : {observation_date}")

    historical_data = data[
        (data["InvoiceDate"] <= observation_date)
        & (data["Quantity"] > 0)
    ].copy()

    future_data = data[
        (data["InvoiceDate"] > observation_date)
        & (data["InvoiceDate"] <= max_date)
        & (data["Quantity"] > 0)
    ].copy()

    print(f"\nHistorical Data Shape : {historical_data.shape}")
    print(f"Future Data Shape : {future_data.shape}")

    # Historical customer features
    print("\nCreating Historical Customer Features...")

    customer_features = (
        historical_data
        .groupby("Customer_ID")
        .agg(
            Recency=(
                "InvoiceDate",
                lambda x: (observation_date - x.max()).days
            ),
            Frequency=("Invoice", "nunique"),
            Monetary=("TotalAmount", "sum"),
            Average_Order_Value=("TotalAmount", "mean"),
            Total_Quantity=("Quantity", "sum"),
            Unique_Products=("StockCode", "nunique"),
            Customer_Lifetime_Days=(
                "InvoiceDate",
                lambda x: max((x.max() - x.min()).days, 1)
            )
        )
        .reset_index()
    )

    # Historical CLV
    customer_features["CLV"] = (
        customer_features["Monetary"]
        / customer_features["Customer_Lifetime_Days"]
    )

    # Purchase frequency per month
    customer_features["Purchase_Frequency_Per_Month"] = (
        customer_features["Frequency"]
        / (
            customer_features["Customer_Lifetime_Days"] / 30
        ).clip(lower=1)
    )

    # Average basket size
    customer_features["Average_Basket_Size"] = (
        customer_features["Total_Quantity"]
        / customer_features["Frequency"].clip(lower=1)
    )

    # Merge existing historical behavioral features
    print("\nLoading Existing Behavioral Features...")

    advanced_features = pd.read_csv(ADVANCED_FILE)
    behavior_features = pd.read_csv(BEHAVIOR_FILE)

    print(f"Advanced Features Shape : {advanced_features.shape}")
    print(f"Behavior Features Shape : {behavior_features.shape}")

    # Keep only customer-level behavioral features
    advanced_columns = [
        "Customer_ID",
        "Purchase_Frequency_Trend",
        "Purchase_Gap_Variance",
        "Seasonal_Purchase_Concentration",
        "Product_Category_Affinity",
        "Engagement_Score"
    ]

    behavior_columns = [
        "Customer_ID",
        "Return_Rate",
        "Price_Sensitivity_Proxy"
    ]

    advanced_features = advanced_features[advanced_columns]
    behavior_features = behavior_features[behavior_columns]

    # Merge behavioral features
    customer_features = customer_features.merge(
        advanced_features,
        on="Customer_ID",
        how="left"
    )

    customer_features = customer_features.merge(
        behavior_features,
        on="Customer_ID",
        how="left"
    )

    print(
        f"Features After Behavioral Merge : "
        f"{customer_features.shape}"
    )

    # Future 90-day value
    print("\nCreating Future 90-Day CLV Target...")

    future_value = (
        future_data
        .groupby("Customer_ID")["TotalAmount"]
        .sum()
        .reset_index()
    )

    future_value = future_value.rename(
        columns={
            "TotalAmount": "Future_90D_CLV"
        }
    )

    # Merge historical features with future target
    clv_data = customer_features.merge(
        future_value,
        on="Customer_ID",
        how="left"
    )

    # Customers with no future purchase have zero future value
    clv_data["Future_90D_CLV"] = (
        clv_data["Future_90D_CLV"]
        .fillna(0)
    )

    # Fill missing historical behavioral values
    behavioral_columns = [
        "Purchase_Frequency_Trend",
        "Purchase_Gap_Variance",
        "Seasonal_Purchase_Concentration",
        "Product_Category_Affinity",
        "Engagement_Score",
        "Return_Rate",
        "Price_Sensitivity_Proxy"
    ]

    clv_data[behavioral_columns] = (
        clv_data[behavioral_columns]
        .fillna(0)
    )

    # Validation
    print("\nValidating CLV Regression Dataset...")

    print(f"Final Dataset Shape : {clv_data.shape}")

    print(
        f"Unique Customers : "
        f"{clv_data['Customer_ID'].nunique()}"
    )

    print(
        f"Duplicate Customer IDs : "
        f"{clv_data['Customer_ID'].duplicated().sum()}"
    )

    print(
        f"Missing Values : "
        f"{clv_data.isnull().sum().sum()}"
    )

    print(
        f"Future CLV Minimum : "
        f"{clv_data['Future_90D_CLV'].min():.2f}"
    )

    print(
        f"Future CLV Maximum : "
        f"{clv_data['Future_90D_CLV'].max():.2f}"
    )

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    clv_data.to_csv(OUTPUT_FILE, index=False)

    print("\nCLV Regression Dataset Created Successfully!")
    
    print(f"Saved File : {OUTPUT_FILE}")

if __name__ == "__main__":
    create_clv_regression_dataset()