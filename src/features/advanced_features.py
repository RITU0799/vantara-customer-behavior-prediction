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
    / "advanced_customer_features.csv"
)

def calculate_trend(group):

    if len(group) < 2:
        return 0

    group = group.sort_values("InvoiceDate")

    months = (
        group["InvoiceDate"] - group["InvoiceDate"].min()
    ).dt.days / 30

    purchases = group["Invoice"].nunique()

    if months.max() == 0:
        return 0

    return purchases / (months.max() + 1)

def create_advanced_features(df, observation_date):

    # Only historical transactions
    df = df[
        df["InvoiceDate"] <= observation_date
    ].copy()

    # Only completed purchases
    df = df[df["Quantity"] > 0].copy()

    df["TotalAmount"] = (
        df["Quantity"] * df["Price"]
    )

    # --------------------------------------------------
    # 1. Purchase Frequency Trend
    # --------------------------------------------------

    trend = (
        df.groupby("Customer_ID")
        .apply(
            calculate_trend,
            include_groups=False
        )
        .reset_index(
            name="Purchase_Frequency_Trend"
        )
    )

    # --------------------------------------------------
    # 2. Purchase Gap Variance
    # --------------------------------------------------

    customer_dates = (
        df.groupby(
            ["Customer_ID", "Invoice"]
        )["InvoiceDate"]
        .min()
        .reset_index()
    )

    customer_dates = customer_dates.sort_values(
        ["Customer_ID", "InvoiceDate"]
    )

    customer_dates["Gap_Days"] = (
        customer_dates
        .groupby("Customer_ID")["InvoiceDate"]
        .diff()
        .dt.days
    )

    gap_variance = (
        customer_dates
        .groupby("Customer_ID")["Gap_Days"]
        .var()
        .fillna(0)
        .reset_index(
            name="Purchase_Gap_Variance"
        )
    )

    # --------------------------------------------------
    # 3. Seasonal Purchase Concentration
    # --------------------------------------------------

    df["Month"] = df["InvoiceDate"].dt.month

    monthly_orders = (
        df.groupby(
            ["Customer_ID", "Month"]
        )["Invoice"]
        .nunique()
        .reset_index(
            name="Monthly_Orders"
        )
    )

    seasonal = (
        monthly_orders
        .groupby("Customer_ID")["Monthly_Orders"]
        .sum()
        .reset_index(
            name="Total_Monthly_Orders"
        )
    )

    max_monthly = (
        monthly_orders
        .groupby("Customer_ID")["Monthly_Orders"]
        .max()
        .reset_index(
            name="Max_Monthly_Orders"
        )
    )

    seasonal = seasonal.merge(
        max_monthly,
        on="Customer_ID"
    )

    seasonal[
        "Seasonal_Purchase_Concentration"
    ] = (
        seasonal["Max_Monthly_Orders"]
        / seasonal["Total_Monthly_Orders"]
    )

    seasonal = seasonal[
        [
            "Customer_ID",
            "Seasonal_Purchase_Concentration"
        ]
    ]

    # --------------------------------------------------
    # 4. Product Category Affinity
    # --------------------------------------------------

    df["Product_Category"] = (
        df["StockCode"]
        .astype(str)
        .str.extract(
            r"([A-Za-z]+)",
            expand=False
        )
        .fillna("Numeric")
    )

    category_orders = (
        df.groupby(
            [
                "Customer_ID",
                "Product_Category"
            ]
        )["Invoice"]
        .nunique()
        .reset_index(
            name="Category_Orders"
        )
    )

    total_orders = (
        category_orders
        .groupby("Customer_ID")[
            "Category_Orders"
        ]
        .sum()
        .reset_index(
            name="Total_Category_Orders"
        )
    )

    category_orders = category_orders.merge(
        total_orders,
        on="Customer_ID"
    )

    category_orders[
        "Category_Affinity"
    ] = (
        category_orders["Category_Orders"]
        / category_orders["Total_Category_Orders"]
    )

    category_affinity = (
        category_orders
        .groupby("Customer_ID")[
            "Category_Affinity"
        ]
        .max()
        .reset_index(
            name="Product_Category_Affinity"
        )
    )

    # --------------------------------------------------
    # 5. Engagement Score
    # --------------------------------------------------

    engagement = (
        df.groupby("Customer_ID")
        .agg(
            Recency=(
                "InvoiceDate",
                lambda x: (
                    observation_date - x.max()
                ).days
            ),
            Frequency=("Invoice", "nunique"),
            Monetary=("TotalAmount", "sum")
        )
        .reset_index()
    )

    engagement["Recency_Score"] = (
        1 / (1 + engagement["Recency"])
    )

    engagement["Frequency_Score"] = (
        engagement["Frequency"]
        / engagement["Frequency"].max()
    )

    engagement["Monetary_Score"] = (
        engagement["Monetary"]
        / engagement["Monetary"].max()
    )

    engagement["Engagement_Score"] = (
        engagement["Recency_Score"]
        + engagement["Frequency_Score"]
        + engagement["Monetary_Score"]
    ) / 3

    engagement = engagement[
        [
            "Customer_ID",
            "Engagement_Score"
        ]
    ]

    # --------------------------------------------------
    # Combine Features
    # --------------------------------------------------

    features = trend.merge(
        gap_variance,
        on="Customer_ID",
        how="left"
    )

    features = features.merge(
        seasonal,
        on="Customer_ID",
        how="left"
    )

    features = features.merge(
        category_affinity,
        on="Customer_ID",
        how="left"
    )

    features = features.merge(
        engagement,
        on="Customer_ID",
        how="left"
    )

    features = features.fillna(0)

    return features

if __name__ == "__main__":

    print("\nLoading Cleaned Data...")

    df = pd.read_csv(INPUT_FILE, parse_dates=["InvoiceDate"])

    print("Rows Loaded:", len(df))

    max_date = df["InvoiceDate"].max()

    observation_date = (max_date - pd.Timedelta(days=90))

    print("Observation Date:", observation_date)

    features = create_advanced_features(df,observation_date)

    print("Customers:", len(features))

    print("Features:", len(features.columns))

    features.to_csv(OUTPUT_FILE, index=False)

    print("\nAdvanced Features Created Successfully!")

    print("\nSaved:", OUTPUT_FILE)