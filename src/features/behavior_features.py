import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "interim"
    / "combined_retail_data.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "behavior_features.csv"
)

def create_behavior_features(
    df,
    observation_date
):

    print("\nCreating Behavior Features...")

    # --------------------------------------------------
    # Historical transactions only
    # --------------------------------------------------

    df = df[
        df["InvoiceDate"] <= observation_date
    ].copy()

    # Keep customers
    df = df.dropna(
        subset=["Customer_ID"]
    )

    # --------------------------------------------------
    # 1. Return Rate
    # --------------------------------------------------

    df["Is_Return"] = (
        (df["Quantity"] < 0)
        |
        (
            df["Invoice"]
            .astype(str)
            .str.startswith("C")
        )
    ).astype(int)

    return_rate = (
        df.groupby("Customer_ID")[
            "Is_Return"
        ]
        .mean()
        .reset_index(
            name="Return_Rate"
        )
    )

    # --------------------------------------------------
    # 2. Price Sensitivity Proxy
    # --------------------------------------------------

    purchase_data = df[
        df["Quantity"] > 0
    ].copy()

    price_features = (
        purchase_data
        .groupby("Customer_ID")
        .agg(
            Average_Price=("Price", "mean"),
            Minimum_Price=("Price", "min"),
            Maximum_Price=("Price", "max")
        )
        .reset_index()
    )

    price_features[
        "Price_Range"
    ] = (
        price_features["Maximum_Price"]
        - price_features["Minimum_Price"]
    )

    price_features[
        "Price_Sensitivity_Proxy"
    ] = (
        1
        / (
            1
            + price_features["Average_Price"]
        )
    )

    price_features = price_features[
        [
            "Customer_ID",
            "Price_Sensitivity_Proxy"
        ]
    ]

    # --------------------------------------------------
    # Combine Features
    # --------------------------------------------------

    features = return_rate.merge(
        price_features,
        on="Customer_ID",
        how="left"
    )

    features = features.fillna(0)

    return features

if __name__ == "__main__":

    print("\nLoading Combined Data...")

    df = pd.read_csv(INPUT_FILE, parse_dates=["InvoiceDate"])

    print("Rows Loaded:", len(df))

    max_date = df["InvoiceDate"].max()

    observation_date = (max_date - pd.Timedelta(days=90))

    print("\nObservation Date:", observation_date)

    features = create_behavior_features(df, observation_date)

    print(
        "Customers:",
        len(features)
    )

    print(
        "Features:",
        len(features.columns)
    )

    print(
        "Average Return Rate:",
        round(
            features["Return_Rate"].mean(),
            4
        )
    )

    features.to_csv(OUTPUT_FILE, index=False)

    print("\nBehavior Features Created Successfully!")

    print("\nSaved:", OUTPUT_FILE)