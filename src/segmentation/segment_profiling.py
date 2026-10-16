from pathlib import Path

import pandas as pd
import numpy as np


# PROJECT PATH
BASE_DIR = Path(__file__).resolve().parents[2]


# FILE PATHS
CUSTOMER_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_ready_data.csv"
)

SEGMENT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "segmentation"
    / "kmeans_customer_segments.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "segmentation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "segment_profiles.csv"
)


# LOAD DATA
print("\nLoading Customer Data...")

customer_df = pd.read_csv(CUSTOMER_FILE)
segment_df = pd.read_csv(SEGMENT_FILE)

print(f"Customer Data Shape : {customer_df.shape}")
print(f"Segment Data Shape  : {segment_df.shape}")


# MERGE CUSTOMER DATA WITH SEGMENTS
print("\nMerging Customer Data with Segments...")

df = customer_df.merge(
    segment_df,
    on="Customer_ID",
    how="inner"
)

print(f"Merged Data Shape : {df.shape}")


# CHECK SEGMENTS
print("\nSegment Distribution")
print("-------------------------------------")

segment_counts = (df["Segment"].value_counts().sort_index())

for segment, count in segment_counts.items():

    percentage = (count / len(df)) * 100

    print(
        f"Segment {segment}: "
        f"{count} customers "
        f"({percentage:.2f}%)"
    )


# SELECT NUMERIC FEATURES
exclude_columns = ["Customer_ID", "Churn", "Segment"]

numeric_features = [
    column
    for column in df.select_dtypes(
        include=np.number
    ).columns
    if column not in exclude_columns
]


# CREATE SEGMENT PROFILE
print("\nCreating Segment Profiles...")

profile = (
    df.groupby("Segment")[numeric_features]
    .mean()
    .round(2)
)


# ADD CUSTOMER COUNT
customer_count = (
    df.groupby("Segment")
    .size()
    .rename("Customer_Count")
)

profile = profile.join(customer_count)


# ADD CUSTOMER PERCENTAGE
profile["Customer_Percentage"] = (
    profile["Customer_Count"]
    / len(df)
    * 100
).round(2)


# ADD BUSINESS-READABLE SEGMENT PROFILES
def build_business_profile(row):
    if row["Recency"] >= df["Recency"].quantile(0.75):
        return "Lapsed / At-Risk Customers"
    if row["Frequency"] >= df["Frequency"].quantile(0.75) and row["Monetary"] >= df["Monetary"].quantile(0.75):
        return "High-Value Frequent Customers"
    if row["Monetary"] >= df["Monetary"].quantile(0.75):
        return "High-Value Customers"
    if row["Frequency"] >= df["Frequency"].quantile(0.75):
        return "Frequent Customers"
    return "Regular Customers"

profile["Business_Profile"] = profile.apply(build_business_profile, axis=1)


# REORDER COUNT COLUMNS
count_columns = [
    "Business_Profile",
    "Customer_Count",
    "Customer_Percentage"
]

other_columns = [
    column
    for column in profile.columns
    if column not in count_columns
]

profile = profile[count_columns + other_columns]


# DISPLAY PROFILE
print("\nSegment Profiles")
print("-------------------------------------")
print(profile.to_string())


# SAVE PROFILE
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
profile.to_csv(OUTPUT_FILE)


# FINAL MESSAGE
print("\nSegment Profiling Completed Successfully!")

print(f"Output File : {OUTPUT_FILE}")