from pathlib import Path

import pandas as pd
import numpy as np


# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "interim"
    / "cleaned_retail_data.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "lstm_data"
)

OUTPUT_FILE = OUTPUT_DIR / "customer_sequences.npz"


# Settings
SEQUENCE_LENGTH = 5


# Load data
print("\nLoading Cleaned Transaction Data...")

df = pd.read_csv(INPUT_FILE)
print(f"Original Rows : {len(df):,}")


# Prepare dates
print("\nPreparing Transaction Dates...")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

df = df.dropna(subset=["Customer_ID", "Invoice", "InvoiceDate"])


# Create purchase events
print("\nCreating Customer Purchase Events...")

purchase_events = (
    df.groupby(
        ["Customer_ID", "Invoice"],
        as_index=False
    )
    .agg(
        InvoiceDate=("InvoiceDate", "min"),
        TotalAmount=("TotalAmount", "sum"),
        Quantity=("Quantity", "sum"),
        Product_Code=("StockCode", "nunique")
    )
)

print(f"Purchase Events : {len(purchase_events):,}")


# Sort purchase events
print("\nSorting Purchase Events...")

purchase_events = purchase_events.sort_values(
    ["Customer_ID", "InvoiceDate"]
).reset_index(drop=True)


# Calculate purchase gap
print("\nCalculating Purchase Gaps...")

purchase_events["Purchase_Gap_Days"] = (
    purchase_events
    .groupby("Customer_ID")["InvoiceDate"]
    .diff()
    .dt.total_seconds()
    / 86400
)

purchase_events["Purchase_Gap_Days"] = (
    purchase_events["Purchase_Gap_Days"]
    .fillna(0)
)


# Calculate next purchase date
print("\nCalculating Next Purchase...")

purchase_events["Next_Purchase_Date"] = (
    purchase_events
    .groupby("Customer_ID")["InvoiceDate"]
    .shift(-1)
)


# Create target
purchase_events["Days_To_Next_Purchase"] = (
    purchase_events["Next_Purchase_Date"]
    - purchase_events["InvoiceDate"]
).dt.total_seconds() / 86400


# Create sequences
print("\nCreating Customer Sequences...")

features = [
    "TotalAmount",
    "Quantity",
    "Purchase_Gap_Days",
    "Product_Code",
]

X_sequences = []
y_targets = []
customer_ids = []


for customer_id, customer_data in purchase_events.groupby("Customer_ID"):

    customer_data = customer_data.reset_index(drop=True)

    if len(customer_data) <= SEQUENCE_LENGTH:
        continue

    for i in range(SEQUENCE_LENGTH, len(customer_data)):

        sequence = customer_data.iloc[
            i - SEQUENCE_LENGTH:i
        ][features].values

        target = customer_data.loc[
            i, "Days_To_Next_Purchase"
        ]

        # Skip invalid targets
        if pd.isna(target) or target < 1:
            continue

        X_sequences.append(sequence)
        y_targets.append(target)
        customer_ids.append(customer_id)


# Convert to arrays
X = np.array(
    X_sequences,
    dtype=np.float32
)

y = np.array(
    y_targets,
    dtype=np.float32
)

customer_ids = np.array(
    customer_ids
)


# Clean invalid values
X = np.nan_to_num(
    X,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

y = np.nan_to_num(
    y,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# Save
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    OUTPUT_FILE,
    X=X,
    y=y,
    customer_ids=customer_ids
)


# Summary
print("\nLSTM Sequence Preparation Completed!")
print("-------------------------------------")
print(f"Original Rows       : {len(df):,}")
print(f"Purchase Events     : {len(purchase_events):,}")
print(f"Customers           : " f"{purchase_events['Customer_ID'].nunique():,}")
print(f"Sequence Length     : {SEQUENCE_LENGTH}")
print(f"Number of Sequences : {len(X):,}")
print(f"Feature Count       : {X.shape[2]}")
print(f"Sequence Shape      : {X.shape}")
print(f"Target Shape        : {y.shape}")
print(f"Target Mean (days)  : {y.mean():.2f}")
print(f"Target Median (days): {np.median(y):.2f}")
print(f"Target Min (days)   : {y.min():.2f}")
print(f"Target Max (days)   : {y.max():.2f}")

print(f"\nSaved File        : {OUTPUT_FILE}")