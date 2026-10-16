import pandas as pd
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parents[2]

# File paths
INPUT_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "online_retail_II.xlsx"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "interim"
    / "combined_retail_data.csv"
)

def prepare_sheet(df):

    # Standardize column names
    df.columns = (df.columns.str.strip().str.replace(" ", "_"))

    # Standardize data types
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["Description"] = df["Description"].astype("string")

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df["Customer_ID"] = pd.to_numeric(df["Customer_ID"], errors="coerce")

    df["Country"] = df["Country"].astype("string")

    return df

def load_data():

    print("\nLoading Dataset...")

    # Load both sheets
    df_2009 = pd.read_excel(INPUT_FILE, sheet_name="Year 2009-2010")

    df_2010 = pd.read_excel(INPUT_FILE, sheet_name="Year 2010-2011")

    # Standardize each sheet before combining
    df_2009 = prepare_sheet(df_2009)
    df_2010 = prepare_sheet(df_2010)

    # Combine both years
    df = pd.concat([df_2009, df_2010], ignore_index=True)

    # Sort chronologically
    df = (df.sort_values("InvoiceDate").reset_index(drop=True))

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDataset Loaded Successfully!")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("\nSaved to:", OUTPUT_FILE)

if __name__ == "__main__":
    load_data()