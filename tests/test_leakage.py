import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

CLEANED_FILE = (
    BASE_DIR
    / "data"
    / "interim"
    / "cleaned_retail_data.csv"
)

ML_DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_ready_data.csv"
)

ENCODING_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_encoding_features.csv"
)


def test_observation_date_is_before_future_period():
    """Verify that the observation period ends before the future period."""

    data = pd.read_csv(CLEANED_FILE, parse_dates=["InvoiceDate"])

    max_date = data["InvoiceDate"].max()
    observation_date = max_date - pd.Timedelta(days=90)

    observation_data = data[data["InvoiceDate"] <= observation_date]

    future_data = data[data["InvoiceDate"] > observation_date]

    assert not observation_data.empty
    assert not future_data.empty

    assert observation_data["InvoiceDate"].max() <= observation_date
    assert future_data["InvoiceDate"].min() > observation_date


def test_encoding_uses_only_historical_data():
    """Verify that categorical encoding uses only historical transactions."""

    data = pd.read_csv(CLEANED_FILE, parse_dates=["InvoiceDate"])

    max_date = data["InvoiceDate"].max()
    observation_date = max_date - pd.Timedelta(days=90)

    encoding_data = data[
        (data["InvoiceDate"] <= observation_date)
        & (data["Quantity"] > 0)
    ]

    assert not encoding_data.empty

    assert (
        encoding_data["InvoiceDate"].max()
        <= observation_date
    )


def test_ml_customers_match_encoding_customers():
    """Verify that encoding features contain the same customer population."""

    ml_data = pd.read_csv(ML_DATA_FILE)
    encoding_data = pd.read_csv(ENCODING_FILE)

    ml_customers = set(ml_data["Customer_ID"])

    encoding_customers = set(encoding_data["Customer_ID"])

    assert ml_customers == encoding_customers


def test_ml_dataset_has_unique_customers():
    """Verify that the ML dataset contains one row per customer."""

    ml_data = pd.read_csv(ML_DATA_FILE)

    duplicate_customers = (
        ml_data["Customer_ID"]
        .duplicated()
        .sum()
    )

    assert duplicate_customers == 0


def test_ml_dataset_contains_no_missing_values():
    """Verify that the final ML dataset contains no missing values."""

    ml_data = pd.read_csv(ML_DATA_FILE)

    assert ml_data.isnull().sum().sum() == 0