"""Create a simple monthly revenue trend with a transparent baseline forecast."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def create_monthly_revenue(source: Path | None = None, output: Path | None = None) -> pd.DataFrame:
    """Aggregate transaction revenue by month and append a three-month mean forecast."""
    source = source or ROOT / "data/interim/cleaned_retail_data.csv"
    output = output or ROOT / "data/processed/eda/monthly_revenue_trend.csv"
    data = pd.read_csv(source, usecols=["InvoiceDate", "Quantity", "Price"], parse_dates=["InvoiceDate"])
    data["Revenue"] = data["Quantity"] * data["Price"]
    data = data[data["Quantity"] > 0]
    monthly = data.groupby(data["InvoiceDate"].dt.to_period("M"))["Revenue"].sum().rename("Revenue").reset_index()
    monthly["Month"] = monthly["InvoiceDate"].astype(str)
    monthly = monthly.drop(columns="InvoiceDate")
    baseline = monthly["Revenue"].tail(3).mean()
    last = pd.Period(monthly["Month"].iloc[-1], freq="M")
    forecast = pd.DataFrame({"Month": [str(last + step) for step in range(1, 4)], "Revenue": [float("nan")] * 3, "Forecast": [baseline] * 3})
    monthly["Forecast"] = float("nan")
    result = pd.concat([monthly, forecast], ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


if __name__ == "__main__":
    create_monthly_revenue()
