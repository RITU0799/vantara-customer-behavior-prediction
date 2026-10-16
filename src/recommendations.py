"""Simple historical basket co-occurrence recommendations."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _load_transactions() -> pd.DataFrame:
    """Load positive purchase lines available before the churn observation cutoff."""
    path = ROOT / "data/interim/cleaned_retail_data.csv"
    data = pd.read_csv(path, usecols=["Invoice", "InvoiceDate", "Customer_ID", "StockCode", "Description", "Quantity"], parse_dates=["InvoiceDate"])
    cutoff = data["InvoiceDate"].max() - pd.Timedelta(days=90)
    return data.loc[(data["InvoiceDate"] <= cutoff) & (data["Quantity"] > 0) & data["Customer_ID"].notna()].copy()


def recommend_products(customer_id: str, limit: int = 5) -> list[dict[str, object]]:
    """Recommend unseen products that frequently co-occur with a customer's items."""
    data = _load_transactions()
    ids = data["Customer_ID"].astype(str).str.replace(r"\.0$", "", regex=True)
    customer = data.loc[ids == str(customer_id).replace(".0", "")]
    if customer.empty:
        return []
    bought = set(customer["StockCode"].astype(str))
    seeds = customer.groupby("StockCode")["Invoice"].nunique().nlargest(5).index.astype(str).tolist()
    baskets = data.assign(StockCode=data["StockCode"].astype(str))
    seed_invoices = baskets.loc[baskets["StockCode"].isin(seeds), "Invoice"].unique()
    candidates = baskets.loc[baskets["Invoice"].isin(seed_invoices) & ~baskets["StockCode"].isin(bought)]
    if candidates.empty:
        candidates = baskets.loc[~baskets["StockCode"].isin(bought)]
    result = candidates.groupby(["StockCode", "Description"], dropna=False)["Invoice"].nunique().sort_values(ascending=False).head(limit).reset_index(name="co_purchase_orders")
    return result.where(pd.notna(result), None).to_dict(orient="records")
