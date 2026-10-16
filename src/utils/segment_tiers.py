"""Assign reusable business tiers from historical customer spend."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_value_tiers(customers: pd.DataFrame) -> pd.DataFrame:
    """Add Low/Medium/High spend tiers at the 50th and 80th percentiles."""
    if "Monetary" not in customers.columns:
        raise ValueError("Customer table must include the Monetary feature")
    result = customers.copy()
    monetary = pd.to_numeric(result["Monetary"], errors="coerce").fillna(0)
    q50, q80 = monetary.quantile([0.50, 0.80]).tolist()
    result["value_tier"] = np.select(
        [monetary <= q50, monetary <= q80],
        ["Low", "Medium"],
        default="High",
    )
    result["historical_monetary_value"] = monetary.astype(float)
    return result
