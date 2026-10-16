"""Checks for reproducible customer value-tier assignments."""
import pandas as pd

from src.utils.segment_tiers import add_value_tiers


def test_value_tiers_use_historical_spend_quantiles() -> None:
    """Spend values are labeled by the documented 50th/80th percentiles."""
    customers = pd.DataFrame({"Customer_ID": [1, 2, 3, 4, 5], "Monetary": [10, 20, 30, 40, 100]})
    tiered = add_value_tiers(customers)
    assert tiered["value_tier"].tolist() == ["Low", "Low", "Low", "Medium", "High"]
    assert tiered["historical_monetary_value"].tolist() == [10, 20, 30, 40, 100]
