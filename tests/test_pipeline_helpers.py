"""Unit checks for reusable project configuration and risk labels."""
from src.utils.config import load_config


def test_prd_targets_are_present_in_configuration() -> None:
    """The operational target thresholds remain visible in YAML config."""
    targets = load_config()["evaluation_targets"]
    assert targets["churn_roc_auc"] == 0.80
    assert targets["churn_recall"] == 0.70
    assert targets["clv_r2"] == 0.60
