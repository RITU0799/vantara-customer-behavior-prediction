"""Integration checks for the public FastAPI contract."""
from fastapi.testclient import TestClient

from api.main import app


def test_health_and_metadata_are_available() -> None:
    """Health and metadata routes expose a loaded model service."""
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        metadata = client.get("/metadata")
        assert metadata.status_code == 200
        assert metadata.json()["churn_features"]
        assert metadata.json()["clv_features"]
        assert metadata.json()["next_purchase_horizon_days"] == 90
        assert metadata.json()["churn_probability_calibrated"] is True


def test_prediction_rejects_missing_features() -> None:
    """Malformed customer input receives a clear validation error."""
    with TestClient(app) as client:
        response = client.post("/prediction/predict", json={"customer_id": "demo", "features": {"unknown": 1.0}})
    assert response.status_code == 422
    assert "missing_churn_features" in str(response.json())


def test_segment_endpoint_filters_by_value_tier() -> None:
    """The segment view exposes value tiers and accepts a tier filter."""
    with TestClient(app) as client:
        response = client.get("/customer/segments", params={"value_tier": "High"})
    assert response.status_code == 200
    customers = response.json()["customers"]
    assert customers
    assert all(customer["value_tier"] == "High" for customer in customers)
    assert all("historical_monetary_value" in customer for customer in customers)
