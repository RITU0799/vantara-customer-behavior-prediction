from pydantic import BaseModel

class ChurnResponse(BaseModel):
    """
    Churn prediction response.
    """

    prediction: int
    probability: float


class CLVResponse(BaseModel):
    """
    Customer Lifetime Value prediction response.
    """

    predicted_value: float


class PredictionResponse(BaseModel):
    """
    Combined churn and CLV response.
    """

    customer_id: str | None = None
    next_purchase_probability: float
    prediction_horizon_days: int
    churn: ChurnResponse
    clv: CLVResponse


class BatchPredictionItem(BaseModel):
    """
    Prediction result for one customer.
    """

    customer_index: int
    customer_id: str | None = None
    next_purchase_probability: float
    prediction_horizon_days: int
    churn: ChurnResponse
    clv: CLVResponse


class BatchPredictionResponse(BaseModel):
    """
    Response for multiple customer predictions.
    """

    count: int
    predictions: list[BatchPredictionItem]
