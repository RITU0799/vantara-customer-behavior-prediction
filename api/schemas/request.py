from pydantic import BaseModel, ConfigDict, Field

class PredictionRequest(BaseModel):
    """
    Request schema for a single customer.
    """

    model_config = ConfigDict(extra="forbid")
    customer_id: str | None = None
    features: dict[str, float] = Field(min_length=1)


class BatchPredictionRequest(BaseModel):
    """
    Request schema for multiple customers.
    """

    model_config = ConfigDict(extra="forbid")
    customers: list[PredictionRequest] = Field(min_length=1, max_length=1000)
