from fastapi import APIRouter, Query

from api.database import list_segments
from api.database import list_predictions
from src.recommendations import recommend_products

router = APIRouter(
    prefix="/customer",
    tags=["Customer"]
)


@router.get("/status")
def customer_status():
    """
    Check customer service availability.
    """

    return {
        "status": "available",
        "service": "customer intelligence"
    }


@router.get("/segments")
def customer_segments(
    segment: str | None = Query(default=None),
    country: str | None = Query(default=None),
    value_tier: str | None = Query(default=None, pattern="^(Low|Medium|High)$"),
):
    """List persisted assignments with segment, country, and value-tier filters."""
    rows = list_segments(segment=segment, country=country, value_tier=value_tier)
    return {"count": len(rows), "customers": rows}


@router.get("/leaderboard")
def churn_leaderboard(limit: int = Query(default=100, ge=1, le=500)):
    """Rank persisted predictions by churn risk, then predicted value."""
    rows = list_predictions(limit=limit)
    return {"count": len(rows), "customers": rows}


@router.get("/recommendations/{customer_id}")
def product_recommendations(customer_id: str, limit: int = Query(default=5, ge=1, le=20)):
    """Return simple co-purchase recommendations for a known customer."""
    recommendations = recommend_products(customer_id, limit=limit)
    return {"customer_id": customer_id, "recommendations": recommendations}
