"""Persistence for scored customers and precomputed segment assignments."""
from __future__ import annotations

import csv
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Float, Integer, String, create_engine, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from api.settings import DATABASE_URL

LOG = logging.getLogger(__name__)
_engine_kwargs: dict[str, Any] = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    """Base class for persisted records."""


class Prediction(Base):
    """One persisted churn and CLV score per scoring request."""
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[str | None] = mapped_column(String(128), index=True)
    churn_probability: Mapped[float] = mapped_column(Float)
    churn_prediction: Mapped[int] = mapped_column(Integer)
    predicted_clv: Mapped[float] = mapped_column(Float)
    features: Mapped[dict[str, Any]] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))


class SegmentAssignment(Base):
    """Latest known customer segment assignment."""
    __tablename__ = "segment_assignments"
    customer_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    segment: Mapped[str] = mapped_column(String(64), index=True)
    business_profile: Mapped[str | None] = mapped_column(String(128))


def initialize_database() -> None:
    """Create persistence tables and import available segmentation output."""
    Base.metadata.create_all(engine)
    segment_file = Path(__file__).resolve().parents[1] / "data/processed/segmentation/kmeans_customer_segments.csv"
    if not segment_file.exists():
        LOG.info("No saved segment assignment file found at %s", segment_file)
        return
    profile_file = segment_file.parent / "segment_profiles.csv"
    profiles: dict[str, str] = {}
    if profile_file.exists():
        with profile_file.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                profiles[str(row.get("Segment", ""))] = row.get("Business_Profile", "")
    with segment_file.open(newline="", encoding="utf-8") as source, SessionLocal.begin() as session:
        for row in csv.DictReader(source):
            customer_id = str(row.get("Customer_ID", "")).strip()
            segment = str(row.get("Segment", "")).strip()
            if not customer_id or not segment:
                continue
            current = session.get(SegmentAssignment, customer_id)
            if current is None:
                session.add(SegmentAssignment(customer_id=customer_id, segment=segment, business_profile=profiles.get(segment)))


def store_prediction(customer_id: str | None, features: dict[str, Any], churn: dict[str, Any], clv: dict[str, Any]) -> None:
    """Persist a completed score, including the inputs used to produce it."""
    with SessionLocal.begin() as session:
        session.add(Prediction(customer_id=customer_id, churn_probability=churn["probability"], churn_prediction=churn["prediction"], predicted_clv=clv["predicted_value"], features=features))


@lru_cache(maxsize=1)
def _segment_feature_index() -> dict[str, dict[str, Any]]:
    """Load segment value and country features once for dashboard filters."""
    import pandas as pd

    feature_path = Path(__file__).resolve().parents[1] / "data/processed/ml_ready_data.csv"
    if not feature_path.exists():
        LOG.warning("Customer feature table is missing: %s", feature_path)
        return {}
    table = pd.read_csv(
        feature_path,
        usecols=lambda column: column in {"Customer_ID", "Monetary"} or column.startswith("Country_"),
    )
    table["customer_key"] = table["Customer_ID"].map(_normalize_id)
    from src.utils.segment_tiers import add_value_tiers

    table = add_value_tiers(table)
    return table.set_index("customer_key").to_dict(orient="index")


def list_segments(
    segment: str | None = None,
    country: str | None = None,
    value_tier: str | None = None,
) -> list[dict[str, Any]]:
    """Return assignments with historical value tiers and optional filters."""
    with SessionLocal() as session:
        query = select(SegmentAssignment)
        if segment:
            query = query.where(SegmentAssignment.segment == segment)
        records = session.scalars(query.order_by(SegmentAssignment.customer_id)).all()
        feature_index = _segment_feature_index()
        result = []
        country_key = f"Country_{country}" if country else None
        for item in records:
            customer_id = _normalize_id(item.customer_id)
            features = feature_index.get(customer_id, {})
            if value_tier and features.get("value_tier") != value_tier:
                continue
            if country_key and not bool(features.get(country_key, 0)):
                continue
            result.append({
                "customer_id": item.customer_id,
                "segment": item.segment,
                "business_profile": item.business_profile,
                "value_tier": features.get("value_tier", "Unknown"),
                "historical_monetary_value": features.get("historical_monetary_value"),
            })
        return result


def _normalize_id(value: Any) -> str:
    """Normalize numeric IDs read from CSV files."""
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    except (TypeError, ValueError):
        return str(value)


def list_predictions(limit: int = 500) -> list[dict[str, Any]]:
    """Return customers ordered by churn risk weighted by predicted value."""
    with SessionLocal() as session:
        priority = Prediction.churn_probability * Prediction.predicted_clv
        rows = session.scalars(select(Prediction).order_by(priority.desc(), Prediction.churn_probability.desc()).limit(limit)).all()
        return [{"customer_id": row.customer_id, "churn_probability": row.churn_probability, "churn_prediction": row.churn_prediction, "predicted_clv": row.predicted_clv, "retention_priority": round(row.churn_probability * row.predicted_clv, 2)} for row in rows]
