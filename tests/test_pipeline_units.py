"""Focused unit coverage for the reusable data and orchestration modules."""
from __future__ import annotations

import subprocess

import pandas as pd
import pytest

from src.data import clean_data, load_data, validation
from src import pipeline, recommendations


def retail_rows() -> pd.DataFrame:
    """Return small transactions that exercise cleaning edge cases."""
    return pd.DataFrame(
        [
            ["100", "A", " mug ", 2, "2010-01-02", 3.0, 10, "UK"],
            ["100", "A", "mug", 2, "2010-01-02", 3.0, 10, "UK"],
            ["C101", "B", None, -1, "2010-01-01", 5.0, 10, "UK"],
            ["102", "C", "plate", 1, "2010-01-03", 0.0, 10, "UK"],
            ["103", "D", "cup", 0, "2010-01-03", 2.0, 10, "UK"],
            ["104", "E", "bowl", 1, None, 2.0, 10, "UK"],
            ["105", "F", "bowl", 1, "2010-01-03", 2.0, None, "UK"],
        ],
        columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer_ID", "Country"],
    ).assign(InvoiceDate=lambda frame: pd.to_datetime(frame["InvoiceDate"]))


def test_clean_data_removes_invalid_records_and_derives_flags() -> None:
    cleaned = clean_data.clean_data(retail_rows())
    assert len(cleaned) == 3
    assert cleaned["InvoiceDate"].is_monotonic_increasing
    assert cleaned["IsCancellation"].tolist() == [True, False, False]
    assert cleaned["IsReturn"].tolist() == [True, False, False]
    assert cleaned["Description"].tolist() == ["UNKNOWN", "MUG", "MUG"]
    assert cleaned["TotalAmount"].tolist() == [-5.0, 6.0, 6.0]
    assert {"Quantity_Outlier", "Price_Outlier"}.issubset(cleaned.columns)


def test_validate_data_checks_schema_nulls_and_date_range() -> None:
    good = retail_rows().iloc[[0]].copy()
    good.loc[:, "InvoiceDate"] = pd.Timestamp("2010-01-02")
    assert validation.validate_data(good) is True
    with pytest.raises(ValueError, match="Missing required columns"):
        validation.validate_data(good.drop(columns=["Country"]))
    bad_null = good.copy()
    bad_null["Price"] = pd.Series([None], dtype="object", index=bad_null.index)
    with pytest.raises(ValueError, match="Null-rate check failed for Price"):
        validation.validate_data(bad_null)
    bad_date = good.copy()
    bad_date.loc[:, "InvoiceDate"] = pd.Timestamp("2012-01-01")
    with pytest.raises(ValueError, match="Date range ends after expected range"):
        validation.validate_data(bad_date)


def test_load_data_combines_sheets_and_sorts(monkeypatch: pytest.MonkeyPatch) -> None:
    first = retail_rows().iloc[[0]].copy()
    first.loc[:, "InvoiceDate"] = pd.Timestamp("2010-02-01")
    second = retail_rows().iloc[[0]].copy()
    second.loc[:, "InvoiceDate"] = pd.Timestamp("2010-01-01")
    monkeypatch.setattr(load_data, "INPUT_FILE", load_data.Path("source.xlsx"))
    output_file = load_data.Path("captured-combined.csv")
    monkeypatch.setattr(load_data, "OUTPUT_FILE", output_file)
    monkeypatch.setattr(load_data.pd, "read_excel", lambda _path, sheet_name: first.copy() if "2009" in sheet_name else second.copy())
    captured: dict[str, pd.DataFrame] = {}
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda frame, path, index=False: captured.setdefault(str(path), frame.copy()))
    load_data.load_data()
    saved = captured[str(output_file)]
    assert len(saved) == 2
    assert saved["InvoiceDate"].is_monotonic_increasing
    assert saved["Customer_ID"].tolist() == [10, 10]


def test_pipeline_runs_stages_in_order_and_stops_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    original_run_stage = pipeline.run_stage
    called: list[str] = []
    monkeypatch.setattr(pipeline, "run_stage", called.append)
    pipeline.run_pipeline()
    assert called[0] == "src.data.load_data"
    assert called[-1] == "src.features.feature_validation"
    assert len(called) == 10

    def fail(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, "unit-test")

    monkeypatch.setattr(pipeline, "run_stage", original_run_stage)
    monkeypatch.setattr(pipeline.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.run_stage("src.data.load_data")


def test_recommendations_return_unseen_co_purchases(monkeypatch: pytest.MonkeyPatch) -> None:
    transactions = pd.DataFrame(
        [
            ["i1", pd.Timestamp("2010-01-01"), 10, "A", "Alpha", 1],
            ["i1", pd.Timestamp("2010-01-01"), 10, "B", "Beta", 1],
            ["i2", pd.Timestamp("2010-01-02"), 20, "A", "Alpha", 1],
            ["i2", pd.Timestamp("2010-01-02"), 20, "C", "Gamma", 1],
        ],
        columns=["Invoice", "InvoiceDate", "Customer_ID", "StockCode", "Description", "Quantity"],
    )
    monkeypatch.setattr(recommendations, "_load_transactions", lambda: transactions)
    suggestions = recommendations.recommend_products("10")
    assert suggestions == [{"StockCode": "C", "Description": "Gamma", "co_purchase_orders": 1}]
    assert recommendations.recommend_products("999") == []
