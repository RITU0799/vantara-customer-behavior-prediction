"""Measure warm single-customer scoring latency through the FastAPI app."""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
from joblib import parallel_backend

logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> None:
    """Benchmark the loaded model with a representative prepared customer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=30, help="Number of warm scoring calls")
    args = parser.parse_args()
    if args.requests < 5:
        parser.error("--requests must be at least 5")

    root = Path(__file__).resolve().parents[1]
    data_path = root / "data/processed/ml_ready_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Prepared customer table not found: {data_path}")

    sys.path.insert(0, str(root))
    # Use a private local DB file because FastAPI's sync worker thread does not
    # share SQLite's per-thread in-memory connections.
    database_path = root / f".benchmark-{uuid.uuid4().hex}.sqlite"
    os.environ["DATABASE_URL"] = f"sqlite:///{database_path.as_posix()}"
    from fastapi.testclient import TestClient

    from api.main import app
    from api.database import engine

    row = pd.read_csv(data_path, nrows=1).iloc[0]
    features: dict[str, float] = {}
    for name, value in row.items():
        if name in {"Customer_ID", "Churn"} or pd.isna(value):
            continue
        try:
            features[name] = float(value)
        except (TypeError, ValueError):
            continue
    customer_id = str(row["Customer_ID"])
    payload = {"customer_id": customer_id, "features": features}

    try:
        with parallel_backend("threading", n_jobs=1), TestClient(app) as client:
            samples_ms: list[float] = []
            for _ in range(args.requests):
                started = time.perf_counter()
                response = client.post("/prediction/predict", json=payload)
                elapsed_ms = (time.perf_counter() - started) * 1000
                response.raise_for_status()
                samples_ms.append(elapsed_ms)
    finally:
        engine.dispose()
        database_path.unlink(missing_ok=True)

    ordered = sorted(samples_ms)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    print(
        json.dumps(
            {
                "measurement": "in-process FastAPI TestClient; warm model; includes SQLite persistence",
                "requests": len(samples_ms),
                "median_ms": round(statistics.median(samples_ms), 2),
                "p95_ms": round(p95, 2),
                "max_ms": round(max(samples_ms), 2),
                "prd_target_ms": 400,
                "meets_target": p95 < 400,
                "note": "This excludes Docker, network, and cold-start overhead.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
