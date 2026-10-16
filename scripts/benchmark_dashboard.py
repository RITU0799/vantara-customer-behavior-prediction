"""Measure cold and warm Streamlit dashboard script execution with AppTest."""
from __future__ import annotations

import argparse
import json
import os
import sys
import statistics
import math
import tempfile
import time
from pathlib import Path

def main() -> None:
    """Run the dashboard script repeatedly and report warm execution time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20, help="Number of measured runs after warm-up (default: 20).")
    parser.add_argument("--warmups", type=int, default=2, help="Unreported warm-up runs (default: 2).")
    args = parser.parse_args()
    if args.runs < 3:
        parser.error("--runs must be at least 3")
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    benchmark_tmp = root / "tmp" / "dashboard-benchmark"
    benchmark_tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(benchmark_tmp)
    os.environ["TMP"] = str(benchmark_tmp)
    os.environ["TEMP"] = str(benchmark_tmp)
    from streamlit.testing.v1 import AppTest

    dashboard = root / "frontend/dashboard.py"

    def run_once() -> float:
        started = time.perf_counter()
        app_test = AppTest.from_file(str(dashboard), default_timeout=60).run()
        duration_ms = (time.perf_counter() - started) * 1000
        if app_test.exception:
            errors = [str(item.message) for item in app_test.exception]
            raise RuntimeError(f"Dashboard raised during AppTest: {errors}")
        return duration_ms

    cold_ms = run_once()
    for _ in range(args.warmups):
        run_once()
    durations = [run_once() for _ in range(args.runs)]

    ordered = sorted(durations)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    print(
        json.dumps(
            {
                "measurement": "Streamlit AppTest script execution for the full-segment Overview",
                "cold_start_ms": round(cold_ms, 2),
                "warmup_runs": args.warmups,
                "runs": len(durations),
                "median_ms": round(statistics.median(durations), 2),
                "p95_ms": round(p95, 2),
                "max_ms": round(max(durations), 2),
                "prd_target_ms": 3000,
                "meets_target": p95 < 3000,
                "note": "Measures Python dashboard script execution, not browser rendering, network transfer, or Docker startup.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
