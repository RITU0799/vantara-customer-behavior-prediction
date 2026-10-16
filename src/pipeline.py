"""Run the scripted transaction-to-customer feature pipeline."""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("vantara.pipeline")


def run_stage(module: str) -> None:
    """Run one project module and fail immediately when it fails."""
    LOG.info("Starting pipeline stage: %s", module)
    subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True, env=os.environ.copy())
    LOG.info("Completed pipeline stage: %s", module)


def run_pipeline() -> None:
    """Rebuild raw, cleaned, validated, and engineered customer tables."""
    stages = (
        "src.data.load_data",
        "src.data.validation",
        "src.data.clean_data",
        "src.eda.revenue_forecast",
        "src.features.advanced_features",
        "src.features.behavior_features",
        "src.features.categorical_encoding",
        "src.features.ml_data",
        "src.features.clv_regression_data",
        "src.features.feature_validation",
    )
    for stage in stages:
        run_stage(stage)


def main() -> None:
    """Command-line entry point."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    run_pipeline()


if __name__ == "__main__":
    main()
