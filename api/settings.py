"""Environment-backed paths and service settings."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./vantara.db")
CHURN_MODEL_PATH = Path(os.getenv("CHURN_MODEL_PATH", "data/processed/model_data/random_forest.pkl"))
CLV_MODEL_PATH = Path(os.getenv("CLV_MODEL_PATH", "models_artifacts/clv_random_forest.pkl"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
