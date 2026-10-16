# Vantara Customer Behavior Prediction Platform

An end-to-end customer intelligence platform for churn prediction, 90-day customer lifetime value (CLV), customer segmentation, and explainable AI. It uses the UCI Online Retail II dataset, pre-trained machine learning models, FastAPI, Streamlit, and PostgreSQL.

## Key Features

* **Churn prediction:** Estimates customer churn risk.
* **Customer lifetime value:** Predicts customer value over a 90-day period.
* **Customer segmentation:** Supports customer grouping and behavioral analysis.
* **Explainable AI:** Uses SHAP and LIME to help explain model predictions.
* **Interactive dashboard:** Provides customer search, filters, segment analysis, retention opportunities, and visualizations.
* **REST API:** Supports individual predictions, batch scoring, and CSV uploads.
* **Database integration:** Uses PostgreSQL in Docker and supports local SQLite-based scoring where configured.

## Technology Stack

* **Language:** Python
* **Data analysis:** Pandas, NumPy, Matplotlib, Seaborn
* **Machine learning:** Scikit-learn, XGBoost, LightGBM, TensorFlow
* **Explainability:** SHAP, LIME
* **Backend:** FastAPI, Uvicorn, Pydantic
* **Frontend:** Streamlit, Plotly
* **Database:** PostgreSQL, SQLAlchemy
* **Deployment:** Docker, Docker Compose
* **Testing and code quality:** Pytest, pytest-cov, Ruff, Black

## Requirements

* Python 3.11 for the Docker runtime.
* Python 3.11+ for local development, subject to dependency compatibility.
* Docker Desktop with Docker Compose v2 for containerized deployment.

All Python dependencies are maintained in a single `requirements.txt` file. Package versions are not pinned, so a fresh installation may resolve to newer versions than those used during development.

## Quick Start: Docker

1. Copy `.env.example` to `.env`.
2. Set a suitable local database password in `.env`.
3. Open a terminal in the project root.
4. Build and start the application:

```powershell
docker compose up --build
```

Docker Compose starts PostgreSQL, the FastAPI backend, and the Streamlit dashboard.

### Access the Application

* **Dashboard:** http://localhost:8501
* **API documentation:** http://localhost:8000/docs
* **API health check:** http://localhost:8000/health

To stop the services:

```powershell
docker compose down
```

To stop the services and delete the local PostgreSQL volume, use `docker compose down -v` only when you intentionally want to remove the database data.

## Local Python Setup

Create and activate a virtual environment from the project root:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The local Python version should be compatible with the installed packages and saved model artifacts. Docker uses Python 3.11; validate both environments before deployment.

Copy the environment template:

```powershell
Copy-Item .env.example .env
```

Start the API in the first terminal:

```powershell
uvicorn api.main:app --reload
```

In a second terminal, activate the same virtual environment and start the dashboard:

```powershell
streamlit run frontend/dashboard.py
```

Configure `API_URL` as `http://127.0.0.1:8000` for the local dashboard when required by your environment.

## API Endpoints

* `POST /prediction/predict` — Predicts an individual customer's outcome.
* `POST /prediction/batch` — Scores a batch of customers.
* `POST /prediction/batch-csv` — Accepts a CSV containing the required model features and returns scored data.
* `GET /metadata` — Returns model metadata.
* `GET /health` — Checks API health.
* `GET /customer/recommendations/{customer_id}` — Provides historical basket co-purchase suggestions.

Prediction requests must follow the API's expected schema and feature requirements. The churn response can include a validation-calibrated 90-day purchase probability when the calibration artifact is available.

See the interactive API documentation at `/docs` for request schemas and endpoint details.

## Project Notebooks

The notebooks provide exploratory analysis and documentation of the data and model experiments:

* `notebooks/01_eda.ipynb` — Explores dataset structure, data quality, missing values, and purchasing patterns.
* `notebooks/02_feature_engineering.ipynb` — Inspects customer features and target distributions.
* `notebooks/03_model_experiments.ipynb` — Reviews available model experiment results without automatically retraining models.

The notebooks use bounded data reads where appropriate to reduce memory usage. The original retail dataset is large, so avoid loading the entire dataset unnecessarily.

## Data Pipeline and Model Training

Place the original UCI Online Retail II workbook at:

`data/raw/online_retail_II.xlsx`

Run the data pipeline from the project root:

```powershell
python -m src.pipeline
```

The pipeline processes the workbook, validates the schema and date range, cleans transactions, builds customer features, and validates the resulting tables.

Model retraining is separate from ordinary application startup. Use the appropriate experiment scripts under `src/models/` when retraining is intentionally required. Deep-learning experiments may require significant CPU, GPU, and disk resources.

The application is designed to load saved model artifacts for inference without retraining at startup.

## Explainable AI

SHAP and LIME dependencies are included in `requirements.txt`.

The API contains optional live explanation functionality. The actual availability of an explanation depends on the endpoint implementation, the loaded model, and the required background or reference data.

## Performance Checks

Start the API before running the benchmark scripts:

```powershell
python scripts/benchmark_api.py --requests 30
```

For the dashboard benchmark:

```powershell
$env:API_URL = "http://127.0.0.1:8000"
python scripts/benchmark_dashboard.py --runs 5
```

The dashboard benchmark uses Streamlit AppTest and measures script execution rather than full browser rendering. Treat results as directional measurements, not proof of end-user page-load performance.

## Testing and Code Quality

Run the test suite:

```powershell
python -m pytest
```

Run tests with coverage:

```powershell
python -m pytest --cov=src --cov-report=term-missing
```

Check code quality:

```powershell
python -m ruff check .
```

Check installed dependency consistency:

```powershell
python -m pip check
```

## Project Structure

```text
customer-behavior-prediction/
├── api/                  # FastAPI application and routes
├── frontend/             # Streamlit dashboard
├── data/
│   ├── raw/               # Original input workbook
│   ├── interim/           # Intermediate data
│   └── processed/         # Processed data and model files
├── models_artifacts/     # Saved model artifacts
├── notebooks/            # EDA, feature engineering, experiments
├── src/
│   ├── models/            # Model training and evaluation
│   ├── explainability/    # SHAP and LIME functionality
│   └── pipeline.py        # Data processing pipeline
├── scripts/               # Benchmark and utility scripts
├── docs/                  # Project guide and documentation
├── requirements.txt       # Unified Python dependencies
├── Dockerfile              # Container image configuration
└── docker-compose.yml      # Application services
```

Some files or folders may vary depending on the current project version.

## Dataset and License

This project uses the **Online Retail II** dataset (UCI Machine Learning Repository, dataset 502), licensed under CC BY 4.0.

Citation: Chen, D. (2012). *Online Retail II*. UCI Machine Learning Repository.

## Limitations

* Model performance depends on the data, feature definitions, and saved model artifacts.
* Churn is defined according to the project's configured prediction window.
* CLV estimates are predictions and should not be interpreted as guaranteed future revenue.
* Explanation outputs describe model behavior; they do not establish causation.
* Unpinned dependencies can change between installations. For reproducible deployments, validate package compatibility and record a tested environment.

For architecture, data rules, evaluation metrics, project requirements, and known limitations, see [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md).
