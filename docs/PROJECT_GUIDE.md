# Project Guide and PRD Traceability

## Product and architecture

This repository delivers a customer-level analytics prototype over UCI Online Retail II. The major layers are raw and interim transaction files, reusable Pandas feature builders, serialized ML/DL and clustering artifacts, a FastAPI scoring service with PostgreSQL persistence, and a Streamlit stakeholder dashboard.

![System architecture](architecture_diagram.svg)

![Persistence entity relationship diagram](er_diagram.svg)

![Project workflow](workflow_diagram.svg)

## Run it

Use the root README quick start to run the API, PostgreSQL, and dashboard using Docker Compose. For local development or Docker, install dependencies from the root `requirements.txt` file. This includes runtime, training, explainability, and development dependencies. Set `API_URL=http://127.0.0.1:8000`, start `uvicorn api.main:app --reload`, and separately start `streamlit run frontend/dashboard.py`. SHAP and LIME endpoints need the optional explanation packages from `requirements.txt`.

The feature pipeline is invoked from the repository root with `python -m src.pipeline`. It expects the two-sheet `data/raw/online_retail_II.xlsx` UCI file. The pipeline orders load, raw validation, cleaning, trend preparation, advanced and behavioral features, encoding, customer dataset construction, CLV dataset construction, and feature validation. It stops on the first failed stage. Model retraining and segmentation/XAI production are separate scripts under `src/models`, `src/segmentation`, and `src/explainability`; the checked-in artifacts are used for the out-of-box service.

## Data rules

- Both source sheets are combined chronologically and the code standardizes transaction types.
- Customer-level modeling excludes missing customer IDs. Nonpositive unit prices and zero quantities are removed during cleaning. Negative quantities and cancellation invoice markers are retained long enough to derive return/cancellation signals; positive purchase activity drives the churn label and many spend features.
- Features for churn/CLV use the historical observation period; the following 90 days are used for labels only. The leakage tests cover date separation, historical encoding inputs, uniqueness, and missingness.
- The cleaning stage computes IQR-based outlier flags; it does not automatically discard all extreme bulk purchases. Description values are normalized and mapped by StockCode.
- Country one-hot encoding and StockCode frequency encoding are available. StandardScaler artifacts are produced for scale-sensitive models.

## Available scoring and dashboard flows

| Route / page | Purpose |
|---|---|
| `POST /prediction/predict` | Single customer churn probability, calibrated 90-day purchase probability, and CLV; request requires all saved churn and CLV feature columns. |
| `POST /prediction/batch` | JSON list scoring, including calibrated 90-day purchase probability. |
| `POST /prediction/batch-csv` | Numeric customer feature CSV upload; returns churn, 90-day purchase probability, and CLV in a scored CSV and persists each score. Limit: 10,000 rows. |
| `POST /prediction/explain?method=shap` | Individual churn model SHAP contribution values. |
| `POST /prediction/explain?method=lime` | Individual LIME explanation using the saved training feature sample. |
| `GET /metadata`, `GET /health` | Required features, model info, readiness. |
| `GET /customer/segments` | Persisted K-Means assignments with historical-spend value tiers; filters by segment, country, and value tier. |
| `GET /customer/leaderboard` | Stored customer predictions sorted by churn risk weighted by predicted CLV. |
| `GET /customer/recommendations/{customer_id}` | Historical basket co-purchase suggestions, excluding products already purchased. |

The dashboard's Overview page has interactive customer-group (segment), country, and customer-value-tier filters that update its metrics and charts. The Priority customers page ranks saved predictions by churn probability weighted by predicted CLV, and supports risk/value filters, Customer ID search, and CSV exports. Before predictions have been saved, it can display the existing score snapshot and labels its value field as historical-spend proxy. Score a customer supports Customer ID search, churn probability, calibrated 90-day purchase chance, CLV, product suggestions, and on-demand SHAP or LIME contributions. Batch score CSV accepts an ad-hoc numeric feature file (up to 10,000 rows) and provides scored CSV downloads. Model & method includes saved test-set model results and the global SHAP feature-importance image. Explanations require saved training rows and the installed `lime`/`shap` dependencies. Revenue forecasting and product suggestions are baseline methods.

## Existing result snapshot

These values are from the provided saved test-result files and are not freshly recomputed by the packaging workflow.

| PRD success metric | Saved result | Target | Status |
|---|---:|---:|---|
| Tuned churn Random Forest ROC-AUC | 0.7918 | >= 0.80 | Below target |
| Deployed churn model ROC-AUC | 0.7905 | >= 0.80 | Below target |
| Tuned churn Random Forest recall | 0.8352 | >= 0.70 | Meets target |
| CLV Random Forest test R-squared | 0.8579 | >= 0.60 | Meets target |
| Churn probability Brier score, raw -> calibrated | 0.1841 -> 0.1824 | Lower is better | Improved on held-out test split |
| Single-customer API p95 | 268.94 ms (30 warm local in-process requests, SQLite persistence) | < 400 ms | Meets locally; excludes Docker/network/cold start |
| Streamlit dashboard AppTest p95 | 1,062.59 ms (20 warmed script runs; median 909.99 ms); cold start 6,548.55 ms | < 3 s | Warmed full-Overview Python execution is below target. AppTest excludes browser rendering, network transfer, and Docker startup; cold start is reported separately. |

The 23-feature Random Forest experiment is kept separate from the API-selected model and from the dashboard's default result table; its reported results must not be represented as the deployed model's score. The model selection table includes classical and deep learning churn models. The LSTM remains a days-to-next-purchase regression model. Separately, the API emits the complement of validation-fitted, Platt-calibrated churn probability as the 90-day purchase chance; its test Brier score improves over raw probabilities, while ROC-AUC remains below target. The API p95 was measured with 30 warm local TestClient scoring calls and includes SQLite persistence, but excludes network, Docker, and cold-start overhead. The dashboard's AppTest p95 includes Streamlit script execution and local API calls, not browser rendering. The recommendation route is a simple basket co-purchase baseline, not a tuned recommendation model.

## PRD coverage and known gaps

| Requirement area | Current delivery | Remaining gap / interpretation |
|---|---|---|
| Data cleaning and engineered features | Pipeline code, IQR flags, returns, RFM, purchase behavior, seasonal and product-affinity proxy features | Discount sensitivity is a price proxy. Product suggestions use a simple co-purchase baseline. |
| Churn and CLV modeling | Required classical classifiers, ANN, CLV regression candidates; test metric tables | Best checked-in churn ROC-AUC is below 0.80. Training run logging is CSV-based and scattered across scripts. |
| Purchase sequences and anomaly detection | LSTM days-to-next-purchase sequences; autoencoder anomaly artifacts; calibrated 90-day purchase probability in the serving API | Purchase probability is calibrated from the churn model, not predicted from LSTM purchase sequences. |
| Segmentation | K-Means plus GMM, evaluation scores, profiles and persisted assignments | The dashboard currently filters the primary saved K-Means customer assignment. |
| Explainability | Global SHAP, three representative SHAP examples, PDPs, plain-language rows, on-demand SHAP and LIME API | LIME explanation is local and available through the API/dashboard; there is no single combined immutable audit record. |
| API and storage | Single/JSON batch/CSV batch, metadata, health, PostgreSQL persistence for scores/segments | Docker/network p95 is not measured; auth is explicitly out of scope in the PRD. |
| Dashboard | All five Section 12.2 requirements are exposed across Overview, Priority customers, Score a customer, Batch score CSV, and Model & method: segment/country/value-tier filters, high-risk/high-value leaderboard, sales trend and forecast, searchable customer explanations and global importance, ad-hoc CSV scoring with CSV downloads | Fresh 20-run warmed AppTest p95 is 1.06 s, meeting the 3 s script-execution proxy; browser load on deployment remains unmeasured. Forecast and product suggestions remain baseline methods; SHAP/LIME depend on optional packages. |
| Docker and handover | Compose API + Streamlit + Postgres; README and guide | Compose execution on a clean machine still needs local validation with Docker and the selected Python package versions. |
| Tests / coverage / lint | 16 tests pass; `pytest --cov=src` reports 88% for modules measured during the run; Ruff passes | The coverage run does not import the separate top-level model-training scripts; Docker Compose, full dashboard load time, and clean-machine checks still need validation. |
| Final documentation | Architecture, ER diagrams, and project guide | See report for metric caveats and acceptance state. |

## Data and model artifacts

Raw input workbook: `data/raw/online_retail_II.xlsx`. Feature and prediction outputs are under `data/processed/`. Churn model artifact defaults to `data/processed/model_data/random_forest.pkl`; CLV defaults to `models_artifacts/clv_random_forest.pkl`. Override artifact locations with `CHURN_MODEL_PATH` and `CLV_MODEL_PATH`.

The runtime and training dependency files pin scikit-learn 1.9.1 to match the checked-in Random Forest pickles. If models are retrained, regenerate the matching model result tables and recalibration artifact together.

The local Compose database password in `.env.example` is only a development default. Change it before sharing the environment. Do not commit `.env` files or secrets.




