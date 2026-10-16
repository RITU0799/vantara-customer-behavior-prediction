from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "ml_ready_data.csv"
SEG_DIR = BASE_DIR / "data" / "processed" / "segmentation"
OUTPUT_PATH = SEG_DIR / "segmentation_comparison.csv"

print("\nLoading Customer Data...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset Shape : {df.shape}")

exclude_columns = ["Customer_ID", "Churn"]
features = [
    column
    for column in df.select_dtypes(include=np.number).columns
    if column not in exclude_columns
]
X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"Features Used : {len(features)}")

rows = []

# Evaluate the exact saved K-Means model/scaler used for customer segments.
kmeans_scaler = joblib.load(SEG_DIR / "kmeans_scaler.pkl")
kmeans_model = joblib.load(SEG_DIR / "kmeans_model.pkl")
X_kmeans = kmeans_scaler.transform(X)
kmeans_labels = kmeans_model.predict(X_kmeans)

kmeans_silhouette = silhouette_score(X_kmeans, kmeans_labels)
kmeans_db = davies_bouldin_score(X_kmeans, kmeans_labels)
rows.append({
    "Method": "K-Means",
    "Clusters": int(kmeans_model.n_clusters),
    "Silhouette": kmeans_silhouette,
    "Davies_Bouldin": kmeans_db,
})

# Evaluate the exact saved GMM model/scaler.
gmm_scaler = joblib.load(SEG_DIR / "gmm_scaler.pkl")
gmm_model = joblib.load(SEG_DIR / "gmm_model.pkl")
X_gmm = gmm_scaler.transform(X)
gmm_labels = gmm_model.predict(X_gmm)

gmm_silhouette = silhouette_score(X_gmm, gmm_labels)
gmm_db = davies_bouldin_score(X_gmm, gmm_labels)
rows.append({
    "Method": "GMM",
    "Clusters": int(gmm_model.n_components),
    "Silhouette": gmm_silhouette,
    "Davies_Bouldin": gmm_db,
})

results = pd.DataFrame(rows).round({"Silhouette": 6, "Davies_Bouldin": 6})
SEG_DIR.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_PATH, index=False)

print("\nSegmentation Comparison")
print("-------------------------------------")
print(results.to_string(index=False))
print(f"\nOutput File : {OUTPUT_PATH}")
