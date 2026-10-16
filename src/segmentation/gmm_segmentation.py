from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score


# PROJECT PATH
BASE_DIR = Path(__file__).resolve().parents[2]


# FILE PATHS
INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_ready_data.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "segmentation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "gmm_customer_segments.csv"
)

GMM_MODEL_FILE = OUTPUT_DIR / "gmm_model.pkl"
GMM_SCALER_FILE = OUTPUT_DIR / "gmm_scaler.pkl"


# RANDOM STATE
RANDOM_STATE = 42


# LOAD DATA
print("\nLoading Customer Data...")

df = pd.read_csv(INPUT_FILE)
print(f"Dataset Shape : {df.shape}")


# CUSTOMER ID
customer_ids = df["Customer_ID"].copy()


# SELECT FEATURES
exclude_columns = ["Customer_ID", "Churn"]

feature_columns = [
    column
    for column in df.select_dtypes(include=np.number).columns
    if column not in exclude_columns
]

X = df[feature_columns].copy()

print(f"Features Used : {len(feature_columns)}")
print(f"Feature Shape : {X.shape}")


# HANDLE MISSING / INFINITE VALUES
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)


# FEATURE SCALING
print("\nScaling Features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature Scaling Completed!")


# TEST DIFFERENT NUMBER OF COMPONENTS
print("\nEvaluating GMM Models...")

k_values = range(2, 7)

silhouette_scores = []
davies_bouldin_scores = []
bic_scores = []

for k in k_values:

    gmm = GaussianMixture(
        n_components=k,
        random_state=RANDOM_STATE,
        n_init=5,
        covariance_type="diag",
        max_iter=200
    )

    labels = gmm.fit_predict(X_scaled)

    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    bic = gmm.bic(X_scaled)

    silhouette_scores.append(silhouette)
    davies_bouldin_scores.append(davies_bouldin)
    bic_scores.append(bic)

    print(
    f"k={k} | "
    f"Silhouette={silhouette:.4f} | "
    f"Davies-Bouldin={davies_bouldin:.4f} | "
    f"BIC={bic:.2f}"
)


# SELECT K
best_index = np.argmax(silhouette_scores)
best_k = list(k_values)[best_index]
best_silhouette = silhouette_scores[best_index]


print("\nGMM Selection")
print("-------------------------------------")
print(f"Selected K        : {best_k}")
print(f"Silhouette Score  : {best_silhouette:.4f}")


# TRAIN FINAL GMM
print("\nTraining Final GMM Model...")

final_gmm = GaussianMixture(
    n_components=best_k,
    random_state=RANDOM_STATE,
    n_init=5,
    covariance_type="diag",
    max_iter=200
)

cluster_labels = final_gmm.fit_predict(X_scaled)

joblib.dump(final_gmm, GMM_MODEL_FILE)
joblib.dump(scaler, GMM_SCALER_FILE)

print(f"\nGMM Model Saved : {GMM_MODEL_FILE}")
print(f"GMM Scaler Saved  : {GMM_SCALER_FILE}")


# CUSTOMER SEGMENT RESULTS
results = pd.DataFrame({"Customer_ID": customer_ids, "GMM_Segment": cluster_labels})


# SAVE RESULTS
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_FILE, index=False)


# SEGMENT DISTRIBUTION
print("\nGMM Segment Distribution")
print("-------------------------------------")

segment_counts = (
    results["GMM_Segment"]
    .value_counts()
    .sort_index()
)

for segment, count in segment_counts.items():

    percentage = (count / len(results)) * 100

    print(
        f"Segment {segment}: "
        f"{count} customers "
        f"({percentage:.2f}%)"
    )


# FINAL MESSAGE
print("\nGMM Segmentation Completed Successfully!")

print(f"Output File : {OUTPUT_FILE}")
