from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Project path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))


# File paths
INPUT_FILE = BASE_DIR / "data" / "processed" / "ml_ready_data.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "segmentation"
OUTPUT_FILE = OUTPUT_DIR / "kmeans_customer_segments.csv"
EVALUATION_FILE = OUTPUT_DIR / "kmeans_evaluation.csv"
ELBOW_PLOT = OUTPUT_DIR / "kmeans_elbow_curve.png"
SILHOUETTE_PLOT = OUTPUT_DIR / "kmeans_silhouette_scores.png"
KMEANS_MODEL_FILE = OUTPUT_DIR / "kmeans_model.pkl"
KMEANS_SCALER_FILE = OUTPUT_DIR / "kmeans_scaler.pkl"

RANDOM_STATE = 42


# 1. Load Customer Data
print("\nLoading Customer Data...")

df = pd.read_csv(INPUT_FILE)
print(f"Dataset Shape : {df.shape}")


# 2. Select Features
customer_ids = df["Customer_ID"].copy()

exclude_columns = ["Customer_ID", "Churn"]

feature_columns = [
    column
    for column in df.select_dtypes(include=np.number).columns
    if column not in exclude_columns
]

X = df[feature_columns].copy()

print(f"Features Used : {len(feature_columns)}")
print(f"Feature Shape : {X.shape}")


# 3. Handle Missing / Infinite Values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)


# 4. Scale Features
print("\nScaling Features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature Scaling Completed!")


# 5. Elbow + Silhouette Evaluation
print("\nEvaluating K-Means Models...")

k_values = range(2, 9)

inertia_values = []
silhouette_values = []
davies_bouldin_values = []

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)

    inertia_values.append(inertia)
    silhouette_values.append(silhouette)
    davies_bouldin_values.append(davies_bouldin)

    print(
    f"k={k} | "
    f"Inertia={inertia:.2f} | "
    f"Silhouette={silhouette:.4f} | "
    f"Davies-Bouldin={davies_bouldin:.4f}"
)


# 6. Determine Elbow K
k_array = np.array(list(k_values), dtype=float)
inertia_array = np.array(inertia_values, dtype=float)

# Normalize values
x_normalized = (k_array - k_array.min()) / (k_array.max() - k_array.min())

y_normalized = (inertia_array - inertia_array.min()) / (inertia_array.max() - inertia_array.min())

# Straight line between first and last points
line_y = 1 - x_normalized

# Distance from each point to the line
distances = np.abs(y_normalized - line_y) / np.sqrt(2)

elbow_index = np.argmax(distances)
elbow_k = int(k_array[elbow_index])


# 7. Determine Best Silhouette K
silhouette_index = np.argmax(silhouette_values)

silhouette_k = int(list(k_values)[silhouette_index])

best_silhouette = silhouette_values[silhouette_index]


# 8. Final K Selection
# Silhouette score is highest at K=3.
# The elbow curve also shows the major improvement
# occurring up to K=3, after which the improvement
# in clustering quality becomes smaller.

selected_k = silhouette_k
selected_silhouette = best_silhouette


# 9. Print K Selection
print("\nK-Means Selection")
print("-------------------------------------")

print(f"Elbow K              : {elbow_k}")
print(f"Best Silhouette K    : {silhouette_k}")
print(f"Selected K           : {selected_k}")
print(f"Selected Silhouette  : {selected_silhouette:.4f}")


print("\nSelection Basis:")
print("Elbow Method + Silhouette Score")
print(f"K={silhouette_k} provides the highest Silhouette Score")
print("and was selected as the final cluster structure.")


# 10. Save Evaluation Results
evaluation = pd.DataFrame({
    "K": list(k_values),
    "Inertia": inertia_values,
    "Silhouette": silhouette_values,
    "Davies_Bouldin": davies_bouldin_values
})

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

evaluation.to_csv(EVALUATION_FILE, index=False)


# 11. Elbow Plot
plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Method")

plt.xticks(list(k_values))
plt.grid(True)

plt.savefig(ELBOW_PLOT, bbox_inches="tight")
plt.close()


# 12. Silhouette Plot
plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    silhouette_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("K-Means Silhouette Scores")

plt.xticks(list(k_values))
plt.grid(True)

plt.savefig(SILHOUETTE_PLOT, bbox_inches="tight")
plt.close()


# 13. Train Final K-Means Model
print("\nTraining Final K-Means Model...")

final_kmeans = KMeans(
    n_clusters=selected_k,
    random_state=RANDOM_STATE,
    n_init=10
)

cluster_labels = final_kmeans.fit_predict(X_scaled)

joblib.dump(final_kmeans, KMEANS_MODEL_FILE)
joblib.dump(scaler, KMEANS_SCALER_FILE)

print(f"\nK-Means Model Saved   : {KMEANS_MODEL_FILE}")
print(f"K-Means Scaler Saved  : {KMEANS_SCALER_FILE}")


# 14. Save Customer Segments
results = pd.DataFrame({"Customer_ID": customer_ids, "Segment": cluster_labels})
results.to_csv(OUTPUT_FILE, index=False)


# 15. Segment Distribution
print("\nCustomer Segment Distribution")
print("-------------------------------------")

segment_counts = (results["Segment"].value_counts().sort_index())

for segment, count in segment_counts.items():

    percentage = (count / len(results)) * 100

    print(
        f"Segment {segment}: "
        f"{count} customers "
        f"({percentage:.2f}%)"
    )


print("\nK-Means Segmentation Completed Successfully!")

print(f"Output File       : {OUTPUT_FILE}")
print(f"Evaluation File   : {EVALUATION_FILE}")
print(f"Elbow Plot        : {ELBOW_PLOT}")
print(f"Silhouette Plot   : {SILHOUETTE_PLOT}")