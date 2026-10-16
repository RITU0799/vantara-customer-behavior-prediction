import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.inspection import PartialDependenceDisplay


# Paths
DATA_DIR = Path("data/processed/model_data")
OUTPUT_DIR = Path("data/processed/explainability")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("\nLoading Random Forest model...")
model = joblib.load(DATA_DIR / "random_forest_tuned.pkl")

print("\nLoading validation data...")
X_val = pd.read_csv(DATA_DIR / "X_val.csv")

# Convert all numerical columns to float
X_val = X_val.astype(float)

print("Validation Data Shape:", X_val.shape)


# Top 3 influential features from SHAP
features = [
    ("Recency", "pdp_recency.png", "Partial Dependence Plot - Recency"),
    ("Recency_Score", "pdp_recency_score.png", "Partial Dependence Plot - Recency Score"),
    ("Engagement_Score", "pdp_engagement_score.png", "Partial Dependence Plot - Engagement Score")
]


# Create PDPs
for feature, filename, title in features:

    print(f"\nCreating PDP for: {feature}")

    fig, ax = plt.subplots(figsize=(8, 5))

    PartialDependenceDisplay.from_estimator(
        model,
        X_val,
        [feature],
        ax=ax
    )

    plt.title(title)
    plt.tight_layout()

    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"PDP Saved Successfully: {output_path}")

print("\nAll 3 PDPs generated Successfully!")