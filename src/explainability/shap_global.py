from pathlib import Path
import sys

# Project root
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt


# Paths
MODEL_DIR = BASE_DIR / "data" / "processed" / "model_data"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "explainability"

MODEL_PATH = MODEL_DIR / "random_forest_tuned.pkl"
DATA_PATH = MODEL_DIR / "X_train.csv"
OUTPUT_PATH = OUTPUT_DIR / "shap_global_summary.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load model
print("\nLoading Random Forest Model...")

model = joblib.load(MODEL_PATH)

print("\nModel Loaded Successfully!")


# Load training data
print("\nLoading Training Data...")

X_train = pd.read_csv(DATA_PATH)
print("Training Data Shape:", X_train.shape)


# Verify feature count
if hasattr(model, "n_features_in_"):
    print("Model Features:", model.n_features_in_)
    print("Data Features:", X_train.shape[1])

    if model.n_features_in_ != X_train.shape[1]:
        raise ValueError(
            f"Feature mismatch: model expects "
            f"{model.n_features_in_} features but X_train has "
            f"{X_train.shape[1]} features."
        )


# Create SHAP explainer
print("\nCreating SHAP Explainer...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

print("\nSHAP Values Calculated Successfully!")


# Handle SHAP output
if isinstance(shap_values, list):
    # Binary classification:
    # Class 1 = Churn
    shap_values_plot = shap_values[1]

else:
    # Newer SHAP versions may return:
    # (samples, features, classes)

    if len(shap_values.shape) == 3:
        # Select churn class
        shap_values_plot = shap_values[:, :, 1]

    else:
        shap_values_plot = shap_values


# Generate SHAP Global Feature Importance Plot
print("\nGenerating SHAP Global Feature Importance Plot...")

plt.figure(figsize=(12, 9))

shap.summary_plot(
    shap_values_plot,
    X_train,
    feature_names=X_train.columns,
    max_display=15,
    plot_size=None,
    show=False
)

# Improve readability
plt.xlabel(
    "SHAP value",
    fontsize=12,
    labelpad=10
)

plt.ylabel(
    "Features",
    fontsize=12,
    labelpad=10
)

plt.xticks(
    fontsize=10
)

plt.yticks(
    fontsize=10
)

plt.title(
    "Global SHAP Feature Importance — Random Forest",
    fontsize=14,
    pad=15
)

# Prevent axis/label overlap
plt.tight_layout(pad=2.0)

# Save high-quality image
plt.savefig(
    OUTPUT_PATH,
    dpi=220,
    bbox_inches="tight",
    pad_inches=0.3
)

plt.close()


# Completion
print("\nSHAP Global Explanation Saved Successfully!")

print("Output:", OUTPUT_PATH)