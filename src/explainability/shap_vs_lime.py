from pathlib import Path
import sys

# Project root
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

import joblib
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer


# Paths
MODEL_DIR = BASE_DIR / "data" / "processed" / "model_data"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "explainability"

MODEL_PATH = MODEL_DIR / "random_forest_tuned.pkl"
DATA_PATH = MODEL_DIR / "X_val.csv"
CUSTOMER_IDS_PATH = MODEL_DIR / "customer_ids_val.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load model and validation data
print("\nLoading Random Forest Model...")
model = joblib.load(MODEL_PATH)

print("\nModel Loaded Successfully!")

print("\nLoading Validation Data...")
X_val = pd.read_csv(DATA_PATH)
customer_ids = pd.read_csv(CUSTOMER_IDS_PATH).iloc[:, 0]

print("Validation Data Shape:", X_val.shape)


# Verify features
if list(model.feature_names_in_) != list(X_val.columns):
    print("\nERROR: Feature names/order do not match.")
    sys.exit(1)

print("Feature Names Match: True")


# Select borderline customer
probabilities = model.predict_proba(X_val)[:, 1]

target_probability = 0.50

row_index = np.abs(
    probabilities - target_probability
).argmin()

customer_probability = probabilities[row_index]
customer_id = customer_ids.iloc[row_index]

print("\nSelected Customer:")
print("Row Index:", row_index)
print(
    "Churn Probability:",
    f"{customer_probability:.4f}"
)


# SHAP Explanation
print("\nCreating SHAP Explanation...")

shap_explainer = shap.TreeExplainer(model)

shap_values = shap_explainer(X_val)

# Select Churn = 1
shap_customer = shap_values[row_index, :, 1]

shap_values_array = shap_customer.values

shap_importance = pd.DataFrame({
    "Customer_ID": customer_id,
        "Feature": X_val.columns,
    "SHAP_Value": shap_values_array,
    "Absolute_SHAP": np.abs(shap_values_array)
})

shap_importance = shap_importance.sort_values("Absolute_SHAP", ascending=False)

print("\nTop SHAP Features:")

print(
    shap_importance[
        ["Feature", "SHAP_Value"]
    ].head(10).to_string(index=False)
)


# LIME Explanation
print("\nCreating LIME Explainer...")

lime_explainer = LimeTabularExplainer(
    training_data=X_val.values,
    feature_names=X_val.columns.tolist(),
    class_names=["No Churn", "Churn"],
    mode="classification",
    discretize_continuous=True,
    random_state=42
)


print("\nGenerating LIME Explanation...")

lime_explanation = lime_explainer.explain_instance(
    X_val.iloc[row_index].values,
    model.predict_proba,
    num_features=10
)


lime_list = lime_explanation.as_list()

lime_importance = pd.DataFrame(lime_list, columns=["Feature", "LIME_Weight"])

lime_importance["Absolute_LIME"] = np.abs(lime_importance["LIME_Weight"])

lime_importance = lime_importance.sort_values("Absolute_LIME", ascending=False)

print("\nTop LIME Features:")

print(
    lime_importance[
        ["Feature", "LIME_Weight"]
    ].to_string(index=False)
)


# Save LIME explanation
lime_output = OUTPUT_DIR / "lime_borderline_explanation.html"

lime_explanation.save_to_file(str(lime_output))

print("\nLIME Explanation Saved:")
print(lime_output)


# Create SHAP vs LIME comparison
comparison = pd.merge(
    shap_importance[
        ["Feature", "SHAP_Value", "Absolute_SHAP"]
    ],
    lime_importance[
        ["Feature", "LIME_Weight", "Absolute_LIME"]
    ],
    on="Feature",
    how="outer"
)

comparison["Absolute_SHAP"] = comparison["Absolute_SHAP"].fillna(0)

comparison["Absolute_LIME"] = comparison["Absolute_LIME"].fillna(0)

comparison["SHAP_Rank"] = (
    comparison["Absolute_SHAP"]
    .rank(
        ascending=False,
        method="min"
    )
)

comparison["LIME_Rank"] = (
    comparison["Absolute_LIME"]
    .rank(
        ascending=False,
        method="min"
    )
)

comparison = comparison.sort_values("Absolute_SHAP", ascending=False)
comparison.insert(0, "Customer_ID", customer_id)

comparison_output = (
    OUTPUT_DIR /
    "shap_vs_lime_comparison.csv"
)

comparison.to_csv(comparison_output, index=False)

print("\nComparison Saved:")
print(comparison_output)


# Plot top SHAP vs LIME features
top_features = comparison.head(10).copy()

top_features = top_features.sort_values("Absolute_SHAP", ascending=True)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Absolute_SHAP"],
    alpha=0.7,
    label="SHAP"
)

lime_values = (top_features["Absolute_LIME"])

plt.barh(
    top_features["Feature"],
    lime_values,
    alpha=0.5,
    label="LIME"
)

plt.xlabel("Absolute Explanation Weight")
plt.ylabel("Feature")

plt.title(
    f"SHAP vs LIME - Borderline Customer "
    f"(Churn Probability: {customer_probability:.2%})"
)

plt.legend()

plt.tight_layout()

plot_output = (
    OUTPUT_DIR /
    "shap_vs_lime_borderline.png"
)

plt.savefig(
    plot_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nComparison Plot Saved:")
print(plot_output)

print("\nSHAP vs LIME Analysis Completed Successfully!")