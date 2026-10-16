from pathlib import Path
import sys

# Project root
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np


# Paths
MODEL_DIR = BASE_DIR / "data" / "processed" / "model_data"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "explainability"

MODEL_PATH = MODEL_DIR / "random_forest_tuned.pkl"
DATA_PATH = MODEL_DIR / "X_val.csv"
CUSTOMER_IDS_PATH = MODEL_DIR / "customer_ids_val.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load model
print("\nLoading Random Forest Model...")
model = joblib.load(MODEL_PATH)

print("\nModel Loaded Successfully!")
print("Model Feature Count:", model.n_features_in_)


# Load validation data
print("\nLoading Validation Data...")

X_val = pd.read_csv(DATA_PATH)
customer_ids = pd.read_csv(CUSTOMER_IDS_PATH).iloc[:, 0]
print("Validation Data Shape:", X_val.shape)


# Verify feature compatibility
if list(model.feature_names_in_) != list(X_val.columns):

    print("\nERROR: Model features and validation features do not match.")

    print("\nModel Features:")
    print(list(model.feature_names_in_))

    print("\nValidation Features:")
    print(list(X_val.columns))

    sys.exit(1)

print("\nFeature Names Match: True")


# Calculate churn probabilities
print("\nCalculating Churn Probabilities...")

probabilities = model.predict_proba(X_val)[:, 1]

print(
    "Probability Range:",
    round(probabilities.min(), 4),
    "to",
    round(probabilities.max(), 4)
)


# Select representative validation customers
targets = {
    "low_risk": 0.15,
    "borderline": 0.50,
    "high_risk": 0.85
}

customers = {}

print("\nSelecting Representative Customers...")

for name, target_probability in targets.items():

    row_index = np.abs(
        probabilities - target_probability
    ).argmin()

    actual_probability = probabilities[row_index]
    customer_id = customer_ids.iloc[row_index]

    customers[name] = row_index

    print(
        f"{name.replace('_', ' ').title()}: "
        f"Row Index = {row_index}, "
        f"Customer_ID = {customer_id}, "
        f"Churn Probability = {actual_probability:.4f}"
    )


# Create SHAP explainer
print("\nCreating SHAP Explainer...")

explainer = shap.TreeExplainer(model)

shap_values = explainer(X_val)

print("\nSHAP Values Calculated Successfully!")
print("SHAP Value Shape:", shap_values.shape)


# Generate force plots
print("\nGenerating SHAP Force Plots...")

for name, row_index in customers.items():

    probability = probabilities[row_index]
    customer_id = customer_ids.iloc[row_index]

    print(
        f"\nGenerating "
        f"{name.replace('_', ' ').title()} SHAP Force Plot..."
    )

    # Select positive class (Churn = 1)
    single_customer = shap_values[row_index, :, 1]

    # Create SHAP Explanation object
    single_explanation = shap.Explanation(
        values=single_customer.values,
        base_values=single_customer.base_values,
        data=single_customer.data,
        feature_names=X_val.columns.tolist()
    )

    # Create force plot
    shap.plots.force(
        single_explanation,
        matplotlib=True,
        show=False
    )

    plt.title(
        f"SHAP Explanation - "
        f"{name.replace('_', ' ').title()} "
        f"(Customer ID: {customer_id} | Churn Probability: {probability:.2%})"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        f"shap_force_{name}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_path)

print("\nAll SHAP Force Plots Generated Successfully!")