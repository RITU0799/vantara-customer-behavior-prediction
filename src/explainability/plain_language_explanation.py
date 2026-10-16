from pathlib import Path
import sys

# Project root
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

import joblib
import pandas as pd
import shap
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

print("Model Loaded Successfully!")


# Load validation data
print("\nLoading Validation Data...")

X_val = pd.read_csv(DATA_PATH)
customer_ids = pd.read_csv(CUSTOMER_IDS_PATH).iloc[:, 0]

print("Validation Data Shape:", X_val.shape)


# Verify feature compatibility
if list(model.feature_names_in_) != list(X_val.columns):

    print("\nERROR: Model features and validation features do not match.")
    sys.exit(1)

print("Feature Names Match: True")


# Calculate churn probabilities
print("\nCalculating Churn Probabilities...")

probabilities = model.predict_proba(X_val)[:, 1]


# Select representative customers
targets = {
    "low_risk": 0.15,
    "borderline": 0.50,
    "high_risk": 0.85
}

customers = {}

for name, target_probability in targets.items():

    row_index = np.abs(
        probabilities - target_probability
    ).argmin()

    customers[name] = row_index


# Create SHAP explainer
print("\nCreating SHAP Explainer...")

explainer = shap.TreeExplainer(model)

shap_values = explainer(X_val)

print("SHAP Values Calculated Successfully!")


# Generate plain-language explanations
print("\nGenerating Plain-Language Explanations...")
print("-------------------------------------")


explanations = []


for name, row_index in customers.items():

    probability = probabilities[row_index]
    customer_id = customer_ids.iloc[row_index]

    customer_shap = shap_values[row_index, :, 1]

    shap_df = pd.DataFrame({
        "Feature": X_val.columns,
        "SHAP_Value": customer_shap.values
    })

    # Strongest contributors toward higher churn
    positive = (
        shap_df[shap_df["SHAP_Value"] > 0]
        .sort_values("SHAP_Value", ascending=False)
        .head(3)
    )

    # Strongest contributors toward lower churn
    negative = (
        shap_df[shap_df["SHAP_Value"] < 0]
        .sort_values("SHAP_Value")
        .head(3)
    )


    # Risk label
    if probability < 0.30:
        risk_label = "Low"
    elif probability < 0.70:
        risk_label = "Borderline"
    else:
        risk_label = "High"


    # Positive-risk explanation
    positive_reasons = []

    for _, row in positive.iterrows():

        positive_reasons.append(
            f"{row['Feature']} increases predicted churn risk"
        )


    # Negative-risk explanation
    negative_reasons = []

    for _, row in negative.iterrows():

        negative_reasons.append(
            f"{row['Feature']} reduces predicted churn risk"
        )


    # Build plain-language flag
    explanation_text = (
        f"Customer is classified as {risk_label} churn risk "
        f"with a predicted churn probability of "
        f"{probability:.2%}. "
    )


    if positive_reasons:

        explanation_text += (
            "The main factors increasing churn risk are "
            + ", ".join(positive_reasons)
            + ". "
        )


    if negative_reasons:

        explanation_text += (
            "Factors reducing churn risk include "
            + ", ".join(negative_reasons)
            + "."
        )


    print(
        f"\n{name.replace('_', ' ').title()}"
    )

    print(
        f"Churn Probability : {probability:.2%}"
    )

    print(
        f"Risk Level        : {risk_label}"
    )

    print(
        f"Explanation       : {explanation_text}"
    )


    explanations.append({
        "Customer_Type": name,
        "Customer_ID": customer_id,
        "Row_Index": row_index,
        "Churn_Probability": probability,
        "Risk_Level": risk_label,
        "Plain_Language_Explanation": explanation_text
    })


# Save explanations
output_path = (
    OUTPUT_DIR /
    "plain_language_explanations.csv"
)

pd.DataFrame(explanations).to_csv(
    output_path,
    index=False
)


# Save text version
text_output = (
    OUTPUT_DIR /
    "plain_language_explanations.txt"
)

with open(
    text_output,
    "w",
    encoding="utf-8"
) as file:

    for item in explanations:

        file.write(
            f"{item['Customer_Type'].replace('_', ' ').title()}\n"
        )

        file.write(
            f"Customer ID: {item['Customer_ID']}\n"
        )

        file.write(
            f"Churn Probability: "
            f"{item['Churn_Probability']:.2%}\n"
        )

        file.write(
            f"Risk Level: "
            f"{item['Risk_Level']}\n"
        )

        file.write(
            f"{item['Plain_Language_Explanation']}\n"
        )

        file.write("\n")


print("\nPlain-Language Explanation Files Saved:")
print(output_path)
print(text_output)

print("\nPlain-Language XAI Completed Successfully!")