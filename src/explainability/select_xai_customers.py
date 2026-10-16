from pathlib import Path
import sys

# Project root
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

import joblib
import pandas as pd


# Paths
MODEL_DIR = BASE_DIR / "data" / "processed" / "model_data"

MODEL_PATH = MODEL_DIR / "random_forest_tuned.pkl"
X_VAL_PATH = MODEL_DIR / "X_val.csv"
CUSTOMER_IDS_PATH = MODEL_DIR / "customer_ids_val.csv"
CUSTOMER_IDS_PATH = MODEL_DIR / "customer_ids_val.csv"


# Load model
print("\nLoading Random Forest Model...")
model = joblib.load(MODEL_PATH)

print("\nModel Loaded Successfully!")


# Load validation data
print("\nLoading Validation Data...")
X_val = pd.read_csv(X_VAL_PATH)
customer_ids = pd.read_csv(CUSTOMER_IDS_PATH).iloc[:, 0]

print("Validation Data Shape:", X_val.shape)


# Predict churn probability
print("\nCalculating Churn Probabilities...")

churn_probability = model.predict_proba(X_val)[:, 1]

results = X_val.copy()
results.insert(0, "Customer_ID", customer_ids.values)
results["Churn_Probability"] = churn_probability


# Select the same representative risk bands used by the SHAP and plain-language scripts.
targets = {
    "low_risk": 0.15,
    "borderline": 0.50,
    "high_risk": 0.85
}

selected = {}
for risk_type, target in targets.items():
    row_index = (results["Churn_Probability"] - target).abs().idxmin()
    selected[risk_type] = results.loc[row_index]

low_risk = selected["low_risk"]
borderline = selected["borderline"]
high_risk = selected["high_risk"]


# Display results
print("\n========== REPRESENTATIVE CUSTOMERS ==========")

print("\nLow-Risk Customer")
print("Customer_ID:", low_risk["Customer_ID"])
print("Churn Probability:", round(low_risk["Churn_Probability"], 4))

print("\nHigh-Risk Customer")
print("Customer_ID:", high_risk["Customer_ID"])
print("Churn Probability:", round(high_risk["Churn_Probability"], 4))

print("\nBorderline Customer")
print("Customer_ID:", borderline["Customer_ID"])
print("Churn Probability:", round(borderline["Churn_Probability"], 4))

representatives = pd.DataFrame([low_risk, borderline, high_risk])[["Customer_ID", "Churn_Probability"]]
representatives.insert(0, "Risk_Type", ["low_risk", "borderline", "high_risk"])
representatives.to_csv(MODEL_DIR.parent / "explainability" / "xai_representative_customers.csv", index=False)
print("\nSaved: data/processed/explainability/xai_representative_customers.csv")
