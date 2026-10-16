from pathlib import Path
import sys

# Project root
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

import joblib
import pandas as pd

from lime.lime_tabular import LimeTabularExplainer


# Paths
MODEL_DIR = BASE_DIR / "data" / "processed" / "model_data"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "explainability"

MODEL_PATH = MODEL_DIR / "random_forest_tuned.pkl"
TRAIN_PATH = MODEL_DIR / "X_train.csv"
VAL_PATH = MODEL_DIR / "X_val.csv"
CUSTOMER_IDS_PATH = MODEL_DIR / "customer_ids_val.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load model
print("\nLoading Random Forest Model...")

model = joblib.load(MODEL_PATH)
print("Model Loaded Successfully!")


# Load data
print("\nLoading Training Data...")

X_train = pd.read_csv(TRAIN_PATH)
print("Training Data Shape:", X_train.shape)


print("\nLoading Validation Data...")

X_val = pd.read_csv(VAL_PATH)
print("Validation Data Shape:", X_val.shape)


# LIME Explainer
print("\nCreating LIME Explainer...")

explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["Active", "Churn"],
    mode="classification",
    random_state=42
)

print("\nLIME Explainer Created Successfully!")


# Borderline customer
probabilities = model.predict_proba(X_val)[:, 1]
row_index = int((probabilities - 0.50).__abs__().argmin())
customer = X_val.iloc[row_index].values


# Generate explanation
print("\nGenerating LIME Explanation...")

explanation = explainer.explain_instance(
    customer,
    model.predict_proba,
    num_features=10
)

print("\nLIME Explanation Generated Successfully!")


# Display explanation
print("\n========== LIME EXPLANATION ==========")

for feature, weight in explanation.as_list():

    print(
        f"{feature} : "
        f"{weight:.4f}"
    )


# Save explanation
output_path = OUTPUT_DIR / "lime_borderline_explanation.html"

explanation.save_to_file(str(output_path))

print("\nLIME Explanation Saved Successfully!")

print("Output:", output_path)