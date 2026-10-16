import time
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# Paths
# ============================================================

X_TEST_PATH = "data/processed/model_data/X_test.csv"
Y_TEST_PATH = "data/processed/model_data/y_test.csv"

RF_MODEL_PATH = "data/processed/model_data/random_forest_tuned.pkl"
LGBM_MODEL_PATH = "data/processed/model_data/lightgbm_tuned.pkl"
XGB_MODEL_PATH = "data/processed/model_data/xgboost_tuned.pkl"

OUTPUT_PATH = "data/processed/model_results/tuned_models_final_test.csv"


# ============================================================
# Load test data
# ============================================================

print("\nLoading protected test data...")

X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).squeeze()

print("Test Shape :", X_test.shape)


# ============================================================
# Load models
# ============================================================

models = {
    "Random Forest Tuned": RF_MODEL_PATH,
    "LightGBM Tuned": LGBM_MODEL_PATH,
    "XGBoost Tuned": XGB_MODEL_PATH,
}


# ============================================================
# Evaluate models
# ============================================================

results = []

for model_name, model_path in models.items():

    print("\n" + "=" * 60)
    print(model_name.upper())
    print("=" * 60)

    print("Loading model...")
    model = joblib.load(model_path)

    start_time = time.time()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    evaluation_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )
    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    print("Accuracy  :", round(accuracy, 4))
    print("Precision :", round(precision, 4))
    print("Recall    :", round(recall, 4))
    print("F1 Score  :", round(f1, 4))
    print("ROC-AUC   :", round(roc_auc, 4))
    print("Eval Time :", round(evaluation_time, 4), "seconds")

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "Evaluation_Time": evaluation_time,
    })


# ============================================================
# Save results
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# Final comparison
# ============================================================

print("\n" + "=" * 60)
print("FINAL TUNED MODEL TEST COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)

print("\nResults saved to:")
print(OUTPUT_PATH)


# ============================================================
# PRD target check
# ============================================================

print("\n" + "=" * 60)
print("PRD TARGET CHECK")
print("=" * 60)

for _, row in results_df.iterrows():

    auc_pass = row["ROC-AUC"] >= 0.80
    recall_pass = row["Recall"] >= 0.70

    print(f"\n{row['Model']}")
    print(
        f"ROC-AUC >= 0.80 : "
        f"{'PASS' if auc_pass else 'FAIL'}"
    )
    print(
        f"Recall >= 0.70  : "
        f"{'PASS' if recall_pass else 'FAIL'}"
    )

    if auc_pass and recall_pass:
        print("Overall PRD Churn Target : PASS")
    else:
        print("Overall PRD Churn Target : NOT MET")