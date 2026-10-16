import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
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

X_TRAIN_PATH = "data/processed/model_data/X_train.csv"
X_VAL_PATH = "data/processed/model_data/X_val.csv"

Y_TRAIN_PATH = "data/processed/model_data/y_train.csv"
Y_VAL_PATH = "data/processed/model_data/y_val.csv"


# ============================================================
# Load data
# ============================================================

print("\nLoading training and validation data...")

X_train = pd.read_csv(X_TRAIN_PATH)
X_val = pd.read_csv(X_VAL_PATH)

y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
y_val = pd.read_csv(Y_VAL_PATH).squeeze()

print("Training Shape   :", X_train.shape)
print("Validation Shape :", X_val.shape)


# ============================================================
# Feature sets
# ============================================================

top_22_features = [
    "Engagement_Score",
    "Recency",
    "RFM_Score",
    "Monetary",
    "Price_Sensitivity_Proxy",
    "Purchase_Frequency_Trend",
    "Total_Quantity",
    "Frequency",
    "StockCode_Frequency",
    "Seasonal_Purchase_Concentration",
    "Unique_Products",
    "Purchase_Gap_Variance",
    "Recency_Score",
    "Average_Order_Value",
    "CLV",
    "Purchase_Frequency_Per_Month",
    "Average_Basket_Size",
    "Customer_Lifetime_Days",
    "Return_Rate",
    "Product_Category_Affinity",
    "Frequency_Score",
    "Monetary_Score",
]


top_16_features = [
    "Engagement_Score",
    "Recency",
    "RFM_Score",
    "Monetary",
    "Price_Sensitivity_Proxy",
    "Purchase_Frequency_Trend",
    "Total_Quantity",
    "Frequency",
    "StockCode_Frequency",
    "Seasonal_Purchase_Concentration",
    "Unique_Products",
    "Purchase_Gap_Variance",
    "Recency_Score",
    "Average_Order_Value",
    "CLV",
    "Purchase_Frequency_Per_Month",
]


feature_sets = {
    "All 64 Features": list(X_train.columns),
    "Top 22 Features": top_22_features,
    "Top 16 Features": top_16_features,
}


# ============================================================
# Cross-validation setup
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# ============================================================
# Results
# ============================================================

results = []


# ============================================================
# Run experiments
# ============================================================

for set_name, features in feature_sets.items():

    print("\n" + "=" * 70)
    print(set_name.upper())
    print("=" * 70)

    print("Number of features :", len(features))

    X_train_selected = X_train[features]
    X_val_selected = X_val[features]

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        max_features="sqrt",
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    cv_scores = cross_val_score(
        model,
        X_train_selected,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    cv_auc = cv_scores.mean()

    print(
        "CV ROC-AUC        :",
        round(cv_auc, 4)
    )

    # --------------------------------------------------------
    # Train on full training data
    # --------------------------------------------------------

    model.fit(
        X_train_selected,
        y_train,
    )

    # --------------------------------------------------------
    # Validation evaluation
    # --------------------------------------------------------

    y_val_pred = model.predict(
        X_val_selected
    )

    y_val_prob = model.predict_proba(
        X_val_selected
    )[:, 1]

    accuracy = accuracy_score(
        y_val,
        y_val_pred,
    )

    precision = precision_score(
        y_val,
        y_val_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        y_val_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        y_val_pred,
        zero_division=0,
    )

    val_auc = roc_auc_score(
        y_val,
        y_val_prob,
    )

    print(
        "Validation Accuracy :",
        round(accuracy, 4)
    )

    print(
        "Validation Precision:",
        round(precision, 4)
    )

    print(
        "Validation Recall   :",
        round(recall, 4)
    )

    print(
        "Validation F1       :",
        round(f1, 4)
    )

    print(
        "Validation ROC-AUC  :",
        round(val_auc, 4)
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append({
        "Feature_Set": set_name,
        "Feature_Count": len(features),
        "CV_ROC_AUC": cv_auc,
        "Validation_Accuracy": accuracy,
        "Validation_Precision": precision,
        "Validation_Recall": recall,
        "Validation_F1": f1,
        "Validation_ROC_AUC": val_auc,
    })


# ============================================================
# Final comparison
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Validation_ROC_AUC",
    ascending=False,
)


print("\n" + "=" * 70)
print("FEATURE SELECTION COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# Save results
# ============================================================

OUTPUT_PATH = (
    "data/processed/model_results/"
    "feature_selection_experiment.csv"
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\nResults saved to:")
print(OUTPUT_PATH)

print("\nFeature selection experiment completed.")