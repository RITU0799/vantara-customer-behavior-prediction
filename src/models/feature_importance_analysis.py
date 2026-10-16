import joblib
import pandas as pd


# ============================================================
# Paths
# ============================================================

RF_MODEL_PATH = "data/processed/model_data/random_forest_tuned.pkl"
LGBM_MODEL_PATH = "data/processed/model_data/lightgbm_tuned.pkl"
XGB_MODEL_PATH = "data/processed/model_data/xgboost_tuned.pkl"

OUTPUT_PATH = "data/processed/model_results/feature_importance_comparison.csv"


# ============================================================
# Load models
# ============================================================

print("\nLoading tuned models...")

rf_model = joblib.load(RF_MODEL_PATH)
lgbm_model = joblib.load(LGBM_MODEL_PATH)
xgb_model = joblib.load(XGB_MODEL_PATH)


# ============================================================
# Feature names
# ============================================================

feature_names = rf_model.feature_names_in_


# ============================================================
# Feature importance
# ============================================================

results = pd.DataFrame({
    "Feature": feature_names,
    "Random_Forest": rf_model.feature_importances_,
    "LightGBM": lgbm_model.feature_importances_,
    "XGBoost": xgb_model.feature_importances_,
})


# ============================================================
# Normalize tree importance
# ============================================================

for column in [
    "Random_Forest",
    "LightGBM",
    "XGBoost",
]:
    total = results[column].sum()

    if total > 0:
        results[column] = results[column] / total


# ============================================================
# Average importance
# ============================================================

results["Average_Importance"] = results[
    [
        "Random_Forest",
        "LightGBM",
        "XGBoost",
    ]
].mean(axis=1)


# ============================================================
# Rank features
# ============================================================

results["Average_Rank"] = results[
    "Average_Importance"
].rank(
    ascending=False,
    method="min"
)


results = results.sort_values(
    by="Average_Importance",
    ascending=False
)


# ============================================================
# Display top features
# ============================================================

print("\n" + "=" * 70)
print("TOP 30 FEATURES BY AVERAGE IMPORTANCE")
print("=" * 70)

print(
    results.head(30).to_string(
        index=False
    )
)


# ============================================================
# Display lowest importance
# ============================================================

print("\n" + "=" * 70)
print("BOTTOM 20 FEATURES BY AVERAGE IMPORTANCE")
print("=" * 70)

print(
    results.tail(20).to_string(
        index=False
    )
)


# ============================================================
# Country feature summary
# ============================================================

country_features = results[
    results["Feature"].str.startswith("Country_")
]

print("\n" + "=" * 70)
print("COUNTRY FEATURE IMPORTANCE")
print("=" * 70)

print(
    "Number of country features :",
    len(country_features)
)

print(
    "Average country importance :",
    round(
        country_features["Average_Importance"].mean(),
        6
    )
)

print(
    "Total country importance   :",
    round(
        country_features["Average_Importance"].sum(),
        6
    )
)


# ============================================================
# Save results
# ============================================================

results.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nFeature importance comparison saved to:")
print(OUTPUT_PATH)

print("\nFeature importance analysis completed.")