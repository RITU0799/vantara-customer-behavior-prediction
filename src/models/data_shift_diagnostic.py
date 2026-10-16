import pandas as pd


# ============================================================
# Paths
# ============================================================

X_TRAIN_PATH = "data/processed/model_data/X_train.csv"
X_VAL_PATH = "data/processed/model_data/X_val.csv"
X_TEST_PATH = "data/processed/model_data/X_test.csv"

Y_TRAIN_PATH = "data/processed/model_data/y_train.csv"
Y_VAL_PATH = "data/processed/model_data/y_val.csv"
Y_TEST_PATH = "data/processed/model_data/y_test.csv"


# ============================================================
# Load data
# ============================================================

print("\nLoading train, validation and test data...")

X_train = pd.read_csv(X_TRAIN_PATH)
X_val = pd.read_csv(X_VAL_PATH)
X_test = pd.read_csv(X_TEST_PATH)

y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
y_val = pd.read_csv(Y_VAL_PATH).squeeze()
y_test = pd.read_csv(Y_TEST_PATH).squeeze()


# ============================================================
# Dataset shapes
# ============================================================

print("\n" + "=" * 60)
print("DATASET SHAPES")
print("=" * 60)

print("Train      :", X_train.shape)
print("Validation :", X_val.shape)
print("Test       :", X_test.shape)


# ============================================================
# Target distribution
# ============================================================

print("\n" + "=" * 60)
print("CHURN DISTRIBUTION")
print("=" * 60)

for name, y in [
    ("Train", y_train),
    ("Validation", y_val),
    ("Test", y_test),
]:

    counts = y.value_counts().sort_index()
    percentages = y.value_counts(
        normalize=True
    ).sort_index() * 100

    print(f"\n{name}")
    print("Churn = 0 :", counts.get(0, 0),
          f"({percentages.get(0, 0):.2f}%)")
    print("Churn = 1 :", counts.get(1, 0),
          f"({percentages.get(1, 0):.2f}%)")


# ============================================================
# Feature distribution shift
# ============================================================

print("\n" + "=" * 60)
print("FEATURE DISTRIBUTION SHIFT")
print("=" * 60)

results = []

for feature in X_train.columns:

    train_mean = X_train[feature].mean()
    val_mean = X_val[feature].mean()
    test_mean = X_test[feature].mean()

    train_std = X_train[feature].std()
    val_std = X_val[feature].std()
    test_std = X_test[feature].std()

    if train_std == 0 or pd.isna(train_std):
        val_shift = 0
        test_shift = 0
    else:
        val_shift = abs(val_mean - train_mean) / train_std
        test_shift = abs(test_mean - train_mean) / train_std

    results.append({
        "Feature": feature,
        "Train_Mean": train_mean,
        "Val_Mean": val_mean,
        "Test_Mean": test_mean,
        "Train_Std": train_std,
        "Val_Std": val_std,
        "Test_Std": test_std,
        "Val_Shift": val_shift,
        "Test_Shift": test_shift,
    })


shift_df = pd.DataFrame(results)

shift_df = shift_df.sort_values(
    by="Test_Shift",
    ascending=False
)


# ============================================================
# Display largest shifts
# ============================================================

print("\nTop 20 features with largest TEST distribution shift:")

print(
    shift_df[
        [
            "Feature",
            "Train_Mean",
            "Val_Mean",
            "Test_Mean",
            "Val_Shift",
            "Test_Shift",
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("SHIFT SUMMARY")
print("=" * 60)

print(
    "Features with Test Shift > 0.5 SD :",
    (shift_df["Test_Shift"] > 0.5).sum()
)

print(
    "Features with Test Shift > 1.0 SD :",
    (shift_df["Test_Shift"] > 1.0).sum()
)

print(
    "Features with Test Shift > 2.0 SD :",
    (shift_df["Test_Shift"] > 2.0).sum()
)


# ============================================================
# Save diagnostic
# ============================================================

OUTPUT_PATH = "data/processed/model_results/data_shift_diagnostic.csv"

shift_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nDiagnostic saved to:")
print(OUTPUT_PATH)

print("\nData shift diagnostic completed.")