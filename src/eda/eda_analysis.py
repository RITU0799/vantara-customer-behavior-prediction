import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

ML_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_ready_data.csv"
)

TRANSACTION_FILE = (
    BASE_DIR
    / "data"
    / "interim"
    / "cleaned_retail_data.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "eda"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Load Data
# --------------------------------------------------

print("\nLoading ML Data...")

ml_df = pd.read_csv(ML_FILE)

print("ML Rows:", len(ml_df))
print("ML Columns:", len(ml_df.columns))


print("\nLoading Transaction Data...")

transaction_df = pd.read_csv(
    TRANSACTION_FILE,
    parse_dates=["InvoiceDate"]
)

print(
    "Transaction Rows:",
    len(transaction_df)
)


# --------------------------------------------------
# 1. Basic Summary
# --------------------------------------------------

print("\n========== BASIC SUMMARY ==========")

summary = ml_df[
    [
        "Recency",
        "Frequency",
        "Monetary",
        "Average_Order_Value",
        "Average_Basket_Size",
        "CLV",
        "Return_Rate",
        "Engagement_Score"
    ]
].describe()

print(summary)

summary.to_csv(
    OUTPUT_DIR / "basic_summary.csv"
)


# --------------------------------------------------
# 2. Churn Distribution
# --------------------------------------------------

print("\n========== CHURN DISTRIBUTION ==========")

churn_counts = (
    ml_df["Churn"]
    .value_counts()
    .sort_index()
)

print(churn_counts)

churn_rate = ml_df["Churn"].mean()

print(
    "Churn Rate:",
    round(churn_rate * 100, 2),
    "%"
)


plt.figure(figsize=(6, 4))

sns.countplot(
    data=ml_df,
    x="Churn"
)

plt.title(
    "Customer Churn Distribution"
)

plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "churn_distribution.png"
)

plt.close()


# --------------------------------------------------
# 3. Univariate Analysis
# --------------------------------------------------

print("\n========== UNIVARIATE ANALYSIS ==========")

features = [
    "Recency",
    "Frequency",
    "Monetary",
    "Average_Order_Value",
    "Average_Basket_Size",
    "CLV",
    "Purchase_Gap_Variance",
    "Seasonal_Purchase_Concentration",
    "Return_Rate",
    "Engagement_Score"
]


for feature in features:

    print(
        f"\n{feature}"
    )

    print(
        ml_df[feature].describe()
    )

    plt.figure(figsize=(7, 4))

    sns.histplot(
        ml_df[feature],
        bins=30,
        kde=True
    )

    plt.title(
        f"{feature} Distribution"
    )

    plt.xlabel(feature)
    plt.ylabel("Customers")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"{feature.lower()}_distribution.png"
    )

    plt.close()


# --------------------------------------------------
# 4. Churn vs Important Features
# --------------------------------------------------

print("\n========== CHURN VS FEATURES ==========")

churn_analysis = (
    ml_df
    .groupby("Churn")[
        [
            "Recency",
            "Frequency",
            "Monetary",
            "Average_Order_Value",
            "CLV",
            "Return_Rate",
            "Engagement_Score"
        ]
    ]
    .mean()
)

print(churn_analysis)

churn_analysis.to_csv(
    OUTPUT_DIR / "churn_vs_features.csv"
)


important_features = [
    "Recency",
    "Frequency",
    "Monetary",
    "CLV",
    "Return_Rate",
    "Engagement_Score"
]


for feature in important_features:

    plt.figure(figsize=(7, 4))

    sns.boxplot(
        data=ml_df,
        x="Churn",
        y=feature
    )

    plt.title(
        f"{feature} vs Churn"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"{feature.lower()}_vs_churn.png"
    )

    plt.close()


# --------------------------------------------------
# 5. Correlation Analysis
# --------------------------------------------------

print("\n========== CORRELATION ANALYSIS ==========")

numeric_df = ml_df.select_dtypes(
    include=np.number
).copy()


# Customer ID is an identifier,
# not a predictive feature

numeric_df = numeric_df.drop(
    columns=["Customer_ID"],
    errors="ignore"
)


correlation = numeric_df.corr()

print(
    correlation["Churn"]
    .sort_values(
        ascending=False
    )
)

correlation.to_csv(
    OUTPUT_DIR / "correlation_matrix.csv"
)


plt.figure(
    figsize=(14, 10)
)

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0
)

plt.title(
    "Feature Correlation Matrix"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png"
)

plt.close()


# --------------------------------------------------
# 6. VIF Analysis
# --------------------------------------------------

print("\n========== VIF ANALYSIS ==========")

vif_features = [
    "Recency",
    "Frequency",
    "Monetary",
    "Total_Quantity",
    "Average_Order_Value",
    "Customer_Lifetime_Days",
    "Average_Basket_Size",
    "CLV",
    "Unique_Products",
    "Unique_Countries",
    "Purchase_Frequency_Per_Month",
    "Purchase_Frequency_Trend",
    "Purchase_Gap_Variance",
    "Seasonal_Purchase_Concentration",
    "Product_Category_Affinity",
    "Engagement_Score",
    "Return_Rate",
    "Price_Sensitivity_Proxy"
]


vif_data = ml_df[
    vif_features
].copy()


# Replace infinite values

vif_data = vif_data.replace(
    [np.inf, -np.inf],
    np.nan
)


# Replace missing values

vif_data = vif_data.fillna(0)


vif_results = []


for i, feature in enumerate(
    vif_data.columns
):

    vif_value = variance_inflation_factor(
        vif_data.values,
        i
    )

    vif_results.append(
        {
            "Feature": feature,
            "VIF": vif_value
        }
    )


vif_df = pd.DataFrame(
    vif_results
).sort_values(
    "VIF",
    ascending=False
)


print(vif_df)

vif_df.to_csv(
    OUTPUT_DIR / "vif_results.csv",
    index=False
)


# --------------------------------------------------
# 7. Revenue Calculation
# --------------------------------------------------

transaction_df["Revenue"] = (
    transaction_df["Quantity"]
    * transaction_df["Price"]
)


# --------------------------------------------------
# 8. Country Analysis
# --------------------------------------------------

print("\n========== COUNTRY ANALYSIS ==========")

country_analysis = (
    transaction_df
    .dropna(subset=["Customer_ID"])
    .groupby("Country")
    .agg(
        Customers=(
            "Customer_ID",
            "nunique"
        ),
        Orders=(
            "Invoice",
            "nunique"
        ),
        Revenue=(
            "Revenue",
            "sum"
        )
    )
    .sort_values(
        "Revenue",
        ascending=False
    )
)


print(
    country_analysis.head(10)
)


country_analysis.to_csv(
    OUTPUT_DIR / "country_analysis.csv"
)


# --------------------------------------------------
# 9. Seasonal Analysis
# --------------------------------------------------

print("\n========== SEASONAL ANALYSIS ==========")

transaction_df["Month"] = (
    transaction_df["InvoiceDate"]
    .dt.month
)


monthly_analysis = (
    transaction_df
    .groupby("Month")
    .agg(
        Orders=(
            "Invoice",
            "nunique"
        ),
        Customers=(
            "Customer_ID",
            "nunique"
        ),
        Revenue=(
            "Revenue",
            "sum"
        )
    )
)


print(monthly_analysis)


monthly_analysis.to_csv(
    OUTPUT_DIR / "monthly_analysis.csv"
)


# Monthly order trend

plt.figure(
    figsize=(8, 4)
)

sns.lineplot(
    data=monthly_analysis,
    x=monthly_analysis.index,
    y="Orders",
    marker="o"
)

plt.title(
    "Monthly Order Trend"
)

plt.xlabel("Month")
plt.ylabel("Orders")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "monthly_order_trend.png"
)

plt.close()


# Monthly revenue trend

plt.figure(
    figsize=(8, 4)
)

sns.lineplot(
    data=monthly_analysis,
    x=monthly_analysis.index,
    y="Revenue",
    marker="o"
)

plt.title(
    "Monthly Revenue Trend"
)

plt.xlabel("Month")
plt.ylabel("Revenue")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "monthly_revenue_trend.png"
)

plt.close()


# --------------------------------------------------
# 10. EDA Hypotheses
# --------------------------------------------------

print("\n========== EDA HYPOTHESES ==========")


hypotheses = """
Vantara Customer Intelligence Platform
EDA Hypotheses for Modeling

1. Customers with higher recency are more likely to churn.

2. Customers with lower purchase frequency are more likely to churn.

3. Customers with lower monetary value are more likely to churn.

4. Customers with higher RFM scores are less likely to churn.

5. Customers with higher engagement scores are less likely to churn.

6. Customers purchasing a wider range of products are less likely
   to churn.

7. Customers with irregular purchase behavior may have different
   churn patterns.

8. Customers with highly seasonal purchasing behavior may show
   different churn risk.

9. Higher return rates may be associated with higher churn risk.

10. Purchase frequency trend may provide additional churn signal
    beyond static purchase frequency.

11. Customer lifetime and historical spending patterns may provide
    useful information for prioritizing high-value customers.

12. Country and seasonal purchasing patterns may reveal meaningful
    differences in customer behavior and revenue distribution.

These hypotheses will be tested during the modeling stage using
statistical analysis and machine learning models.
"""

print(hypotheses)


with open(
    OUTPUT_DIR / "eda_hypotheses.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(hypotheses)


# --------------------------------------------------
# Completed
# --------------------------------------------------

print("\n====================================")
print("EDA COMPLETED SUCCESSFULLY!")
print("====================================")

print("\nEDA Files saved in:")

print(OUTPUT_DIR)