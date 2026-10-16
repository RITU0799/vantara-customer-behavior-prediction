"""Vantara customer intelligence dashboard."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="Vantara | Customer Intelligence",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_RESULTS_DIR = PROCESSED_DIR / "model_results"
EXPLAINABILITY_DIR = PROCESSED_DIR / "explainability"
ML_READY_PATH = PROCESSED_DIR / "ml_ready_data.csv"
TABLEAU_DATA_PATH = PROCESSED_DIR / "tableau_customer_data.csv"
SEGMENT_PROFILE_PATH = PROCESSED_DIR / "segmentation" / "segment_profiles.csv"
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

GRID, TEXT, MUTED = "#223247", "#eaf1f8", "#93a4b8"
TEAL, BLUE, ORANGE, PINK, GREEN, PURPLE = (
    "#41d6c2",
    "#61a9ff",
    "#ffad5a",
    "#ff6584",
    "#69d7a5",
    "#aa8cff",
)
PALETTE = [BLUE, TEAL, ORANGE, PURPLE, PINK, GREEN]

st.markdown(
    """
<style>
.stApp {background:#080e18;color:#eaf1f8}
[data-testid="stHeader"] {background:rgba(8,14,24,.92)}
.block-container {max-width:1500px;padding:1.55rem 2.3rem 3rem}
section[data-testid="stSidebar"] {display:none !important}
[data-testid="collapsedControl"] {display:none !important}
section[data-testid="stSidebar"] {background:#091522;border-right:1px solid #1c3044}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#9fb1c5}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {padding:.38rem .55rem;border-radius:7px}
h1,h2,h3,p,span,label {font-family:Inter,ui-sans-serif,system-ui,sans-serif}
h1,h2,h3 {color:#f4f8fc;letter-spacing:-.035em}
p,label,[data-testid="stCaptionContainer"] {color:#a8b6c7}
[data-testid="stMetric"] {background:transparent;padding:.38rem .15rem .58rem;border:0;border-bottom:1px solid #243247}
[data-testid="stMetricLabel"] {color:#9aabc0;font-size:.82rem}
[data-testid="stMetricValue"] {color:#f3f7fb;font-size:1.72rem;font-weight:650}
[data-testid="stTabs"] button {color:#a9b8c9;font-size:.92rem}
[data-testid="stTabs"] button[aria-selected="true"] {color:#49ddc6}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {background:#41d6c2}
[data-testid="stDataFrame"] {border:1px solid #26364a;border-radius:9px;overflow:hidden}
[data-testid="stAlert"] {background:#111b29;border-color:#2b3c51}
hr {display:none}
.brand {color:#42ddc5;font-size:clamp(3.3rem,6.2vw,5rem);font-weight:820;letter-spacing:-.065em;line-height:1}
.hero-title {color:#c9d5e3;font-size:clamp(1.35rem,2.5vw,1.8rem);font-weight:650;letter-spacing:-.035em;line-height:1.1;margin:.48rem 0 .55rem}
.hero-sub {color:#a7b6c8;font-size:1rem;margin:0}
.period {color:#94a7bd;font-size:.76rem;letter-spacing:.08em;text-transform:uppercase;text-align:right;padding-top:1.45rem}
.section-kicker {color:#43d9c2;font-size:.70rem;font-weight:700;letter-spacing:.17em;text-transform:uppercase;margin:1.55rem 0 .2rem}
.section-title {color:#eef4fa;font-size:1.26rem;font-weight:640;letter-spacing:-.025em;margin:0 0 .7rem}
.small-note {color:#8193a9;font-size:.78rem}
.side-brand {font-size:1.15rem;color:#f2f7fc;font-weight:800;letter-spacing:.12em}
.side-mark {color:#42ddc5;font-size:.72rem;font-weight:750;letter-spacing:.2em}
.author-footer {
    margin-top: 3rem;
    padding: 1.25rem 0 0.75rem;
    border-top: 1px solid rgba(148, 163, 184, 0.22);
    text-align: center;
    color: #94a3b8;
    font-size: 0.82rem;
}
.author-footer .author-name {
    color: #42ddc5;
    font-weight: 700;
    letter-spacing: 0.04em;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.is_file():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_customer_data() -> pd.DataFrame:
    customers = load_csv(str(ML_READY_PATH))
    display = load_csv(str(TABLEAU_DATA_PATH))

    if customers.empty:
        return pd.DataFrame()

    if not display.empty and "Customer_ID" in display:
        extra = [
            column
            for column in (
                "Customer_ID",
                "Churn_Probability",
                "Risk_Level",
                "Segment",
            )
            if column in display
        ]
        customers = customers.merge(
            display[extra],
            on="Customer_ID",
            how="left",
            suffixes=("", "_view"),
        )

    if "Segment" not in customers:
        segments = load_csv(
            str(PROCESSED_DIR / "segmentation" / "kmeans_customer_segments.csv")
        )
        if not segments.empty:
            customers = customers.merge(
                segments,
                on="Customer_ID",
                how="left",
            )

    profiles = load_csv(str(SEGMENT_PROFILE_PATH))
    profile_map = (
        dict(zip(profiles["Segment"].astype(str), profiles["Business_Profile"]))
        if {"Segment", "Business_Profile"}.issubset(profiles.columns)
        else {}
    )

    if "Segment" in customers:
        customers["Segment"] = (
            pd.to_numeric(customers["Segment"], errors="coerce")
            .astype("Int64")
            .astype(str)
        )
        customers["Segment profile"] = customers["Segment"].map(
            profile_map
        ).fillna("Customer group " + customers["Segment"])

    if "Risk_Level" not in customers or customers["Risk_Level"].isna().all():
        probability = pd.to_numeric(
            customers.get("Churn_Probability"),
            errors="coerce",
        )
        if probability.notna().any():
            customers["Risk_Level"] = pd.cut(
                probability,
                [-0.001, 0.30, 0.70, 1.001],
                labels=["Low", "Borderline", "High"],
            ).astype("string")
        else:
            customers["Risk_Level"] = pd.NA

    country_cols = [
        column
        for column in customers.columns
        if column.startswith("Country_")
    ]

    if country_cols:
        country_values = customers[country_cols].fillna(0)
        customers["Country"] = (
            country_values.idxmax(axis=1)
            .str.removeprefix("Country_")
        )
        customers.loc[
            country_values.max(axis=1).eq(0),
            "Country",
        ] = "Unknown"
    else:
        customers["Country"] = "Unknown"

    from src.utils.segment_tiers import add_value_tiers

    return add_value_tiers(customers)


def money(value: float, decimals: int = 0) -> str:
    return f"\u00A3{value:,.{decimals}f}"


def chart_style(fig: go.Figure, height: int = 315) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "family": "Inter, Arial, sans-serif",
            "color": TEXT,
            "size": 11,
        },
        margin={"l": 12, "r": 16, "t": 18, "b": 12},
        height=height,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "font": {"size": 10},
        },
        hoverlabel={
            "bgcolor": "#0e1826",
            "bordercolor": GRID,
            "font": {"color": TEXT},
        },
    )
    fig.update_xaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor=GRID,
        tickfont={"color": MUTED},
    )
    fig.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor=GRID,
        tickfont={"color": MUTED},
    )
    return fig


def section(label: str, title: str, note: str | None = None) -> None:
    st.markdown(
        f'<div class="section-kicker">{label}</div>'
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )
    if note:
        st.caption(note)


def api_get(path: str, **kwargs) -> dict | None:
    try:
        response = requests.get(
            f"{API_URL}{path}",
            timeout=8,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


data = load_customer_data()
revenue = load_csv(str(PROCESSED_DIR / "eda" / "monthly_revenue_trend.csv"))

hero_left, hero_right = st.columns([5, 1])

with hero_left:
    st.markdown(
        '<div class="brand">VANTARA</div>'
        '<div class="hero-title">Customer Behavior Prediction</div>'
        '<p class="hero-sub">Customer intelligence dashboard for spending, '
        'purchasing behavior, churn risk, and lifetime value.</p>',
        unsafe_allow_html=True,
    )

with hero_right:
    if not revenue.empty and {"Month", "Revenue"}.issubset(revenue.columns):
        observed = revenue[
            pd.to_numeric(revenue["Revenue"], errors="coerce").notna()
        ]
        months = pd.to_datetime(
            observed["Month"],
            errors="coerce",
        ).dropna()
    else:
        months = pd.Series(dtype="datetime64[ns]")

    period = (
        f"{months.min():%Y} - {months.max():%Y}"
        if not months.empty
        else "Online Retail II"
    )
    st.markdown(
        f'<div class="period">DATA PERIOD<br><b>{period}</b></div>',
        unsafe_allow_html=True,
    )

if data.empty:
    st.error(
        "Customer data is missing. Check that "
        "data/processed/ml_ready_data.csv is present."
    )
    st.stop()


# ---------------------------------------------------------------------
# Customer overview and filters
# ---------------------------------------------------------------------
section(
    "Overview",
    "Customer behavior",
    "Filters update the customer metrics and charts on this page.",
)

f1, f2, f3, f4 = st.columns([1.15, 1.15, 1.15, 1.15])

group_values = (
    sorted(data["Segment profile"].dropna().unique())
    if "Segment profile" in data
    else []
)
risk_values = (
    [
        name
        for name in ["High", "Borderline", "Low"]
        if name in set(data["Risk_Level"].dropna())
    ]
    if "Risk_Level" in data
    else []
)
country_values = sorted(
    name
    for name in data["Country"].dropna().unique()
    if name != "Unknown"
)
tier_values = (
    [
        value
        for value in ["High", "Medium", "Low"]
        if value in set(data["value_tier"].dropna())
    ]
    if "value_tier" in data
    else []
)

chosen_group = f1.selectbox("Customer group", ["All groups", *group_values])
chosen_risk = f2.selectbox("Risk level", ["All risks", *risk_values])
chosen_country = f3.selectbox("Country", ["All countries", *country_values])
chosen_tier = f4.selectbox("Customer value", ["All values", *tier_values])

view = data.copy()

if chosen_group != "All groups":
    view = view[view["Segment profile"] == chosen_group]
if chosen_risk != "All risks":
    view = view[view["Risk_Level"] == chosen_risk]
if chosen_country != "All countries":
    view = view[view["Country"] == chosen_country]
if chosen_tier != "All values":
    view = view[view["value_tier"] == chosen_tier]

churn_values = pd.to_numeric(view.get("Churn"), errors="coerce")
clv_values = pd.to_numeric(view.get("CLV"), errors="coerce")
high_risk = (
    int(view["Risk_Level"].eq("High").sum())
    if "Risk_Level" in view
    else 0
)

total_customers, churn_rate, average_clv, high_risk_count = st.columns(4)

total_customers.metric("Customers", f"{len(view):,}")
churn_rate.metric(
    "Historical churn rate",
    f"{churn_values.mean():.1%}"
    if churn_values.notna().any()
    else "-",
)
average_clv.metric(
    "Average Estimated CLV",
    money(clv_values.mean(), 2) if clv_values.notna().any() else "N/A",
)
high_risk_count.metric("High-risk customers", f"{high_risk:,}")

st.caption(
    "Historical churn is the dataset label; risk levels come from saved model "
    "scores. Value tiers use historical spend. Financial values are shown in GBP (\u00A3)."
)


# ---------------------------------------------------------------------
# Customer profile and buying activity
# ---------------------------------------------------------------------
section("Customer profile", "Customer groups and buying activity")

left, middle, right = st.columns([1, 1, 1.15], gap="large")

with left:
    if "Segment profile" in view and not view.empty:
        counts = (
            view["Segment profile"]
            .value_counts()
            .rename_axis("Customer group")
            .reset_index(name="Customers")
        )
        fig = px.pie(
            counts,
            names="Customer group",
            values="Customers",
            hole=0.64,
            color_discrete_sequence=PALETTE,
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent",
            marker={"line": {"color": "#0b1420", "width": 2}},
        )
        fig.update_layout(
            showlegend=True,
            legend={"orientation": "v", "x": 1.02, "y": 0.5},
            annotations=[
                {
                    "text": f"{len(view):,}<br>customers",
                    "showarrow": False,
                    "font": {"size": 15, "color": TEXT},
                }
            ],
        )
        st.plotly_chart(
            chart_style(fig, 310),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    else:
        st.info("No customer group data is available for this selection.")

with middle:
    if {"Frequency", "Monetary"}.issubset(view.columns):
        point_columns = ["Frequency", "Monetary"]
        if "Segment profile" in view:
            point_columns.append("Segment profile")

        points = view[point_columns].copy()
        points["Frequency"] = pd.to_numeric(
            points["Frequency"],
            errors="coerce",
        )
        points["Monetary"] = pd.to_numeric(
            points["Monetary"],
            errors="coerce",
        )
        points = points.dropna(subset=["Frequency", "Monetary"])

        if len(points) > 700:
            points = points.sample(700, random_state=21)

        fig = px.scatter(
            points,
            x="Frequency",
            y="Monetary",
            color="Segment profile" if "Segment profile" in points else None,
            color_discrete_sequence=PALETTE,
            opacity=0.65,
            labels={
                "Frequency": "Purchase frequency",
                "Monetary": "Historical sales (\u00A3)",
                "Segment profile": "Customer group",
            },
        )
        fig.update_traces(marker={"size": 6, "line": {"width": 0}})
        fig.update_yaxes(type="log", title="Historical sales (\u00A3, log scale)")
        st.plotly_chart(
            chart_style(fig, 310),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    else:
        st.info("Purchase activity fields are not available.")

with right:
    if "Recency" in view:
        activity = pd.to_numeric(
            view["Recency"],
            errors="coerce",
        ).dropna()
        fig = px.histogram(
            x=activity,
            nbins=20,
            color_discrete_sequence=[TEAL],
            labels={
                "x": "Days since last purchase",
                "count": "Customers",
            },
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title="Days since last purchase",
            yaxis_title="Customers",
            bargap=0.1,
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(
            chart_style(fig, 310),
            use_container_width=True,
            config={"displayModeBar": False},
        )


# ---------------------------------------------------------------------
# Churn intelligence and revenue trend
# ---------------------------------------------------------------------
section("Churn intelligence", "Retention signals")

lower_left, lower_right = st.columns([1, 1.4], gap="large")

with lower_left:
    if "Risk_Level" in view and view["Risk_Level"].notna().any():
        risk_counts = (
            view["Risk_Level"]
            .value_counts()
            .reindex(["High", "Borderline", "Low"])
            .dropna()
        )
        risk_colors = {
            "High": ORANGE,
            "Borderline": BLUE,
            "Low": GREEN,
        }
        fig = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            color=risk_counts.index,
            color_discrete_map=risk_colors,
            labels={
                "x": "Model risk level",
                "y": "Customers",
                "color": "Risk level",
            },
            text=risk_counts.values,
        )
        fig.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
            cliponaxis=False,
            marker_line_width=0,
        )
        fig.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(
            chart_style(fig, 290),
            use_container_width=True,
            config={"displayModeBar": False},
        )

with lower_right:
    if {"Month", "Revenue"}.issubset(revenue.columns):
        trend = revenue.copy()
        trend["Month"] = pd.to_datetime(
            trend["Month"],
            errors="coerce",
        )
        trend["Revenue"] = pd.to_numeric(
            trend["Revenue"],
            errors="coerce",
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=trend["Month"],
                y=trend["Revenue"],
                mode="lines",
                name="Monthly sales",
                line={"color": TEAL, "width": 2.5},
                fill="tozeroy",
                fillcolor="rgba(65,214,194,.08)",
            )
        )

        if "Forecast" in trend:
            trend["Forecast"] = pd.to_numeric(
                trend["Forecast"],
                errors="coerce",
            )
            fig.add_trace(
                go.Scatter(
                    x=trend["Month"],
                    y=trend["Forecast"],
                    mode="lines",
                    name="Baseline forecast",
                    line={"color": ORANGE, "width": 2, "dash": "dash"},
                )
            )

        fig.update_layout(
            xaxis_title=None,
            yaxis_title="Monthly sales (\u00A3)",
            yaxis_tickprefix="\u00A3",
            yaxis_tickformat="~s",
            hovermode="x unified",
        )
        st.plotly_chart(
            chart_style(fig, 290),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption("The sales projection is a baseline forecast for context.")


# ---------------------------------------------------------------------
# Priority customer leaderboard and CSV downloads
# ---------------------------------------------------------------------
section(
    "Churn intelligence",
    "Priority customers",
    "The list is ordered by churn probability weighted by predicted customer value.",
)

leaderboard_response = api_get(
    "/customer/leaderboard",
    params={"limit": 500},
)
leaderboard = pd.DataFrame(
    (leaderboard_response or {}).get("customers", [])
)
using_saved_scores = leaderboard.empty

if using_saved_scores and {"Churn_Probability", "Monetary"}.issubset(data.columns):
    score_snapshot = data.dropna(subset=["Churn_Probability"]).copy()

    if not score_snapshot.empty:
        probability_snapshot = pd.to_numeric(
            score_snapshot["Churn_Probability"],
            errors="coerce",
        ).fillna(0)
        spend_snapshot = pd.to_numeric(
            score_snapshot["Monetary"],
            errors="coerce",
        ).fillna(0)

        leaderboard = pd.DataFrame(
            {
                "customer_id": score_snapshot["Customer_ID"].astype(str),
                "churn_probability": probability_snapshot,
                "churn_prediction": (probability_snapshot >= 0.5).astype(int),
                "predicted_clv": np.nan,
                "retention_priority": probability_snapshot * spend_snapshot,
                "historical_spend": spend_snapshot,
            }
        )

if leaderboard.empty:
    st.info(
        "No saved predictions yet. Run an individual customer prediction "
        "to start this list."
    )
else:
    for column in [
        "churn_probability",
        "predicted_clv",
        "retention_priority",
    ]:
        if column not in leaderboard:
            leaderboard[column] = np.nan
        leaderboard[column] = pd.to_numeric(
            leaderboard[column],
            errors="coerce",
        )

    leaderboard["risk_level"] = pd.cut(
        leaderboard["churn_probability"],
        [-0.001, 0.30, 0.70, 1.001],
        labels=["Low", "Borderline", "High"],
    )

    value_column = (
        "historical_spend"
        if using_saved_scores
        else "predicted_clv"
    )
    if value_column not in leaderboard:
        leaderboard[value_column] = np.nan

    value_cutoff = leaderboard[value_column].quantile(0.80)
    leaderboard["value_tier"] = np.where(
        leaderboard[value_column] >= value_cutoff,
        "High",
        "Standard",
    )

    controls = st.columns(3)
    risk_filter = controls[0].selectbox(
        "Risk level",
        ["All risks", "High", "Borderline", "Low"],
        key="leaderboard_risk",
    )
    value_filter = controls[1].selectbox(
        "Value tier",
        ["All values", "High", "Standard"],
        key="leaderboard_value",
    )
    id_search = controls[2].text_input(
        "Find customer ID",
        key="leaderboard_search",
    )

    filtered = leaderboard.copy()

    if risk_filter != "All risks":
        filtered = filtered[filtered["risk_level"] == risk_filter]
    if value_filter != "All values":
        filtered = filtered[filtered["value_tier"] == value_filter]
    if id_search.strip():
        filtered = filtered[
            filtered["customer_id"]
            .astype(str)
            .str.contains(id_search.strip(), regex=False)
        ]

    high_risk_high_value = int(
        (
            (leaderboard["risk_level"] == "High")
            & (leaderboard["value_tier"] == "High")
        ).sum()
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Saved predictions", f"{len(leaderboard):,}")
    m2.metric(
        "High-risk customers",
        f"{(leaderboard['risk_level'] == 'High').sum():,}",
    )
    m3.metric(
        "High-value and high-risk",
        f"{high_risk_high_value:,}",
    )

    filtered = filtered.sort_values(
        "retention_priority",
        ascending=False,
        na_position="last",
    )
    filtered["value_amount"] = filtered[value_column]

    visible = filtered.head(100).rename(
        columns={
            "customer_id": "Customer ID",
            "risk_level": "Risk level",
            "churn_probability": "Churn probability",
            "value_amount": "Value estimate (\u00A3)",
            "retention_priority": "Priority score (\u00A3)",
            "value_tier": "Value tier",
            "churn_prediction": "Predicted churn",
        }
    )

    st.caption(
        "Risk probabilities are from the saved customer scores. Value is "
        + (
            "historical spend"
            if using_saved_scores
            else "the API-predicted CLV"
        )
        + "; the priority score multiplies risk by value."
    )

    st.dataframe(
        visible[
            [
                "Customer ID",
                "Risk level",
                "Value tier",
                "Churn probability",
                "Value estimate (\u00A3)",
                "Priority score (\u00A3)",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Churn probability": st.column_config.NumberColumn(
                format="percent"
            ),
            "Value estimate (\u00A3)": st.column_config.NumberColumn(
                format="\u00A3 %.2f"
            ),
            "Priority score (\u00A3)": st.column_config.NumberColumn(
                format="\u00A3 %.2f"
            ),
        },
    )

    if using_saved_scores:
        report = filtered[
            [
                "customer_id",
                "churn_prediction",
                "churn_probability",
                "historical_spend",
                "retention_priority",
            ]
        ].rename(
            columns={
                "customer_id": "Customer_ID",
                "churn_prediction": "Churn_Prediction",
                "churn_probability": "Churn_Probability",
                "historical_spend": "Historical_Spend",
                "retention_priority": "Risk_Weighted_Historical_Spend",
            }
        )
        st.download_button(
            "Download priority list (CSV)",
            report.to_csv(index=False).encode("utf-8"),
            "vantara_priority_customers.csv",
            "text/csv",
        )
    else:
        report = pd.DataFrame(
            {
                "Customer_ID": filtered["customer_id"].astype(str),
                "Churn_Prediction": filtered["churn_prediction"].astype(int),
                "Churn_Probability": filtered["churn_probability"],
                "90_Day_Purchase_Probability": (
                    1 - filtered["churn_probability"]
                ),
                "Predicted_CLV": filtered["predicted_clv"],
            }
        )

        # CSV export only: PDF report generation has been removed.
        st.download_button(
            "Download scored list (CSV)",
            report.to_csv(index=False).encode("utf-8"),
            "vantara_priority_customers.csv",
            "text/csv",
        )


# ---------------------------------------------------------------------
# Individual customer assessment
# ---------------------------------------------------------------------
section(
    "Individual assessment",
    "Score a customer",
    "Search by Customer ID, review the prediction, then request a local explanation.",
)

def format_customer_id(value: object) -> str:
    """Format numeric customer IDs consistently for search and API requests."""
    text = str(value)
    try:
        number = float(text)
        return str(int(number)) if number.is_integer() else text
    except (TypeError, ValueError):
        return text


ids = (
    data["Customer_ID"]
    .dropna()
    .map(format_customer_id)
    .unique()
    .tolist()
)

if not ids:
    st.warning("No valid customer IDs are available for individual scoring.")
    st.stop()

selected_id = st.selectbox(
    "Search Customer ID",
    ids,
    key="score_customer",
)

matching_rows = data[
    data["Customer_ID"].map(format_customer_id) == selected_id
]
row = matching_rows.iloc[0]


summary = st.columns(4)
summary[0].metric("Customer group", str(row.get("Segment profile", "-")))
summary[1].metric(
    "Historical spend",
    money(float(row.get("Monetary", 0))),
)
summary[2].metric(
    "Days since purchase",
    f"{float(row.get('Recency', 0)):.0f}",
)
summary[3].metric(
    "Current risk group",
    str(row.get("Risk_Level", "-")),
)


def make_features(customer: pd.Series) -> dict[str, float]:
    result: dict[str, float] = {}
    excluded = {
        "Customer_ID",
        "Churn",
        "Risk_Level",
        "Churn_Probability",
        "Segment",
        "Segment profile",
        "Country",
        "value_tier",
        "historical_monetary_value",
    }

    for column, value in customer.items():
        if column in excluded or pd.isna(value):
            continue
        try:
            result[column] = float(value)
        except (TypeError, ValueError):
            continue

    return result


features = make_features(row)
payload = {
    "customer_id": selected_id,
    "features": features,
}

if st.button(
    "Run prediction",
    type="primary",
    key="run_customer_prediction",
):
    try:
        response = requests.post(
            f"{API_URL}/prediction/predict",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        st.session_state["last_customer_prediction"] = {
            "customer_id": selected_id,
            "result": response.json(),
            "payload": payload,
        }
    except requests.RequestException as error:
        st.error(
            f"Could not reach the scoring API at {API_URL}. "
            "Start it with uvicorn api.main:app --reload. "
            f"Details: {error}"
        )

saved = st.session_state.get("last_customer_prediction")

if saved and saved["customer_id"] == selected_id:
    result = saved["result"]
    probability = float(result.get("churn", {}).get("probability", 0))
    clv_prediction = float(
        result.get("clv", {}).get("predicted_value", 0)
    )

    output = st.columns(4)
    output[0].metric("Churn probability", f"{probability:.1%}")
    output[1].metric(
        "Risk classification",
        "High"
        if probability >= 0.70
        else "Borderline"
        if probability >= 0.30
        else "Low",
    )
    output[2].metric(
        "90-day purchase chance",
        f"{float(result.get('next_purchase_probability', 0)):.1%}",
    )
    output[3].metric(
        "Predicted 90-day CLV",
        money(clv_prediction, 2),
    )

    recommendation_response = api_get(
        f"/customer/recommendations/{selected_id}"
    )
    recommendation_rows = pd.DataFrame(
        (recommendation_response or {}).get("recommendations", [])
    )

    if not recommendation_rows.empty:
        st.markdown("**Related product suggestions**")
        st.dataframe(
            recommendation_rows,
            use_container_width=True,
            hide_index=True,
        )

    explanation_method = st.selectbox(
        "Explanation method",
        ["SHAP", "LIME"],
        key="explanation_method",
    )

    if st.button(
        "Explain this prediction",
        key="explain_customer_prediction",
    ):
        try:
            explain_response = requests.post(
                f"{API_URL}/prediction/explain",
                params={"method": explanation_method.lower()},
                json=saved["payload"],
                timeout=60,
            )
            explain_response.raise_for_status()

            contributions = pd.DataFrame(
                explain_response.json().get("contributions", [])
            )

            if contributions.empty:
                st.info("No feature contributions were returned.")
            else:
                contributions["impact"] = pd.to_numeric(
                    contributions["impact"],
                    errors="coerce",
                )
                contributions = contributions.dropna(
                    subset=["impact"]
                ).sort_values(
                    "impact",
                    key=lambda series: series.abs(),
                )
                contributions["Effect"] = np.where(
                    contributions["impact"] >= 0,
                    "Raises churn score",
                    "Lowers churn score",
                )

                fig = px.bar(
                    contributions,
                    x="impact",
                    y="feature",
                    orientation="h",
                    color="Effect",
                    color_discrete_map={
                        "Raises churn score": PINK,
                        "Lowers churn score": TEAL,
                    },
                    labels={
                        "impact": "Contribution to churn score",
                        "feature": "Feature",
                        "Effect": "Effect",
                    },
                    hover_data=[
                        column
                        for column in ["value"]
                        if column in contributions
                    ],
                )
                fig.update_layout(
                    yaxis_title=None,
                    legend={"orientation": "h", "y": 1.1, "x": 0},
                )
                fig.update_traces(marker_line_width=0)

                st.plotly_chart(
                    chart_style(fig, 330),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

        except requests.RequestException as error:
            detail = (
                error.response.text
                if error.response is not None
                else str(error)
            )
            st.warning(
                f"{explanation_method} explanation is unavailable. "
                "Check the optional explanation packages and API status. "
                f"{detail}"
            )


# ---------------------------------------------------------------------
# Model evaluation and comparison
# ---------------------------------------------------------------------
section(
    "Evaluation",
    "Model performance",
    "Saved test-set results. These describe the existing evaluation, not a newly trained run.",
)

tuned_results = load_csv(
    str(MODEL_RESULTS_DIR / "tuned_models_final_test.csv")
)
churn_results = load_csv(
    str(MODEL_RESULTS_DIR / "final_test_results.csv")
)
clv_results = load_csv(
    str(MODEL_RESULTS_DIR / "clv_final_comparison.csv")
)
experiment_results = load_csv(
    str(MODEL_RESULTS_DIR / "model_selection_results.csv")
)

other_results = (
    experiment_results[
        experiment_results.get(
            "Task",
            pd.Series(dtype=str),
        ).isin(["Regression", "Anomaly Detection"])
    ]
    if not experiment_results.empty
    else pd.DataFrame()
)

test_tab, tuned_tab, clv_tab, other_tab = st.tabs(
    [
        "Churn model comparison",
        "Tuned models",
        "Customer value models",
        "Sequence & anomaly",
    ]
)

with test_tab:
    st.markdown("**Held-out churn test results**")
    st.caption(
        "All evaluated classifiers are listed with available performance "
        "metrics and evaluation time."
    )

    if not churn_results.empty:
        model_col = next(
            (
                column
                for column in churn_results
                if column.lower()
                in {"model", "model_name", "classifier"}
            ),
            None,
        )
        auc_col = next(
            (
                column
                for column in churn_results
                if column.lower().replace("-", "_")
                in {"roc_auc", "auc"}
            ),
            None,
        )

        if model_col and auc_col:
            chart = churn_results[[model_col, auc_col]].copy()
            chart[auc_col] = pd.to_numeric(
                chart[auc_col],
                errors="coerce",
            )
            chart = chart.dropna().sort_values(auc_col)

            fig = px.bar(
                chart,
                x=auc_col,
                y=model_col,
                orientation="h",
                color_discrete_sequence=[TEAL],
                labels={
                    auc_col: "ROC-AUC",
                    model_col: "Model",
                },
            )
            fig.update_xaxes(range=[0, 1], tickformat=".0%")
            fig.update_layout(showlegend=False, yaxis_title=None)

            st.plotly_chart(
                chart_style(fig, 320),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        percent_columns = {
            name: st.column_config.NumberColumn(format="percent")
            for name in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC")
            if name in churn_results
        }

        st.dataframe(
            churn_results,
            use_container_width=True,
            hide_index=True,
            column_config=percent_columns,
        )
    else:
        st.info("Churn evaluation file is not available.")

with tuned_tab:
    st.markdown("**Tuned churn model results**")
    st.caption(
        "Shown separately from the broader held-out comparison because "
        "these scores come from a separate tuning run."
    )

    if not tuned_results.empty:
        percent_columns = {
            name: st.column_config.NumberColumn(format="percent")
            for name in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC")
            if name in tuned_results
        }

        st.dataframe(
            tuned_results,
            use_container_width=True,
            hide_index=True,
            column_config=percent_columns,
        )
    else:
        st.info("Tuned-model results are not available.")

with clv_tab:
    st.markdown("**Customer value regression results**")
    st.caption("Regression models are reported with MAE, RMSE, and R\u00B2.")

    if not clv_results.empty:
        model_col = next(
            (
                column
                for column in clv_results
                if column.lower()
                in {"model", "model_name", "estimator"}
            ),
            None,
        )
        r2_col = next(
            (
                column
                for column in clv_results
                if column.lower().replace("_", "")
                in {"r2", "r\u00B2", "r2score"}
            ),
            None,
        )

        if model_col and r2_col:
            chart = clv_results[[model_col, r2_col]].copy()
            chart[r2_col] = pd.to_numeric(
                chart[r2_col],
                errors="coerce",
            )
            chart = chart.dropna().sort_values(r2_col)

            fig = px.bar(
                chart,
                x=r2_col,
                y=model_col,
                orientation="h",
                color_discrete_sequence=[BLUE],
                labels={
                    r2_col: "R\u00B2",
                    model_col: "Model",
                },
            )
            fig.update_layout(showlegend=False, yaxis_title=None)

            st.plotly_chart(
                chart_style(fig, 280),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        st.dataframe(
            clv_results,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Customer value evaluation file is not available.")

with other_tab:
    st.markdown("**Sequence and anomaly model results**")
    st.caption(
        "These experiment metrics are separate from the final churn and CLV "
        "test tables. The anomaly detector has no labeled test metric in "
        "the saved results."
    )

    if not other_results.empty:
        columns = [
            name
            for name in (
                "Model",
                "Task",
                "MAE",
                "RMSE",
                "R2",
                "Training Time",
            )
            if name in other_results
        ]
        st.dataframe(
            other_results[columns],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "Sequence and anomaly experiment results are not available."
        )

shap_image = EXPLAINABILITY_DIR / "shap_global_summary.png"

if shap_image.is_file():
    with st.expander("Global model drivers"):
        st.image(
            str(shap_image),
            caption="Global SHAP feature importance across the evaluation sample",
            use_container_width=True,
        )

st.caption(
    "Currency is GBP (\u00A3), matching the Online Retail II transaction data. "
    "Revenue forecasting and product suggestions are baseline methods."
)

st.markdown(
    """
    <div class="author-footer">
        Vantara Customer Intelligence Platform
        <br>
        Created by <span class="author-name">Ritul Gaikwad</span>
        <br>
        Customer Analytics · Machine Learning · Explainable AI
    </div>
    """,
    unsafe_allow_html=True,
)
