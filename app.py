"""Streamlit app for dead stock risk analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from src.data_pipeline import PreparedData, prepare_dataset

DATA_PATH = Path("retail_store_inventory.csv")
MODEL_PATH = Path("models/dead_stock_model.joblib")
METRICS_PATH = Path("models/metrics.json")


st.set_page_config(
    page_title="Dead Stock Risk Radar",
    layout="wide",
    page_icon="📦",
)


@st.cache_data(show_spinner=False)
def load_raw_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Date"])


@st.cache_resource(show_spinner=False)
def load_model() -> joblib:
    return joblib.load(MODEL_PATH)


def build_dataset(df: pd.DataFrame) -> PreparedData:
    return prepare_dataset(df)


def run_inference(prepared: PreparedData) -> pd.DataFrame:
    model = load_model()
    probabilities = model.predict_proba(prepared.features)[:, 1]
    result = prepared.raw.copy()
    result["model_probability"] = probabilities
    result["predicted_label"] = (probabilities >= 0.5).astype(int)
    return result


def generate_recommendation(row: pd.Series) -> str:
    actions = []
    if row["model_probability"] >= 0.75:
        actions.append("Launch clearance promo")
    elif row["model_probability"] >= 0.6:
        actions.append("Bundle w/ fast movers")
    if row["Units Ordered"] > row["Units Sold"]:
        actions.append("Freeze re-orders")
    if row["Discount"] == 0 and row["Inventory Level"] > row["Demand Forecast"]:
        actions.append("Introduce markdown")
    if not actions:
        actions.append("Monitor weekly trend")
    return "; ".join(actions)


def filter_data(
    df: pd.DataFrame,
    stores: list[str],
    categories: list[str],
    date_range: Tuple[pd.Timestamp, pd.Timestamp],
) -> pd.DataFrame:
    start, end = date_range
    mask = (
        df["Store ID"].isin(stores)
        & df["Category"].isin(categories)
        & (df["Date"].between(start, end))
    )
    return df.loc[mask].copy()


def main() -> None:
    st.title("📦 Dead Stock Risk Radar")
    st.caption("Predictive AI assistant for proactive inventory management.")

    uploaded_file = st.sidebar.file_uploader(
        "Upload inventory CSV",
        type=["csv"],
        help="Optional: use your own dataset with the same schema.",
    )

    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    else:
        raw_df = load_raw_data(DATA_PATH)

    prepared = build_dataset(raw_df)
    dataset = run_inference(prepared)

    st.sidebar.subheader("Filters")
    store_options = sorted(dataset["Store ID"].unique().tolist())
    category_options = sorted(dataset["Category"].unique().tolist())

    selected_stores = st.sidebar.multiselect(
        "Stores", store_options, default=store_options
    )
    selected_categories = st.sidebar.multiselect(
        "Categories", category_options, default=category_options
    )
    min_date, max_date = dataset["Date"].min(), dataset["Date"].max()
    date_range = st.sidebar.date_input(
        "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
    )
    if isinstance(date_range, tuple):
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(
            date_range[1]
        )
    else:
        start_date = min_date
        end_date = pd.to_datetime(date_range)
    risk_threshold = st.sidebar.slider(
        "Risk alert threshold",
        min_value=0.3,
        max_value=0.9,
        value=0.6,
        step=0.05,
    )
    top_k = st.sidebar.slider("Top risky SKUs", min_value=5, max_value=50, value=15)

    filtered = filter_data(
        dataset,
        selected_stores if selected_stores else store_options,
        selected_categories if selected_categories else category_options,
        (start_date, end_date),
    )

    high_risk = filtered[filtered["model_probability"] >= risk_threshold]
    risk_ratio = (len(high_risk) / len(filtered)) if len(filtered) else 0
    avg_sell_through = (
        filtered["Units Sold"].sum() / filtered["Inventory Level"].sum()
        if filtered["Inventory Level"].sum()
        else 0
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Inventory at risk",
        f"{high_risk['Inventory Level'].sum():,.0f} units",
        help="Sum of on-hand units flagged above the alert threshold.",
    )
    col2.metric(
        "% SKUs high risk",
        f"{risk_ratio*100:,.1f}%",
        help="Share of filtered records exceeding the alert threshold.",
    )
    col3.metric(
        "Avg sell-through",
        f"{avg_sell_through*100:,.1f}%",
        help="Sell-through (units sold / inventory) for filtered data.",
    )

    st.markdown("---")
    st.subheader("Trend overview")
    trend_df = (
        filtered.groupby("Date")["model_probability"]
        .mean()
        .reset_index()
        .rename(columns={"model_probability": "avg_risk"})
    )
    st.line_chart(trend_df, x="Date", y="avg_risk", height=250)
    st.caption("Daily average risk across the filtered inventory highlights seasonal spikes or stabilization.")

    st.subheader("Risk by category")
    category_pivot = (
        filtered.groupby("Category")["model_probability"].mean().sort_values()
    )
    st.bar_chart(category_pivot)
    st.caption("Average model probability per category surfaces assortments driving the bulk of dead stock exposure.")

    st.subheader("Store vs category heatmap")
    heatmap_df = (
        filtered.groupby(["Store ID", "Category"])["model_probability"]
        .mean()
        .reset_index()
    )
    if not heatmap_df.empty:
        heatmap_chart = (
            alt.Chart(heatmap_df)
            .mark_rect()
            .encode(
                x=alt.X("Category:N", title="Category"),
                y=alt.Y("Store ID:N", title="Store"),
                color=alt.Color(
                    "model_probability:Q",
                    title="Avg Risk",
                    scale=alt.Scale(scheme="reds"),
                ),
                tooltip=[
                    alt.Tooltip("Store ID:N"),
                    alt.Tooltip("Category:N"),
                    alt.Tooltip("model_probability:Q", format=".2f", title="Avg risk"),
                ],
            )
            .properties(height=240)
        )
        st.altair_chart(heatmap_chart, use_container_width=True)
        st.caption(
            "Heatmap compares average risk across store-category pairs to spot localized build-ups."
        )

    st.subheader("Inventory vs. risk scatter")
    scatter_df = filtered.copy()
    if not scatter_df.empty:
        scatter_chart = (
            alt.Chart(scatter_df.sample(min(len(scatter_df), 1000)))
            .mark_circle(opacity=0.7)
            .encode(
                x=alt.X("Inventory Level:Q", title="Inventory Level"),
                y=alt.Y(
                    "model_probability:Q", title="Model Probability", scale=alt.Scale()
                ),
                size=alt.Size("Units Sold:Q", title="Units Sold (bubble size)"),
                color=alt.Color("Category:N", title="Category"),
                tooltip=[
                    alt.Tooltip("Date:T"),
                    alt.Tooltip("Store ID:N"),
                    alt.Tooltip("Product ID:N"),
                    alt.Tooltip("Inventory Level:Q"),
                    alt.Tooltip("Units Sold:Q"),
                    alt.Tooltip("model_probability:Q", format=".2f"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(scatter_chart, use_container_width=True)
        st.caption(
            "Bubble plot reveals which SKUs combine high on-hand units with elevated risk so planners can act first."
        )

    st.subheader("Top risky SKUs")
    top_risky = (
        high_risk.sort_values("model_probability", ascending=False)
        .head(top_k)
        .copy()
    )
    if not top_risky.empty:
        top_risky["Recommendation"] = top_risky.apply(generate_recommendation, axis=1)
        display_cols = [
            "Date",
            "Store ID",
            "Product ID",
            "Category",
            "Inventory Level",
            "Units Sold",
            "Demand Forecast",
            "model_probability",
            "risk_score",
            "Recommendation",
        ]
        st.dataframe(
            top_risky[display_cols].style.format(
                {"model_probability": "{:.2f}", "risk_score": "{:.2f}"}
            ),
            use_container_width=True,
        )
        csv_data = top_risky[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download recommendations (CSV)",
            csv_data,
            file_name="dead_stock_risk_insights.csv",
            mime="text/csv",
        )
    else:
        st.info("No records exceed the current risk threshold for the selected filters.")

    st.markdown("---")
    st.subheader("Model performance snapshot")
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        class_one = metrics["1"]
        st.write(
            f"Precision: **{class_one['precision']:.2f}** | "
            f"Recall: **{class_one['recall']:.2f}** | "
            f"F1-score: **{class_one['f1-score']:.2f}** | "
            f"ROC AUC: **{metrics['roc_auc']:.2f}**"
        )
    else:
        st.warning("Train the model to view performance metrics.")

    st.caption(
        "Upload a new CSV or tweak filters to simulate scenarios. "
        "Powered by RandomForest classifier + engineered sell-through features."
    )


if __name__ == "__main__":
    main()
