"""Utility functions for preparing inventory data and labeling dead stock risk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Demand Forecast",
    "Price",
    "Discount",
    "Competitor Pricing",
]

CATEGORICAL_FEATURES = [
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality",
]

BOOLEAN_FEATURES = ["Holiday/Promotion"]


@dataclass
class PreparedData:
    features: pd.DataFrame
    labels: pd.Series
    raw: pd.DataFrame


def _days_since_last_sale(values: pd.Series) -> pd.Series:
    """Return number of days since the last non-zero sale for each row."""
    counter = 0
    result = []
    for value in values:
        counter = 0 if value > 0 else counter + 1
        result.append(counter)
    return pd.Series(result, index=values.index)


def load_data(source: Union[Path, str, pd.DataFrame]) -> pd.DataFrame:
    """Load inventory CSV with parsed dates."""
    if isinstance(source, (str, Path)):
        df = pd.read_csv(source, parse_dates=["Date"])
    elif isinstance(source, pd.DataFrame):
        df = source.copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        else:
            raise ValueError("DataFrame must include a 'Date' column.")
    else:
        raise TypeError(f"Unsupported data source type: {type(source)}")
    df.sort_values(["Store ID", "Product ID", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling metrics and engineered signals for modeling."""
    group_cols = ["Store ID", "Product ID"]
    grouped = df.groupby(group_cols, sort=False)

    df["rolling_sales_7"] = grouped["Units Sold"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    df["rolling_sales_30"] = grouped["Units Sold"].transform(
        lambda s: s.rolling(30, min_periods=1).mean()
    )
    df["rolling_inventory_30"] = grouped["Inventory Level"].transform(
        lambda s: s.rolling(30, min_periods=1).mean()
    )
    df["sell_through_rate"] = df["rolling_sales_30"] / (df["rolling_inventory_30"] + 1e-3)
    df["days_since_sale"] = grouped["Units Sold"].transform(_days_since_last_sale)
    df["inventory_trend"] = grouped["Inventory Level"].transform(
        lambda s: s.pct_change().fillna(0.0)
    )
    df["forecast_error"] = (df["Units Sold"] - df["Demand Forecast"]) / (
        df["Demand Forecast"] + 1e-3
    )
    df["order_gap"] = df["Units Ordered"] - df["Units Sold"]

    # Normalize key numeric signals to 0-1 for simple risk scoring heuristics.
    for col in ["rolling_inventory_30", "days_since_sale", "sell_through_rate"]:
        norm_col = f"{col}_norm"
        values = df[col].to_numpy()
        if col == "sell_through_rate":
            # Higher sell-through = lower risk, but keep normalized value for interpretability.
            max_val = np.nanmax(values) or 1.0
            df[norm_col] = np.nan_to_num(values / max_val)
            continue
        min_val = np.nanmin(values)
        span = np.nanmax(values) - min_val
        df[norm_col] = np.nan_to_num((values - min_val) / (span if span else 1.0))

    return df


def label_dead_stock(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary labels and heuristic risk scores."""
    high_inventory = df["rolling_inventory_30"] > df["rolling_inventory_30"].quantile(
        0.45
    )
    slow_turnover = df["rolling_sales_7"] <= (0.1 * df["rolling_inventory_30"])
    weak_sell_through = df["sell_through_rate"] < 0.3
    overstock_vs_forecast = (df["Inventory Level"] - df["Demand Forecast"]) > 50
    no_sales_today = df["Units Sold"] == 0
    low_units_sold = df["Units Sold"] <= 5
    df["dead_stock_label"] = (
        (high_inventory & slow_turnover)
        | (overstock_vs_forecast & weak_sell_through)
        | (high_inventory & no_sales_today)
        | (high_inventory & low_units_sold)
    ).astype(int)

    # Risk score combines normalized factors (higher = riskier).
    df["risk_score"] = (
        0.4 * df["rolling_inventory_30_norm"]
        + 0.3 * (1 - df["sell_through_rate_norm"])
        + 0.3 * np.nan_to_num(
            1
            - (
                df["rolling_sales_7"] / (df["rolling_inventory_30"] + 1e-3)
            ).clip(0, 1)
        )
    ).clip(0, 1)

    return df


def prepare_dataset(source: Union[Path, str, pd.DataFrame]) -> PreparedData:
    """Full pipeline for loading data, engineering features, and labeling."""
    df = load_data(source)
    df = engineer_features(df)
    df = label_dead_stock(df)

    feature_cols = [
        "Date",
        *NUMERIC_FEATURES,
        *BOOLEAN_FEATURES,
        *CATEGORICAL_FEATURES,
        "rolling_sales_7",
        "rolling_sales_30",
        "rolling_inventory_30",
        "sell_through_rate",
        "days_since_sale",
        "inventory_trend",
        "forecast_error",
        "order_gap",
        "risk_score",
    ]

    features = df[feature_cols].copy()
    features["day_of_year"] = features["Date"].dt.dayofyear
    features["month"] = features["Date"].dt.month
    features["is_weekend"] = features["Date"].dt.weekday >= 5
    features.drop(columns=["Date"], inplace=True)
    labels = df["dead_stock_label"].copy()

    return PreparedData(features=features, labels=labels, raw=df.copy())


def train_test_split(
    data: PreparedData, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chronological split to avoid leakage."""
    df = data.features.assign(
        dead_stock=data.labels, Date=data.raw["Date"].values
    )
    df.sort_values("Date", inplace=True)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    feature_cols = [col for col in train_df.columns if col not in {"dead_stock", "Date"}]
    return (
        train_df[feature_cols],
        test_df[feature_cols],
        train_df["dead_stock"],
        test_df["dead_stock"],
    )
