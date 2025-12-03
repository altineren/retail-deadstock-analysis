"""Train a baseline model for dead stock risk classification."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    BOOLEAN_FEATURES,
    prepare_dataset,
    train_test_split,
)

MODEL_PATH = PROJECT_ROOT / "models" / "dead_stock_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
RAW_DATA_PATH = PROJECT_ROOT / "retail_store_inventory.csv"


def build_pipeline() -> Pipeline:
    numeric_features = NUMERIC_FEATURES + [
        "rolling_sales_7",
        "rolling_sales_30",
        "rolling_inventory_30",
        "sell_through_rate",
        "days_since_sale",
        "inventory_trend",
        "forecast_error",
        "order_gap",
        "risk_score",
        "day_of_year",
        "month",
    ]
    boolean_features = BOOLEAN_FEATURES + ["is_weekend"]
    categorical_features = CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    bool_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("bool", bool_transformer, boolean_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
    return model


def main() -> None:
    prepared = prepare_dataset(RAW_DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(prepared, test_size=0.2)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)
    metrics["roc_auc"] = roc_auc

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(
        json.dumps(
            {
                "precision": metrics["1"]["precision"],
                "recall": metrics["1"]["recall"],
                "f1": metrics["1"]["f1-score"],
                "support": metrics["1"]["support"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
