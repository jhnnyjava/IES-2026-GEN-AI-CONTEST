from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import (
    CATEGORICAL_COLUMNS,
    CLEANED_DATA_PATH,
    METADATA_COLUMNS,
    TARGET_COLUMNS,
    ensure_project_dirs,
    safe_to_numeric,
)


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder that works across scikit-learn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - compatibility fallback for older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def drop_unusable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove map metadata and other non-predictive fields."""

    columns_to_drop = [column for column in METADATA_COLUMNS if column in df.columns]
    return df.drop(columns=columns_to_drop, errors="ignore")


def clean_maize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize names, convert numerics safely, and keep only usable fields."""

    if df.empty:
        raise ValueError("Cannot preprocess an empty dataframe.")

    working = df.copy()
    working.columns = [str(column).strip().lower().replace(" ", "_") for column in working.columns]
    working = drop_unusable_columns(working)

    for column in working.columns:
        if column in CATEGORICAL_COLUMNS:
            working[column] = working[column].astype("string").str.strip().fillna("unknown")
        elif column in TARGET_COLUMNS:
            working[column] = safe_to_numeric(working[column])
        elif pd.api.types.is_numeric_dtype(working[column]):
            working[column] = pd.to_numeric(working[column], errors="coerce")
        else:
            numeric_version = safe_to_numeric(working[column])
            if numeric_version.notna().sum() > 0:
                working[column] = numeric_version
            else:
                working[column] = working[column].astype("string").str.strip().fillna("unknown")

    working = working.replace([np.inf, -np.inf], np.nan)
    working = working.drop_duplicates().reset_index(drop=True)

    if working.empty:
        raise ValueError("Preprocessing removed all rows from the dataset.")

    return working


def save_cleaned_dataset(df: pd.DataFrame, output_path: str | Path = CLEANED_DATA_PATH) -> Path:
    """Persist the cleaned dataset for traceability and later inspection."""

    ensure_project_dirs()
    output = Path(output_path)
    df.to_csv(output, index=False)
    return output


def prepare_task_data(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split a cleaned dataframe into model features and a numeric target."""

    target_column = target_column.lower()
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' was not found in the cleaned dataset.")

    feature_columns = [column for column in df.columns if column not in set(TARGET_COLUMNS) and column != target_column]
    if not feature_columns:
        raise ValueError(f"No usable feature columns remain for target '{target_column}'.")

    task_frame = df.loc[:, feature_columns + [target_column]].copy()
    task_frame[target_column] = safe_to_numeric(task_frame[target_column])
    task_frame = task_frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[target_column])

    if task_frame.empty:
        raise ValueError(f"No valid rows remain after cleaning target '{target_column}'.")

    X = task_frame[feature_columns].copy()
    y = task_frame[target_column].astype(float)

    return X, y, feature_columns


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical columns for the preprocessing pipeline."""

    categorical_features = [
        column
        for column in X.columns
        if column in CATEGORICAL_COLUMNS or pd.api.types.is_object_dtype(X[column]) or pd.api.types.is_string_dtype(X[column])
    ]
    numeric_features = [column for column in X.columns if column not in categorical_features]
    return numeric_features, categorical_features


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a reusable preprocessing pipeline for classical ML models."""

    numeric_features, categorical_features = infer_feature_types(X)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def build_model_metadata(
    target_column: str,
    feature_columns: list[str],
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """Create a lightweight metadata payload for inference and reporting."""

    numeric_features, categorical_features = infer_feature_types(X)
    return {
        "target": target_column,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "rows": int(len(X)),
        "target_non_null_rows": int(y.notna().sum()),
        "target_zero_share": float((y == 0).mean()) if len(y) else 0.0,
    }
