from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import CATEGORICAL_FEATURE_COLUMNS, ENVIRONMENTAL_FEATURE_COLUMNS, MODEL_INPUT_COLUMNS, TARGET_COLUMNS, TARGET_PRIMARY, ensure_project_dirs, safe_to_numeric


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - compatibility fallback
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def standardize_maize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and retain only the columns required for modeling."""

    if df.empty:
        raise ValueError("Cannot preprocess an empty dataframe.")

    working = df.copy()
    working.columns = [str(column).strip().lower().replace(" ", "_") for column in working.columns]

    for column in TARGET_COLUMNS:
        if column in working.columns:
            working[column] = safe_to_numeric(working[column])
    if "areaharv" in working.columns:
        working["areaharv"] = safe_to_numeric(working["areaharv"])
    for column in CATEGORICAL_FEATURE_COLUMNS:
        if column in working.columns:
            working[column] = working[column].astype("string").str.strip().fillna("unknown")
    for column in ENVIRONMENTAL_FEATURE_COLUMNS:
        if column in working.columns:
            working[column] = safe_to_numeric(working[column])

    keep_columns = [column for column in MODEL_INPUT_COLUMNS + TARGET_COLUMNS if column in working.columns]
    if not keep_columns:
        raise ValueError("No usable columns remain after standardization.")

    working = working.loc[:, keep_columns]
    working = working.replace([np.inf, -np.inf], np.nan)
    working = working.drop_duplicates().reset_index(drop=True)
    return working


def save_cleaned_dataset(df: pd.DataFrame, output_path: str | Path) -> Path:
    ensure_project_dirs()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return output


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_features = [
        column
        for column in X.columns
        if column in CATEGORICAL_FEATURE_COLUMNS or pd.api.types.is_object_dtype(X[column]) or pd.api.types.is_string_dtype(X[column])
    ]
    numeric_features = [column for column in X.columns if column not in categorical_features]
    return numeric_features, categorical_features


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features, categorical_features = infer_feature_types(X)
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _make_one_hot_encoder())])
    return ColumnTransformer([("numeric", numeric_transformer, numeric_features), ("categorical", categorical_transformer, categorical_features)], remainder="drop")


def split_features_target(df: pd.DataFrame, target_column: str = TARGET_PRIMARY) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    target_column = target_column.lower()
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' was not found in the dataset.")

    feature_columns = [column for column in df.columns if column not in set(TARGET_COLUMNS) and column != target_column]
    if not feature_columns:
        raise ValueError(f"No usable feature columns remain for target '{target_column}'.")

    task_frame = df.loc[:, feature_columns + [target_column]].copy()
    task_frame[target_column] = safe_to_numeric(task_frame[target_column])
    task_frame = task_frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[target_column])
    if task_frame.empty:
        raise ValueError(f"No valid rows remain after cleaning target '{target_column}'.")

    return task_frame[feature_columns].copy(), task_frame[target_column].astype(float), feature_columns


def ensure_required_features(df: pd.DataFrame) -> None:
    missing = [column for column in MODEL_INPUT_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Dataset is missing required model input columns: {missing}")
