from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .data_loader import load_and_prepare_dataset, load_raw_dataset
from .utils import DEFAULT_RAW_DATA_PATH, summarize_dataframe


def load_dataset(csv_path: str | Path | None = None, preserve_raw_copy: bool = True) -> pd.DataFrame:
    return load_raw_dataset(csv_path=csv_path, preserve_copy=preserve_raw_copy)


def audit_dataset(df: pd.DataFrame) -> dict[str, Any]:
    print("\nDataset schema")
    print("--------------")
    print(df.dtypes.to_string())
    print("\nColumn names")
    print("------------")
    print(list(df.columns))
    missing = df.isna().sum().sort_values(ascending=False)
    print("\nMissing values")
    print("---------------")
    print(missing.to_string())
    duplicates = int(df.duplicated().sum())
    print("\nDuplicate rows")
    print("--------------")
    print(duplicates)
    print("\nBasic summary stats")
    print("-------------------")
    print(df.describe(include="all").transpose())
    summary = summarize_dataframe(df)
    summary["missing_by_column"] = missing.to_dict()
    summary["summary_stats"] = df.describe(include="all").transpose().reset_index().to_dict(orient="records")
    return summary


def load_and_audit(csv_path: str | Path | None = None, preserve_raw_copy: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = load_dataset(csv_path=csv_path, preserve_raw_copy=preserve_raw_copy)
    summary = audit_dataset(df)
    return df, summary


def load_and_prepare(csv_path: str | Path | None = None) -> Any:
    return load_and_prepare_dataset(csv_path=csv_path)
