from __future__ import annotations

import shutil
from pathlib import Path
from typing import Tuple

import pandas as pd

from .utils import DEFAULT_RAW_DATA_PATH, RAW_COPY_PATH, ensure_project_dirs, summarize_dataframe


def load_dataset(csv_path: str | Path | None = None, preserve_raw_copy: bool = True) -> pd.DataFrame:
    """Load the maize production CSV from the configured path."""

    ensure_project_dirs()
    path = Path(csv_path) if csv_path is not None else DEFAULT_RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the Kenya maize CSV there or pass --data-path."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset at {path} is empty.")

    if preserve_raw_copy:
        shutil.copy2(path, RAW_COPY_PATH)

    return df


def audit_dataset(df: pd.DataFrame) -> dict:
    """Print a compact schema and quality audit for research traceability."""

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


def load_and_audit(csv_path: str | Path | None = None, preserve_raw_copy: bool = True) -> Tuple[pd.DataFrame, dict]:
    """Load the dataset and print a quality audit in one step."""

    df = load_dataset(csv_path=csv_path, preserve_raw_copy=preserve_raw_copy)
    summary = audit_dataset(df)
    return df, summary
