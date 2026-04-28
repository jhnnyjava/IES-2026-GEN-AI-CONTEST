from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .feature_engineering import augment_and_persist
from .utils import (
    BASE_FEATURE_COLUMNS,
    CLEANED_DATA_PATH,
    DEFAULT_RAW_DATA_PATH,
    EXPECTED_SOURCE_COLUMNS,
    METADATA_COLUMNS,
    RAW_COPY_PATH,
    RANDOM_STATE,
    TARGET_COLUMNS,
    clean_column_names,
    ensure_project_dirs,
    safe_to_numeric,
    save_json,
    save_text,
    summarize_dataframe,
)


@dataclass(slots=True)
class DatasetBundle:
    raw: pd.DataFrame
    cleaned: pd.DataFrame
    augmented: pd.DataFrame
    raw_summary: dict[str, Any]
    cleaned_summary: dict[str, Any]
    augmented_summary: dict[str, Any]
    raw_path: Path
    cleaned_path: Path
    augmented_path: Path


def load_raw_dataset(csv_path: str | Path | None = None, preserve_copy: bool = True) -> pd.DataFrame:
    """Load the primary Kenya maize CSV."""

    ensure_project_dirs()
    path = Path(csv_path) if csv_path is not None else DEFAULT_RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Place the Kenya maize CSV there or pass --data-path.")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset at {path} is empty.")

    if preserve_copy:
        RAW_COPY_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, RAW_COPY_PATH)

    return df


def validate_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Validate required fields and report schema alignment."""

    normalized = clean_column_names(df)
    missing_required = [column for column in BASE_FEATURE_COLUMNS + TARGET_COLUMNS if column not in normalized.columns]
    unexpected = [column for column in normalized.columns if column not in EXPECTED_SOURCE_COLUMNS]
    return {
        "missing_required_columns": missing_required,
        "unexpected_columns": unexpected,
        "row_count": int(len(normalized)),
        "column_count": int(len(normalized.columns)),
    }


def clean_maize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names, keep research-relevant fields, and coerce numeric types."""

    if df.empty:
        raise ValueError("Cannot clean an empty dataframe.")

    working = clean_column_names(df).copy()
    keep_columns = [column for column in working.columns if column in EXPECTED_SOURCE_COLUMNS - METADATA_COLUMNS | set(BASE_FEATURE_COLUMNS) | set(TARGET_COLUMNS)]
    working = working.loc[:, [column for column in working.columns if column in keep_columns]]

    for column in BASE_FEATURE_COLUMNS:
        if column in working.columns and column != "areaharv":
            working[column] = working[column].astype("string").str.strip().fillna("unknown")
    if "areaharv" in working.columns:
        working["areaharv"] = safe_to_numeric(working["areaharv"])
    for column in TARGET_COLUMNS:
        if column in working.columns:
            working[column] = safe_to_numeric(working[column])

    working = working.replace([pd.NA, float("inf"), float("-inf")], pd.NA)
    working = working.drop_duplicates().reset_index(drop=True)

    if working.empty:
        raise ValueError("Cleaning removed all rows from the dataset.")

    return working


def clip_numeric_outliers(df: pd.DataFrame, columns: list[str], whisker: float = 1.5) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Clip numeric outliers using an IQR rule for reporting stability."""

    cleaned = df.copy()
    summary: dict[str, dict[str, float]] = {}
    for column in columns:
        if column not in cleaned.columns:
            continue
        series = pd.to_numeric(cleaned[column], errors="coerce")
        if series.dropna().empty:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - whisker * iqr
        upper = q3 + whisker * iqr
        cleaned[column] = series.clip(lower=lower, upper=upper)
        summary[column] = {"q1": q1, "q3": q3, "lower": lower, "upper": upper}
    return cleaned, summary


def save_dataset_profile(raw: pd.DataFrame, cleaned: pd.DataFrame, augmented: pd.DataFrame, raw_path: Path, cleaned_path: Path, augmented_path: Path, profile_path: str | Path) -> Path:
    """Write a concise dataset profile for reporting and reproducibility."""

    profile = [
        "# Dataset Profile",
        "",
        f"- Raw rows: {len(raw)}",
        f"- Cleaned rows: {len(cleaned)}",
        f"- Augmented rows: {len(augmented)}",
        f"- Raw copy: {raw_path}",
        f"- Cleaned file: {cleaned_path}",
        f"- Augmented file: {augmented_path}",
        "",
        "## Raw Schema Summary",
        str(summarize_dataframe(raw)),
        "",
        "## Cleaned Schema Summary",
        str(summarize_dataframe(cleaned)),
        "",
        "## Augmented Schema Summary",
        str(summarize_dataframe(augmented)),
    ]
    path = Path(profile_path)
    save_text(path, "\n".join(profile) + "\n")
    return path


def load_and_prepare_dataset(csv_path: str | Path | None = None, seed: int = RANDOM_STATE) -> DatasetBundle:
    """Load, validate, clean, augment, and persist all dataset stages."""

    raw = load_raw_dataset(csv_path=csv_path, preserve_copy=True)
    raw_summary = summarize_dataframe(raw)
    validation = validate_schema(raw)

    cleaned = clean_maize_dataset(raw)
    cleaned, outlier_summary = clip_numeric_outliers(cleaned, columns=["areaharv", "totmazprod", "mazyield"])
    cleaned_path = Path(CLEANED_DATA_PATH)
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(cleaned_path, index=False)
    cleaned_summary = summarize_dataframe(cleaned)
    cleaned_summary.update(validation)
    cleaned_summary["outlier_summary"] = outlier_summary

    augmented, augmented_path, augmentation_summary = augment_and_persist(cleaned, seed=seed)
    augmented_summary = summarize_dataframe(augmented)
    augmented_summary.update(augmentation_summary)

    profile_path = save_dataset_profile(raw, cleaned, augmented, RAW_COPY_PATH, cleaned_path, augmented_path, profile_path=Path("reports") / "dataset_profile.md")
    save_json(profile_path.with_suffix(".json"), {"raw": raw_summary, "cleaned": cleaned_summary, "augmented": augmented_summary})

    return DatasetBundle(
        raw=raw,
        cleaned=cleaned,
        augmented=augmented,
        raw_summary=raw_summary,
        cleaned_summary=cleaned_summary,
        augmented_summary=augmented_summary,
        raw_path=RAW_COPY_PATH,
        cleaned_path=cleaned_path,
        augmented_path=augmented_path,
    )
