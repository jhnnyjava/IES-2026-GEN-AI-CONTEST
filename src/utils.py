from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
AUGMENTED_DATA_DIR = DATA_DIR / "augmented"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

DEFAULT_RAW_DATA_PATH = DATA_DIR / "ken_maize_production.csv"
RAW_COPY_PATH = RAW_DATA_DIR / "raw_ken_maize_production.csv"
CLEANED_DATA_PATH = CLEANED_DATA_DIR / "cleaned_ken_maize_production.csv"
AUGMENTED_DATA_PATH = AUGMENTED_DATA_DIR / "augmented_ken_maize_production.csv"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "best_model_metadata.json"
RESULTS_CSV_PATH = REPORTS_DIR / "model_results.csv"
EDGE_LATENCY_CSV_PATH = REPORTS_DIR / "edge_latency.csv"
EDGE_LATENCY_TXT_PATH = REPORTS_DIR / "edge_latency.txt"
DATASET_PROFILE_PATH = REPORTS_DIR / "dataset_profile.md"
METHODOLOGY_NOTES_PATH = REPORTS_DIR / "methodology_notes.md"
RESULTS_SUMMARY_PATH = REPORTS_DIR / "results_summary.md"
PAPER_DISCUSSION_PATH = REPORTS_DIR / "paper_discussion_summary.txt"

RANDOM_STATE = 42
TARGET_PRIMARY = "totmazprod"
TARGET_SECONDARY = "mazyield"
TARGET_COLUMNS = (TARGET_PRIMARY, TARGET_SECONDARY)
BASE_FEATURE_COLUMNS = ("adlevel1", "adlevel2", "adlevel3", "year", "areaharv")
ENVIRONMENTAL_FEATURE_COLUMNS = ("rainfall_mm", "temperature_c", "humidity_pct")
CATEGORICAL_FEATURE_COLUMNS = ("adlevel1", "adlevel2", "adlevel3", "year")
NUMERIC_FEATURE_COLUMNS = ("areaharv", "rainfall_mm", "temperature_c", "humidity_pct")
MODEL_INPUT_COLUMNS = BASE_FEATURE_COLUMNS + ENVIRONMENTAL_FEATURE_COLUMNS

EXPECTED_SOURCE_COLUMNS = {
    "_id",
    "fid",
    "the_geom",
    "area",
    "perimeter",
    "regions_",
    "regions_id",
    "sqkm",
    "admsqkm",
    "code",
    "adminid",
    "country",
    "adlevel1",
    "adlevel2",
    "adlevel3",
    "totmazprod",
    "mazyield",
    "areaharv",
    "year",
}
METADATA_COLUMNS = EXPECTED_SOURCE_COLUMNS - set(BASE_FEATURE_COLUMNS) - set(TARGET_COLUMNS)


def ensure_project_dirs() -> None:
    """Create all output folders used by the pipeline."""

    for path in [DATA_DIR, RAW_DATA_DIR, CLEANED_DATA_DIR, AUGMENTED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, NOTEBOOKS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_column_name(name: str) -> str:
    """Convert a raw column label to a stable snake_case name."""

    normalized = name.strip().lower()
    normalized = normalized.replace("/", "_").replace("-", "_")
    normalized = re.sub(r"[^0-9a-zA-Z_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the frame with sanitized column names."""

    renamed = df.copy()
    renamed.columns = [normalize_column_name(column) for column in renamed.columns]
    return renamed


def safe_to_numeric(series: pd.Series) -> pd.Series:
    """Coerce mixed string columns into numeric values where possible."""

    cleaned = series.astype("string").str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def stable_int_hash(*parts: Any, seed: int = RANDOM_STATE) -> int:
    """Return a deterministic integer hash for reproducible synthetic augmentation."""

    payload = "|".join([str(seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def save_text(path: str | Path, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stringify_columns(columns: Iterable[str]) -> str:
    return ", ".join(columns)


def summarize_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Build a compact audit summary for reports and console output."""

    numeric_columns = list(df.select_dtypes(include=["number"]).columns)
    categorical_columns = [column for column in df.columns if column not in numeric_columns]
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "missing_by_column": df.isna().sum().sort_values(ascending=False).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }
    return summary


def format_metric_value(value: Any) -> str:
    """Format a metric for readable report text."""

    if value is None:
        return "n/a"
    if pd.isna(value):
        return "n/a"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.4f}"
    return str(value)


def safe_percent(value: float) -> str:
    """Format a proportion as a percentage string."""

    return f"{100.0 * value:.2f}%" if not math.isnan(value) else "n/a"
