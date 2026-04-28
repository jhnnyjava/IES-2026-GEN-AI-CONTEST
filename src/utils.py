from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

DEFAULT_RAW_DATA_PATH = DATA_DIR / "ken_maize_production.csv"
RAW_COPY_PATH = DATA_DIR / "raw_ken_maize_production.csv"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_maize_data.csv"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "best_model_metadata.json"
RESULTS_CSV_PATH = REPORTS_DIR / "model_results.csv"
EDGE_LATENCY_CSV_PATH = REPORTS_DIR / "edge_latency.csv"
EDGE_LATENCY_TXT_PATH = REPORTS_DIR / "edge_latency.txt"
METHODOLOGY_NOTES_PATH = REPORTS_DIR / "methodology_notes.md"
RESULTS_SUMMARY_PATH = REPORTS_DIR / "results_summary.md"
PAPER_DISCUSSION_PATH = REPORTS_DIR / "paper_discussion_summary.txt"

RANDOM_STATE = 42
TARGET_COLUMNS = ("totmazprod", "mazyield")
CATEGORICAL_COLUMNS = ("adlevel1", "adlevel2", "adlevel3", "year")
METADATA_COLUMNS = {
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
}


def ensure_project_dirs() -> None:
    """Create all output folders used by the pipeline."""

    for path in [DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, NOTEBOOKS_DIR]:
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

    cleaned = series.astype("string").str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


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

    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "missing_by_column": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    return summary


def format_metric_value(value: Any) -> str:
    """Format a metric for readable report text."""

    if value is None:
        return "n/a"
    if pd.isna(value):
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)
