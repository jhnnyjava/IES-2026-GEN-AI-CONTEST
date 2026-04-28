from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import (
    AUGMENTED_DATA_PATH,
    BASE_FEATURE_COLUMNS,
    ENVIRONMENTAL_FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMNS,
    clean_column_names,
    ensure_project_dirs,
    safe_to_numeric,
    save_json,
    stable_int_hash,
)

REGION_PRIORS: dict[str, dict[str, float]] = {
    "central": {"rainfall_mm": 1180.0, "rainfall_sd": 110.0, "temperature_c": 20.4, "temperature_sd": 1.2, "humidity_pct": 69.0, "humidity_sd": 4.0},
    "coastal": {"rainfall_mm": 980.0, "rainfall_sd": 140.0, "temperature_c": 28.2, "temperature_sd": 1.4, "humidity_pct": 78.0, "humidity_sd": 4.5},
    "eastern": {"rainfall_mm": 690.0, "rainfall_sd": 120.0, "temperature_c": 24.7, "temperature_sd": 1.3, "humidity_pct": 58.0, "humidity_sd": 5.0},
    "nyanza": {"rainfall_mm": 1120.0, "rainfall_sd": 120.0, "temperature_c": 22.5, "temperature_sd": 1.1, "humidity_pct": 73.0, "humidity_sd": 4.0},
    "rift valley": {"rainfall_mm": 860.0, "rainfall_sd": 130.0, "temperature_c": 19.4, "temperature_sd": 1.5, "humidity_pct": 60.0, "humidity_sd": 4.5},
    "western": {"rainfall_mm": 1320.0, "rainfall_sd": 125.0, "temperature_c": 21.6, "temperature_sd": 1.1, "humidity_pct": 76.0, "humidity_sd": 4.0},
}

YEAR_SHIFTS = {
    "86-90": -0.12,
    "91-95": -0.08,
    "96-00": -0.03,
    "01-05": 0.00,
    "06-10": 0.04,
    "11-15": 0.08,
}

ENVIRONMENTAL_BOUNDS = {
    "rainfall_mm": (120.0, 2400.0),
    "temperature_c": (10.0, 36.0),
    "humidity_pct": (20.0, 98.0),
}


def _rng_for_row(*parts: Any, seed: int = RANDOM_STATE) -> np.random.Generator:
    return np.random.default_rng(stable_int_hash(*parts, seed=seed))


def _region_key(value: Any) -> str:
    return str(value).strip().lower() if value is not None else "unknown"


def _year_shift(year_value: Any) -> float:
    return YEAR_SHIFTS.get(str(year_value).strip(), 0.0)


def _clip_feature(name: str, value: float) -> float:
    lower, upper = ENVIRONMENTAL_BOUNDS[name]
    return float(np.clip(value, lower, upper))


def add_environmental_features(df: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Add reproducible synthetic environmental proxies to the maize records."""

    if df.empty:
        raise ValueError("Cannot engineer features for an empty dataframe.")

    frame = clean_column_names(df).copy()
    missing_columns = [column for column in BASE_FEATURE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns for feature engineering: {missing_columns}")

    frame["areaharv"] = safe_to_numeric(frame["areaharv"]).fillna(frame["areaharv"].median())
    frame["year"] = frame["year"].astype("string").str.strip().fillna("unknown")
    frame["adlevel1"] = frame["adlevel1"].astype("string").str.strip().fillna("unknown")
    frame["adlevel2"] = frame["adlevel2"].astype("string").str.strip().fillna("unknown")
    frame["adlevel3"] = frame["adlevel3"].astype("string").str.strip().fillna("unknown")

    rainfall_values: list[float] = []
    temperature_values: list[float] = []
    humidity_values: list[float] = []

    for _, row in frame.iterrows():
        priors = REGION_PRIORS.get(_region_key(row["adlevel1"]), REGION_PRIORS["rift valley"])
        year_shift = _year_shift(row["year"])
        area_signal = float(np.log1p(max(float(row["areaharv"]), 1.0)))
        rng = _rng_for_row(row["adlevel1"], row["adlevel2"], row["adlevel3"], row["year"], seed=seed)

        rainfall = priors["rainfall_mm"] + (area_signal * 18.0) + (year_shift * 120.0) + rng.normal(0.0, priors["rainfall_sd"])
        rainfall = _clip_feature("rainfall_mm", rainfall)

        temperature = priors["temperature_c"] - 0.0025 * (rainfall - priors["rainfall_mm"]) + rng.normal(0.0, priors["temperature_sd"])
        temperature = _clip_feature("temperature_c", temperature)

        humidity = priors["humidity_pct"] + 0.008 * (rainfall - priors["rainfall_mm"]) - 0.55 * (temperature - priors["temperature_c"]) + rng.normal(0.0, priors["humidity_sd"])
        humidity = _clip_feature("humidity_pct", humidity)

        rainfall_values.append(round(rainfall, 2))
        temperature_values.append(round(temperature, 2))
        humidity_values.append(round(humidity, 2))

    frame["rainfall_mm"] = rainfall_values
    frame["temperature_c"] = temperature_values
    frame["humidity_pct"] = humidity_values

    return frame


def summarize_environmental_features(df: pd.DataFrame) -> dict[str, float]:
    """Return a compact summary of the engineered proxy features."""

    summary: dict[str, float] = {}
    for column in ENVIRONMENTAL_FEATURE_COLUMNS:
        if column in df.columns:
            summary[f"{column}_mean"] = float(pd.to_numeric(df[column], errors="coerce").mean())
            summary[f"{column}_std"] = float(pd.to_numeric(df[column], errors="coerce").std())
    return summary


def save_augmented_dataset(df: pd.DataFrame, output_path: str | Path = AUGMENTED_DATA_PATH) -> Path:
    ensure_project_dirs()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return output


def augment_and_persist(df: pd.DataFrame, seed: int = RANDOM_STATE, output_path: str | Path = AUGMENTED_DATA_PATH) -> tuple[pd.DataFrame, Path, dict[str, float]]:
    augmented = add_environmental_features(df, seed=seed)
    saved_path = save_augmented_dataset(augmented, output_path=output_path)
    summary = summarize_environmental_features(augmented)
    save_json(Path(saved_path).with_suffix(".json"), summary)
    return augmented, saved_path, summary


def infer_environmental_features(sample: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Augment a single-sample or batch inference frame with deterministic environmental proxies."""

    return add_environmental_features(sample, seed=seed)
