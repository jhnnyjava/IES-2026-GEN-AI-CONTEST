"""Climate data integration: merge rainfall and temperature with maize production data.

This module loads monthly rainfall and temperature data, aggregates them into
yearly features (aligned with maize production period labels), and merges with
the maize dataset to produce a climate-aware feature set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
MERGED_PATH = PROCESSED_DIR / "merged_dataset.csv"

ENV_YEARLY_COLUMNS = [
    "annual_rainfall",
    "long_rains",
    "short_rains",
    "rainfall_std",
    "avg_temp",
    "temp_max",
    "temp_min",
    "temp_std",
]


def _parse_month_label(label: str) -> int:
    """Extract numeric month from label like 'Jan Average' -> 1."""
    label = str(label).strip()
    month_token = label.split()[0].lower()
    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    return months.get(month_token[:3], 0)


def _load_climate(csv_path: Path | str, value_column_hint: str = None) -> pd.DataFrame:
    """Load climate CSV and parse year/month/value columns."""
    path = Path(csv_path) if isinstance(csv_path, str) else csv_path
    if not path.exists():
        raise FileNotFoundError(f"Climate file not found: {path}")

    df = pd.read_csv(path)
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    # Detect columns
    year_col = next((c for c in df.columns if "year" in c.lower()), None)
    month_col = next((c for c in df.columns if "month" in c.lower()), None)
    value_col = None

    if value_column_hint:
        value_col = next((c for c in df.columns if value_column_hint.lower() in c.lower()), None)

    if value_col is None:
        # Pick first numeric column that's not year/month
        for c in df.columns:
            if c not in {year_col, month_col}:
                try:
                    pd.to_numeric(df[c].dropna())
                    value_col = c
                    break
                except Exception:
                    continue

    if year_col is None or month_col is None or value_col is None:
        raise ValueError(
            f"Could not identify year/month/value columns in {path}. "
            f"Found: {df.columns.tolist()}"
        )

    df = df[[year_col, month_col, value_col]].rename(
        columns={year_col: "year_raw", month_col: "month_raw", value_col: "value"}
    )
    df["year"] = df["year_raw"].astype(str).str.strip().astype(int)
    df["month"] = df["month_raw"].apply(_parse_month_label)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["year", "month", "value"]).reset_index(drop=True)
    return df[["year", "month", "value"]]


def _aggregate_rainfall(rain_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly rainfall into yearly metrics."""
    pivot = rain_df.pivot_table(index="year", columns="month", values="value", aggfunc="mean")
    pivot = pivot.fillna(np.nan)

    agg = pd.DataFrame()
    agg["year"] = pivot.index.to_numpy()
    agg["annual_rainfall"] = rain_df.groupby("year")["value"].sum().to_numpy()
    agg["rainfall_std"] = rain_df.groupby("year")["value"].std().to_numpy()
    # Long rains: Mar(3), Apr(4), May(5)
    agg["long_rains"] = (pivot.get(3, 0.0) + pivot.get(4, 0.0) + pivot.get(5, 0.0)).to_numpy()
    # Short rains: Oct(10), Nov(11), Dec(12)
    agg["short_rains"] = (pivot.get(10, 0.0) + pivot.get(11, 0.0) + pivot.get(12, 0.0)).to_numpy()

    agg = agg.reset_index(drop=True)
    return agg


def _aggregate_temperature(temp_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly temperature into yearly metrics."""
    pivot = temp_df.pivot_table(index="year", columns="month", values="value", aggfunc="mean")
    pivot = pivot.fillna(np.nan)

    agg = pd.DataFrame()
    agg["year"] = pivot.index.to_numpy()
    agg["avg_temp"] = pivot.mean(axis=1).to_numpy()
    agg["temp_max"] = pivot.max(axis=1).to_numpy()
    agg["temp_min"] = pivot.min(axis=1).to_numpy()
    agg["temp_std"] = pivot.std(axis=1).to_numpy()

    agg = agg.reset_index(drop=True)
    return agg


def _period_to_year_range(period: str) -> Tuple[int, int]:
    """Convert period like '86-90' or '01-05' to (start_year, end_year)."""
    s = str(period).strip()
    if "-" not in s:
        try:
            y = int(s)
            return y, y
        except Exception:
            raise ValueError(f"Unrecognized period format: {period}")

    a, b = s.split("-", 1)
    a = a.strip()
    b = b.strip()

    def to_full_year(x: str) -> int:
        x = x.strip()
        if len(x) == 4:
            return int(x)
        if len(x) == 2:
            y = int(x)
            # Assume 30+ = 1900s, <30 = 2000s
            if y <= 30:
                return 2000 + y
            return 1900 + y
        return int(x)

    start = to_full_year(a)
    end = to_full_year(b)
    if end < start:
        # Handle century crossing (e.g., "96-00" -> "1996-2000")
        end += 100
    return start, end


def _period_mean_for_years(
    yearly_df: pd.DataFrame, start: int, end: int, columns: List[str]
) -> Dict[str, float]:
    """Compute mean of yearly features for a given period, with fallback."""
    years = list(range(start, end + 1))
    available = yearly_df[yearly_df["year"].isin(years)]

    if not available.empty:
        vals = {}
        for col in columns:
            vals[col] = float(available[col].mean()) if col in available.columns else float("nan")
        return vals

    # Fallback: use nearest available window of same length
    all_years = sorted(yearly_df["year"].unique())
    if not all_years:
        return {col: float("nan") for col in columns}

    window = max(1, min(len(all_years), len(years)))
    center = (start + end) / 2.0
    best_idx = 0
    best_dist = float("inf")

    for i in range(0, len(all_years) - window + 1):
        window_years = all_years[i : i + window]
        w_center = (window_years[0] + window_years[-1]) / 2.0
        dist = abs(w_center - center)
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    chosen = all_years[best_idx : best_idx + window]
    available = yearly_df[yearly_df["year"].isin(chosen)]
    vals = {}
    for col in columns:
        vals[col] = float(available[col].mean()) if col in available.columns else float("nan")
    return vals


def integrate_climate_with_maize(
    maize_df: pd.DataFrame,
    rainfall_path: Path | str | None = None,
    temperature_path: Path | str | None = None,
    save_path: Path | str = MERGED_PATH,
) -> pd.DataFrame:
    """Integrate rainfall and temperature yearly aggregates into maize dataframe.

    Args:
        maize_df: Maize production dataframe with 'year' column (period labels like '86-90').
        rainfall_path: Path to rainfall CSV. If None, attempts DATA_DIR / "rainfall.csv".
        temperature_path: Path to temperature CSV. If None, attempts DATA_DIR / "temperature.csv".
        save_path: Path to save merged dataset.

    Returns:
        Merged dataframe with environmental features added.
    """
    # Resolve paths
    if rainfall_path is None:
        rainfall_path = DATA_DIR / "rainfall.csv"
    if temperature_path is None:
        temperature_path = DATA_DIR / "temperature.csv"

    rainfall_path = Path(rainfall_path)
    temperature_path = Path(temperature_path)

    # If climate files don't exist, return original maize_df
    if not rainfall_path.exists() or not temperature_path.exists():
        print(f"Warning: Climate files not found at {rainfall_path}, {temperature_path}")
        print("Returning original maize dataset without climate integration.")
        return maize_df.copy()

    # Load and aggregate climate data
    rain = _load_climate(rainfall_path, value_column_hint="Rainfall")
    temp = _load_climate(temperature_path, value_column_hint="Temperature")

    rain_yearly = _aggregate_rainfall(rain)
    temp_yearly = _aggregate_temperature(temp)

    yearly = pd.merge(rain_yearly, temp_yearly, on="year", how="outer")

    # For each maize period, compute period mean of yearly features
    merged_rows = []
    for _, row in maize_df.iterrows():
        period = row.get("year")
        try:
            start, end = _period_to_year_range(period)
        except Exception:
            start, end = None, None

        env_vals = {col: float("nan") for col in ENV_YEARLY_COLUMNS}
        if start is not None:
            vals = _period_mean_for_years(
                yearly,
                start,
                end,
                ["annual_rainfall", "long_rains", "short_rains", "rainfall_std",
                 "avg_temp", "temp_max", "temp_min", "temp_std"],
            )
            env_vals.update(vals)

            # Convert Series to dict, merge with env_vals, and create new dict
            row_dict = row.to_dict()
            row_dict.update(env_vals)
            merged_rows.append(row_dict)

    merged = pd.DataFrame(merged_rows)

    # Save merged dataset
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(save_path, index=False)
    print(f"Saved integrated dataset to {save_path}")

    return merged
