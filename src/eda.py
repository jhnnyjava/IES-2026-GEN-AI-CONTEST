from __future__ import annotations

import os
from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATHS = [
    Path("data/ken_maize_production.csv"),
    Path("data/raw/raw_ken_maize_production.csv"),
    Path("c:/Users/johnk/Downloads/ken_maize_production.csv"),
]

FIG_1 = Path("figures/figure_1_maize_production_by_region.png")
FIG_2 = Path("figures/figure_2_correlation_matrix.png")
SUMMARY_CSV = Path("reports/data_summary.csv")
SUMMARY_TXT = Path("reports/data_summary.txt")
INSIGHTS_TXT = Path("reports/eda_insights.txt")


def ensure_dirs():
    for p in [FIG_1.parent, FIG_2.parent, SUMMARY_CSV.parent]:
        p.mkdir(parents=True, exist_ok=True)


def snake_case(s: str) -> str:
    s = str(s).strip()
    s = s.replace(" ", "_").replace("-", "_")
    s = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in s)
    return "_".join([part for part in s.lower().split("_") if part])


def load_and_clean(path_candidates: list[Path]) -> pd.DataFrame:
    path = None
    for p in path_candidates:
        if p.exists():
            path = p
            break
    if path is None:
        raise FileNotFoundError("Could not find data/ken_maize_production.csv or fallback paths.")

    df = pd.read_csv(path)
    # normalize columns
    df.columns = [snake_case(c) for c in df.columns]

    # drop irrelevant columns if present
    drop_cols = {"the_geom", "fid", "area", "perimeter", "regions_", "regions_id", "sqkm", "admsqkm", "code", "adminid", "country"}
    keep = ["adlevel1", "adlevel2", "adlevel3", "totmazprod", "mazyield", "areaharv", "year"]
    # remove leading/trailing whitespace in string-like columns (avoid select_dtypes deprecation)
    for col in df.columns:
        try:
            if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip()
        except Exception:
            df[col] = df[col].astype(str).str.strip()

    for c in list(df.columns):
        if c in drop_cols:
            df = df.drop(columns=[c], errors="ignore")

    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after cleaning: {missing}")

    df = df.loc[:, keep].copy()

    # coerce numeric columns
    for col in ["totmazprod", "mazyield", "areaharv"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # basic safety: drop rows where target is NaN
    df = df.dropna(subset=["totmazprod"]).reset_index(drop=True)
    return df


def figure_1_bar(df: pd.DataFrame, out_path: Path) -> None:
    grouped = df.groupby("adlevel1", dropna=False)["totmazprod"].sum()
    grouped = grouped.sort_values(ascending=False)

    plt.rcParams.update({"figure.dpi": 300, "font.size": 10})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(grouped.index.astype(str), grouped.values, color="#2f6f9f")
    ax.set_title("Total Maize Production by Region (Kenya)", fontsize=12, weight="bold")
    ax.set_xlabel("Region (adlevel1)")
    ax.set_ylabel("Total Maize Production (TOTMAZPROD)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def figure_2_corr(df: pd.DataFrame, out_path: Path) -> None:
    nums = df[["totmazprod", "mazyield", "areaharv"]].copy()
    corr = nums.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(5, 4))
    cmap = plt.get_cmap("coolwarm")
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    # annotate
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            text = f"{corr.iat[i,j]:.2f}"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=9)

    ax.set_title("Correlation Matrix of Key Agricultural Variables")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def data_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["totmazprod", "mazyield", "areaharv"]
    desc = df[cols].describe().transpose()
    # keep only required rows and rename percentiles
    desc = desc.loc[cols, ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    return desc


def write_reports(df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    ensure_dirs()
    # CSV
    summary_df.to_csv(SUMMARY_CSV)

    # text summary
    total_rows = int(len(df))
    n_regions = int(df["adlevel1"].nunique())
    zeros = {col: int((df[col] == 0).sum()) for col in ["totmazprod", "mazyield", "areaharv"]}

    with SUMMARY_TXT.open("w", encoding="utf-8") as f:
        f.write("Data Summary\n")
        f.write("============\n\n")
        f.write(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))
        f.write("\n\n")
        f.write(f"Total rows: {total_rows}\n")
        f.write(f"Number of regions (adlevel1): {n_regions}\n")
        f.write("Zero counts:\n")
        for k, v in zeros.items():
            f.write(f" - {k}: {v}\n")


def insights_and_top_regions(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    grouped = df.groupby("adlevel1", dropna=False)["totmazprod"].sum().sort_values(ascending=False)
    top5 = grouped.head(5).reset_index()

    # correlation insight between areaharv and totmazprod
    corr_val = df["areaharv"].corr(df["totmazprod"]) if df["areaharv"].notna().any() else np.nan
    interp = textwrap.dedent(
        f"""
        Top 5 regions by total maize production:\n{top5.to_string(index=False)}\n\n
        Relationship: Pearson correlation between `areaharv` and `totmazprod` = {corr_val:.3f}.
        Interpretation: production generally increases with harvested area; however, variance indicates other factors also matter.
        """
    )
    return interp.strip(), top5


def main() -> None:
    ensure_dirs()
    df = load_and_clean(DATA_PATHS)

    # Figures
    figure_1_bar(df, FIG_1)
    figure_2_corr(df, FIG_2)

    # Summary
    summary_df = data_summary(df)
    write_reports(df, summary_df)

    # Insights
    interp, top5 = insights_and_top_regions(df)
    with INSIGHTS_TXT.open("w", encoding="utf-8") as f:
        f.write(interp + "\n")


if __name__ == "__main__":
    main()
