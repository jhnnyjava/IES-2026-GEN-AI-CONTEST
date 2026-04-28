from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .train import TrainingBundle
from .utils import FIGURES_DIR, PAPER_DISCUSSION_PATH, RESULTS_SUMMARY_PATH, ensure_project_dirs, format_metric_value, save_text
from .visualize import save_actual_vs_predicted_plot, save_feature_importance_plot, save_residual_plot


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the core regression metrics used in the paper."""

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _extract_feature_importance(bundle: TrainingBundle) -> tuple[list[str], np.ndarray] | None:
    model = bundle.pipeline.named_steps["model"]
    preprocessor = bundle.pipeline.named_steps["preprocessor"]

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        importances = np.asarray(model.coef_).ravel()
    else:
        return None

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = bundle.feature_columns

    if len(feature_names) != len(importances):
        return None

    return feature_names, np.abs(importances)


def evaluate_best_model(bundle: TrainingBundle, output_dir: str | Path = FIGURES_DIR) -> dict[str, Any]:
    """Generate evaluation plots and a compact metrics dictionary."""

    ensure_project_dirs()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(bundle.y_test, bundle.y_pred)

    actual_vs_predicted_path = save_actual_vs_predicted_plot(
        bundle.y_test,
        bundle.y_pred,
        output_path / f"actual_vs_predicted_{bundle.target}_{bundle.model_name}.png",
    )
    residual_plot_path = save_residual_plot(
        bundle.y_test,
        bundle.y_pred,
        output_path / f"residuals_{bundle.target}_{bundle.model_name}.png",
    )

    feature_importance_path = None
    importance_bundle = _extract_feature_importance(bundle)
    if importance_bundle is not None:
        feature_names, importances = importance_bundle
        feature_importance_path = save_feature_importance_plot(
            feature_names,
            importances,
            output_path / f"feature_importance_{bundle.target}_{bundle.model_name}.png",
        )

    return {
        "metrics": metrics,
        "actual_vs_predicted_path": str(actual_vs_predicted_path),
        "residual_plot_path": str(residual_plot_path),
        "feature_importance_path": str(feature_importance_path) if feature_importance_path else None,
    }


def build_results_summary(results_df: pd.DataFrame, bundle: TrainingBundle, dataset_summary: dict[str, Any]) -> tuple[Path, Path]:
    """Write the manuscript-ready summary and discussion note files."""

    ensure_project_dirs()

    ranked = results_df.sort_values("mae", ascending=True).reset_index(drop=True)
    target_best = ranked.groupby("target", as_index=False).first().sort_values("mae")

    lines = [
        "# AgriResilAI+ Results Summary",
        "",
        f"- Dataset rows: {dataset_summary.get('rows', 'n/a')}",
        f"- Dataset columns: {dataset_summary.get('columns', 'n/a')}",
        f"- Best target: {bundle.target}",
        f"- Best model: {bundle.model_name}",
        f"- Holdout MAE: {format_metric_value(bundle.mae)}",
        f"- Holdout RMSE: {format_metric_value(bundle.rmse)}",
        f"- Holdout R2: {format_metric_value(bundle.r2)}",
        f"- Zero share for target: {format_metric_value(bundle.target_zero_share)}",
        "",
        "## Model Comparison",
    ]

    for _, row in ranked.iterrows():
        lines.append(
            f"- {row['target']} | {row['model']} | MAE={format_metric_value(row['mae'])} | RMSE={format_metric_value(row['rmse'])} | R2={format_metric_value(row['r2'])}"
        )

    lines.extend([
        "",
        "## Target-Level Best Scores",
    ])
    for _, row in target_best.iterrows():
        lines.append(f"- {row['target']} best MAE={format_metric_value(row['mae'])} using {row['model']}")

    summary_text = "\n".join(lines) + "\n"
    save_text(RESULTS_SUMMARY_PATH, summary_text)

    discussion_lines = [
        "AgriResilAI+ paper discussion note",
        "",
        f"The cleaned maize dataset contained {dataset_summary.get('rows', 'n/a')} rows and {dataset_summary.get('columns', 'n/a')} columns after removing geometry and pure metadata fields.",
        f"Among the candidate regression tasks, {bundle.target} produced the strongest holdout performance and was therefore selected for the final edge-ready model.",
        f"The selected estimator was {bundle.model_name}, with MAE={format_metric_value(bundle.mae)}, RMSE={format_metric_value(bundle.rmse)}, and R2={format_metric_value(bundle.r2)} on the test split.",
        "The task retained administrative and seasonal descriptors together with harvested area while treating YEAR as a categorical descriptor because the source data are aggregated by period rather than by exact calendar year.",
        "These results support the use of lightweight classical ML for Raspberry Pi deployment without requiring deep learning or cloud inference.",
    ]
    discussion_text = "\n".join(discussion_lines) + "\n"
    save_text(PAPER_DISCUSSION_PATH, discussion_text)

    return RESULTS_SUMMARY_PATH, PAPER_DISCUSSION_PATH
