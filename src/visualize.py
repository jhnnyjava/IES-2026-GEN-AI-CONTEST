from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .utils import FIGURES_DIR, ensure_project_dirs


def _prepare_output_path(output_path: str | Path) -> Path:
    ensure_project_dirs()
    return Path(output_path)


def save_actual_vs_predicted_plot(y_true: Sequence[float], y_pred: Sequence[float], output_path: str | Path) -> Path:
    """Create a parity plot for actual and predicted values."""

    output = _prepare_output_path(output_path)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.75, edgecolor="black", linewidths=0.4)
    bounds = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    plt.plot(bounds, bounds, linestyle="--", color="dimgray", linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()
    return output


def save_residual_plot(y_true: Sequence[float], y_pred: Sequence[float], output_path: str | Path) -> Path:
    """Plot residuals against predicted values to inspect bias."""

    output = _prepare_output_path(output_path)
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_pred, residuals, alpha=0.75, edgecolor="black", linewidths=0.4)
    plt.axhline(0.0, color="dimgray", linestyle="--", linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()
    return output


def save_feature_importance_plot(feature_names: Sequence[str], importances: Sequence[float], output_path: str | Path) -> Path:
    """Save a simple horizontal bar plot for the most important features."""

    output = _prepare_output_path(output_path)
    feature_importances = sorted(zip(feature_names, importances), key=lambda item: abs(item[1]), reverse=True)[:15]
    if not feature_importances:
        raise ValueError("No feature importances were provided for plotting.")

    names = [item[0] for item in feature_importances][::-1]
    values = [item[1] for item in feature_importances][::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(names, values, color="#2e7d32")
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()
    return output
