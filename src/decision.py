from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class DecisionOutcome:
    label: str
    threshold: float
    margin: float


def derive_risk_threshold(training_target: pd.Series, quantile: float = 0.25) -> float:
    """Derive a conservative production threshold for high-risk classification."""

    if training_target.empty:
        raise ValueError("Cannot derive a threshold from an empty target series.")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1.")
    return float(training_target.quantile(quantile))


def classify_production_risk(predicted_production: float, threshold: float) -> DecisionOutcome:
    """Convert a production forecast into a simple decision intelligence label."""

    label = "HIGH RISK REGION" if predicted_production < threshold else "LOW RISK REGION"
    return DecisionOutcome(label=label, threshold=float(threshold), margin=float(predicted_production - threshold))


def format_decision_message(predicted_value: float, threshold: float) -> str:
    outcome = classify_production_risk(predicted_value, threshold)
    return f"{outcome.label} | predicted={predicted_value:.4f} | threshold={outcome.threshold:.4f} | margin={outcome.margin:.4f}"