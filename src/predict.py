from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .decision import classify_production_risk, format_decision_message
from .feature_engineering import infer_environmental_features
from .utils import MODEL_METADATA_PATH, MODEL_PATH, load_json


def load_model_artifacts(model_path: str | Path = MODEL_PATH, metadata_path: str | Path = MODEL_METADATA_PATH) -> tuple[Any, dict[str, Any]]:
    """Load the saved pipeline and its metadata from disk."""

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Saved model not found at {model_file}. Run training first.")

    pipeline = joblib.load(model_file)
    metadata = load_json(metadata_path) if Path(metadata_path).exists() else {}
    return pipeline, metadata


def _parse_key_value_pairs(items: list[str]) -> dict[str, Any]:
    sample: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid feature assignment '{item}'. Use feature=value.")
        key, value = item.split("=", 1)
        sample[key.strip().lower()] = value.strip()
    return sample


def build_input_frame(sample: dict[str, Any], metadata: dict[str, Any]) -> pd.DataFrame:
    """Align user-provided features with the trained model schema."""

    feature_columns = metadata.get("feature_columns")
    if not feature_columns:
        raise ValueError("Model metadata does not contain feature column information.")

    numeric_features = set(metadata.get("numeric_features", []))
    categorical_features = set(metadata.get("categorical_features", []))

    record: dict[str, Any] = {}
    for column in feature_columns:
        raw_value = sample.get(column, np.nan)
        if column in numeric_features:
            record[column] = pd.to_numeric(raw_value, errors="coerce") if raw_value not in (None, "", "nan", "NaN") else np.nan
        elif column in categorical_features:
            record[column] = str(raw_value).strip() if raw_value not in (None, "") else "unknown"
        else:
            record[column] = raw_value

    frame = pd.DataFrame([record], columns=feature_columns)
    environmental_columns = ["rainfall_mm", "temperature_c", "humidity_pct"]
    if any(column in feature_columns for column in environmental_columns):
        frame = infer_environmental_features(frame)
    return frame


def predict_from_sample(model: Any, sample: dict[str, Any], metadata: dict[str, Any]) -> tuple[float, str]:
    """Run a single prediction and decision classification."""

    input_frame = build_input_frame(sample, metadata)
    prediction = float(np.asarray(model.predict(input_frame)).ravel()[0])
    threshold = float(metadata.get("decision_threshold", 0.0))
    decision = classify_production_risk(prediction, threshold)
    return prediction, decision.label


def prompt_for_sample(metadata: dict[str, Any]) -> dict[str, Any]:
    """Collect a single sample from the command line for portable inference."""

    feature_columns = metadata.get("feature_columns", [])
    numeric_features = set(metadata.get("numeric_features", []))
    sample: dict[str, Any] = {}

    print("Enter values for the model features. Press Enter to leave a field blank.")
    for column in feature_columns:
        if column in {"rainfall_mm", "temperature_c", "humidity_pct"}:
            continue
        raw_value = input(f"{column}: ").strip()
        if column in numeric_features:
            sample[column] = pd.to_numeric(raw_value, errors="coerce") if raw_value else np.nan
        else:
            sample[column] = raw_value if raw_value else "unknown"

    return sample


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference with the saved AgriResilAI+ model.")
    parser.add_argument("--json", type=str, help="Feature values as a JSON object string.")
    parser.add_argument("--values", nargs="*", help="Feature values as key=value pairs.")
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH), help="Path to the saved model.")
    parser.add_argument("--metadata-path", type=str, default=str(MODEL_METADATA_PATH), help="Path to the model metadata JSON.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    model, metadata = load_model_artifacts(args.model_path, args.metadata_path)

    if args.json:
        sample = {key.lower(): value for key, value in json.loads(args.json).items()}
    elif args.values:
        sample = _parse_key_value_pairs(args.values)
    else:
        sample = prompt_for_sample(metadata)

    prediction, decision = predict_from_sample(model, sample, metadata)
    threshold = float(metadata.get("decision_threshold", 0.0))
    target = metadata.get("target", "target")
    print(f"Predicted {target}: {prediction:.4f}")
    print(format_decision_message(prediction, threshold))
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
