from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_loader import load_and_prepare_dataset
from src.edge_demo import run_latency_demo
from src.evaluate import build_results_summary, evaluate_model
from src.predict import load_model_artifacts, predict_from_sample
from src.preprocessing import save_cleaned_dataset
from src.train import save_model_artifacts, save_results_table, train_project
from src.utils import AUGMENTED_DATA_PATH, CLEANED_DATA_PATH, DEFAULT_RAW_DATA_PATH, MODEL_METADATA_PATH, MODEL_PATH, PAPER_DISCUSSION_PATH, RESULTS_CSV_PATH, ensure_project_dirs, save_text


def _select_demo_sample(bundle) -> dict[str, Any]:
    sample_row = bundle.X_test.iloc[0].to_dict()
    return {key: value if not pd.isna(value) else "unknown" for key, value in sample_row.items()}


def _format_console_summary(results_df: pd.DataFrame, bundle, dataset_summary: dict[str, Any], augmentation_summary: dict[str, Any]) -> str:
    primary = results_df[results_df["target"] == bundle.target].sort_values("mae", ascending=True).reset_index(drop=True)
    baseline = primary[primary["model"] == "linear_regression"].iloc[0]
    best = primary.iloc[0]
    improvement = ((baseline["mae"] - best["mae"]) / baseline["mae"] * 100.0) if baseline["mae"] else 0.0

    lines = [
        "AgriResilAI+ workflow complete",
        f"Dataset rows: {dataset_summary.get('rows', 'n/a')}",
        f"Dataset columns: {dataset_summary.get('columns', 'n/a')}",
        f"Rainfall mean: {augmentation_summary.get('rainfall_mm_mean', 'n/a')}",
        f"Temperature mean: {augmentation_summary.get('temperature_c_mean', 'n/a')}",
        f"Humidity mean: {augmentation_summary.get('humidity_pct_mean', 'n/a')}",
        f"Features used: {', '.join(bundle.feature_columns)}",
        f"Best target: {bundle.target}",
        f"Best model: {bundle.model_name}",
        f"Holdout MAE: {bundle.holdout_metrics['mae']:.4f}",
        f"Holdout RMSE: {bundle.holdout_metrics['rmse']:.4f}",
        f"Holdout R2: {bundle.holdout_metrics['r2']:.4f}",
        f"CV MAE: {bundle.cv_metrics['cv_mae_mean']:.4f} ± {bundle.cv_metrics['cv_mae_std']:.4f}",
        f"Linear baseline improvement: {improvement:.2f}%",
        "Target comparison:",
    ]
    for _, row in primary.iterrows():
        lines.append(f"- {row['model']} | MAE={row['mae']:.4f} | RMSE={row['rmse']:.4f} | R2={row['r2']:.4f} | CV MAE={row['cv_mae_mean']:.4f}")
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the complete AgriResilAI+ training and inference workflow.")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_RAW_DATA_PATH), help="Path to the Kenya maize production CSV.")
    parser.add_argument("--sample-json", type=str, help="Optional JSON sample to use for the inference demo.")
    parser.add_argument("--edge-iterations", type=int, default=50, help="Number of latency measurements to run.")
    parser.add_argument("--edge-batch-size", type=int, default=8, help="Batch size for edge latency benchmarking.")
    parser.add_argument("--skip-edge-demo", action="store_true", help="Skip the Raspberry Pi edge latency demo.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    ensure_project_dirs()
    dataset_bundle = load_and_prepare_dataset(args.data_path)
    save_cleaned_dataset(dataset_bundle.cleaned, CLEANED_DATA_PATH)

    training_report = train_project(dataset_bundle.augmented)
    save_results_table(training_report.comparison_table, RESULTS_CSV_PATH)
    save_model_artifacts(training_report.primary_bundle, training_report.comparison_table, dataset_bundle.cleaned_summary, dataset_bundle.augmented_summary)

    evaluation_summary = evaluate_model(training_report.primary_bundle)
    build_results_summary(training_report.comparison_table, training_report.primary_bundle, dataset_bundle.cleaned_summary, dataset_bundle.augmented_summary)

    if args.sample_json:
        sample = {key.lower(): value for key, value in json.loads(args.sample_json).items()}
    else:
        sample = _select_demo_sample(training_report.primary_bundle)

    model, metadata = load_model_artifacts(MODEL_PATH, MODEL_METADATA_PATH)
    prediction, decision = predict_from_sample(model, sample, metadata)

    discussion_text = [
        "AgriResilAI+ automated run",
        f"Selected target: {training_report.primary_bundle.target}",
        f"Selected model: {training_report.primary_bundle.model_name}",
        f"Prediction demo output: {prediction:.4f}",
        f"Decision layer: {decision}",
        f"Actual vs predicted: {evaluation_summary['actual_vs_predicted_path']}",
        f"Residual plot: {evaluation_summary['residual_plot_path']}",
    ]
    if evaluation_summary.get("feature_importance_path"):
        discussion_text.append(f"Feature importance plot: {evaluation_summary['feature_importance_path']}")
    save_text(PAPER_DISCUSSION_PATH, "\n".join(discussion_text) + "\n")

    if not args.skip_edge_demo:
        run_latency_demo(sample, MODEL_PATH, MODEL_METADATA_PATH, iterations=args.edge_iterations, warmup=10, batch_size=args.edge_batch_size)

    print(_format_console_summary(training_report.comparison_table, training_report.primary_bundle, dataset_bundle.cleaned_summary, dataset_bundle.augmented_summary))


if __name__ == "__main__":
    main()
