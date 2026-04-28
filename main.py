from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_loading import load_and_audit
from src.evaluate import build_results_summary, evaluate_best_model
from src.edge_demo import run_latency_demo
from src.predict import predict_from_sample
from src.preprocessing import clean_maize_dataframe, save_cleaned_dataset
from src.train import save_best_model, save_results_table, train_all_targets
from src.utils import (
    CLEANED_DATA_PATH,
    DEFAULT_RAW_DATA_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PAPER_DISCUSSION_PATH,
    RESULTS_CSV_PATH,
    RESULTS_SUMMARY_PATH,
    ensure_project_dirs,
    save_text,
)


def _select_demo_sample(bundle) -> dict[str, Any]:
    sample_row = bundle.X_test.iloc[0].to_dict()
    return {key: value if not pd.isna(value) else "unknown" for key, value in sample_row.items()}


def _format_console_summary(results_df: pd.DataFrame, bundle, dataset_summary: dict[str, Any]) -> str:
    target_rank = results_df.sort_values("mae", ascending=True).reset_index(drop=True)
    best_target_rows = results_df.groupby("target", as_index=False).first().sort_values("mae")
    lines = [
        "AgriResilAI+ workflow complete",
        f"Dataset rows: {dataset_summary.get('rows', 'n/a')}",
        f"Dataset columns: {dataset_summary.get('columns', 'n/a')}",
        f"Features used: {', '.join(bundle.feature_columns)}",
        f"Best target: {bundle.target}",
        f"Best model: {bundle.model_name}",
        f"Holdout MAE: {bundle.mae:.4f}",
        f"Holdout RMSE: {bundle.rmse:.4f}",
        f"Holdout R2: {bundle.r2:.4f}",
        "Target comparison:",
    ]
    for _, row in best_target_rows.iterrows():
        lines.append(f"- {row['target']}: best MAE={row['mae']:.4f} using {row['model']}")
    lines.append("Model ranking:")
    for _, row in target_rank.iterrows():
        lines.append(f"- {row['target']} | {row['model']} | MAE={row['mae']:.4f} | RMSE={row['rmse']:.4f} | R2={row['r2']:.4f}")
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the complete AgriResilAI+ training and inference workflow.")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_RAW_DATA_PATH), help="Path to the Kenya maize production CSV.")
    parser.add_argument("--sample-json", type=str, help="Optional JSON sample to use for the inference demo.")
    parser.add_argument("--edge-iterations", type=int, default=20, help="Number of latency measurements to run.")
    parser.add_argument("--skip-edge-demo", action="store_true", help="Skip the Raspberry Pi edge latency demo.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    ensure_project_dirs()

    raw_df, dataset_summary = load_and_audit(args.data_path)
    cleaned_df = clean_maize_dataframe(raw_df)
    save_cleaned_dataset(cleaned_df, CLEANED_DATA_PATH)

    results_df, best_bundle = train_all_targets(cleaned_df)
    save_results_table(results_df, RESULTS_CSV_PATH)
    save_best_model(best_bundle, MODEL_PATH, MODEL_METADATA_PATH)

    evaluation_summary = evaluate_best_model(best_bundle)
    build_results_summary(results_df, best_bundle, dataset_summary)

    if args.sample_json:
        sample = {key.lower(): value for key, value in json.loads(args.sample_json).items()}
    else:
        sample = _select_demo_sample(best_bundle)

    prediction = predict_from_sample(best_bundle.pipeline, sample, {
        "feature_columns": best_bundle.feature_columns,
        "numeric_features": best_bundle.numeric_features,
        "categorical_features": best_bundle.categorical_features,
    })

    discussion_text = [
        "AgriResilAI+ automated run",
        f"Selected target: {best_bundle.target}",
        f"Selected model: {best_bundle.model_name}",
        f"Prediction demo output: {prediction:.4f}",
        f"Figures: {evaluation_summary['actual_vs_predicted_path']}",
        f"Residual plot: {evaluation_summary['residual_plot_path']}",
    ]
    if evaluation_summary.get("feature_importance_path"):
        discussion_text.append(f"Feature importance plot: {evaluation_summary['feature_importance_path']}")
    save_text(PAPER_DISCUSSION_PATH, "\n".join(discussion_text) + "\n")

    if not args.skip_edge_demo:
        run_latency_demo(sample, MODEL_PATH, MODEL_METADATA_PATH, iterations=args.edge_iterations, warmup=5)

    print(_format_console_summary(results_df, best_bundle, dataset_summary))


if __name__ == "__main__":
    main()
