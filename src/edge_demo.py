from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .predict import build_input_frame, load_model_artifacts, load_real_sample_from_dataset, prompt_for_sample
from .utils import EDGE_LATENCY_CSV_PATH, EDGE_LATENCY_TXT_PATH, ensure_project_dirs, save_text


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else 0.0


def run_latency_demo(
    sample: dict[str, Any],
    model_path: str | Path,
    metadata_path: str | Path,
    iterations: int = 50,
    warmup: int = 5,
    batch_size: int = 8,
) -> tuple[pd.DataFrame, Path, Path]:
    """Measure single-sample and batch inference latency for Raspberry Pi testing."""

    ensure_project_dirs()
    model, metadata = load_model_artifacts(model_path, metadata_path)
    single_frame = build_input_frame(sample, metadata)
    batch_frame = pd.concat([single_frame] * max(1, batch_size), ignore_index=True)

    for _ in range(max(0, warmup)):
        model.predict(single_frame)
        model.predict(batch_frame)

    rows: list[dict[str, Any]] = []
    for index in range(1, iterations + 1):
        start = perf_counter()
        single_prediction = model.predict(single_frame)
        single_latency_ms = (perf_counter() - start) * 1000.0

        start = perf_counter()
        batch_prediction = model.predict(batch_frame)
        batch_latency_ms = (perf_counter() - start) * 1000.0
        batch_per_sample_latency_ms = batch_latency_ms / max(1, batch_size)

        rows.append({"iteration": index, "mode": "single", "batch_size": 1, "latency_ms": single_latency_ms, "per_sample_latency_ms": single_latency_ms, "prediction_mean": float(np.asarray(single_prediction).mean())})
        rows.append({"iteration": index, "mode": "batch", "batch_size": batch_size, "latency_ms": batch_latency_ms, "per_sample_latency_ms": batch_per_sample_latency_ms, "prediction_mean": float(np.asarray(batch_prediction).mean())})

    results = pd.DataFrame(rows)
    results.to_csv(EDGE_LATENCY_CSV_PATH, index=False)

    summary_rows = []
    for mode in ["single", "batch"]:
        subset = results[results["mode"] == mode]
        summary_rows.append(f"{mode.title()} mean latency (ms): {mean(subset['latency_ms']):.4f} | median: {median(subset['latency_ms']):.4f} | p95: {_percentile(subset['latency_ms'].tolist(), 95):.4f}")
        summary_rows.append(f"{mode.title()} per-sample latency (ms): {mean(subset['per_sample_latency_ms']):.4f} | median: {median(subset['per_sample_latency_ms']):.4f} | p95: {_percentile(subset['per_sample_latency_ms'].tolist(), 95):.4f}")

    summary_text = "\n".join(["AgriResilAI+ edge latency summary", f"Iterations: {iterations}", f"Warmup runs: {warmup}", f"Batch size: {batch_size}", *summary_rows])
    save_text(EDGE_LATENCY_TXT_PATH, summary_text + "\n")
    return results, EDGE_LATENCY_CSV_PATH, EDGE_LATENCY_TXT_PATH


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure edge inference latency for the saved model.")
    parser.add_argument("--json", type=str, help="Feature values as a JSON object string.")
    parser.add_argument("--values", nargs="*", help="Feature values as key=value pairs.")
    parser.add_argument("--model-path", type=str, default="models/best_model.pkl")
    parser.add_argument("--metadata-path", type=str, default="models/best_model_metadata.json")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.json:
        sample = {key.lower(): value for key, value in json.loads(args.json).items()}
    elif args.values:
        from .predict import _parse_key_value_pairs

        sample = _parse_key_value_pairs(args.values)
    else:
        _, metadata = load_model_artifacts(args.model_path, args.metadata_path)
        sample, source_path = load_real_sample_from_dataset(metadata)
        print(f"Using real sample extracted from {source_path}.")

    results, _, _ = run_latency_demo(sample, args.model_path, args.metadata_path, args.iterations, args.warmup, args.batch_size)
    if not results.empty:
        print(results.groupby("mode")["latency_ms"].mean().to_string())


if __name__ == "__main__":
    main()
