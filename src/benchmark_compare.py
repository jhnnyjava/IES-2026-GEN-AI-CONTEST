"""Benchmark comparison: Local vs Simulated Edge inference latency.

Usage:
    python -m src.benchmark_compare --iterations 200 --batch-size 8

The script loads the saved pipeline at `models/best_model.pkl` and measures
prediction latency (ms) for single-sample and batch inference in two modes:
LOCAL and SIMULATED_EDGE. SIMULATED_EDGE constrains native thread counts to
approximate a single-core edge device. The input and preprocessing pipeline
are identical for both modes to ensure a fair comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .predict import build_input_frame, load_model_artifacts, load_real_sample_from_dataset
from .utils import MODEL_PATH, MODEL_METADATA_PATH, FIGURES_DIR, REPORTS_DIR


def set_thread_limits(single_thread: bool) -> None:
    # Best-effort limit of native numeric threads to approximate edge CPU
    if single_thread:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
    else:
        os.environ.pop("OMP_NUM_THREADS", None)
        os.environ.pop("OPENBLAS_NUM_THREADS", None)
        os.environ.pop("MKL_NUM_THREADS", None)


def time_predictions(model: Any, metadata: Dict[str, Any], sample: Dict[str, Any], iterations: int, batch_size: int, warmup: int = 5) -> Tuple[List[float], List[float]]:
    """Return two lists of latencies (per-batch-ms, per-sample-ms) for repeated runs.

    per-batch-ms: elapsed ms for each batch prediction
    per-sample-ms: per-sample equivalent (batch_ms / batch_size)
    """
    frame = build_input_frame(sample, metadata)
    # prepare batch
    batch_frame = pd.concat([frame.copy() for _ in range(batch_size)], ignore_index=True)

    # warmup
    for _ in range(warmup):
        _ = model.predict(batch_frame)

    batch_ms: List[float] = []
    per_sample_ms: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model.predict(batch_frame)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        batch_ms.append(elapsed_ms)
        per_sample_ms.append(elapsed_ms / float(batch_size))

    return batch_ms, per_sample_ms


def summarize(latencies_ms: List[float]) -> Dict[str, float]:
    return {
        "mean_ms": float(statistics.mean(latencies_ms)),
        "median_ms": float(statistics.median(latencies_ms)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
    }


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_txt(summary: Dict[str, Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as fh:
        fh.write(json.dumps(summary, indent=2))


def plot_latency(local_ms: List[float], edge_ms: List[float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [local_ms, edge_ms]
    labels = ["Local", "Simulated Edge"]

    plt.style.use(["seaborn-whitegrid"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, labels=labels, showfliers=False, notch=True)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Local vs Simulated Edge Latency (per-sample)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare model inference latency locally vs simulated edge.")
    p.add_argument("--iterations", type=int, default=200, help="Number of timed iterations per mode")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for inference tests")
    p.add_argument("--warmup", type=int, default=5, help="Warmup iterations before timing")
    p.add_argument("--model-path", type=str, default=str(MODEL_PATH), help="Path to saved model pipeline")
    p.add_argument("--metadata-path", type=str, default=str(MODEL_METADATA_PATH), help="Path to model metadata JSON")
    p.add_argument("--out-csv", type=str, default=str(Path(REPORTS_DIR) / "edge_vs_local_comparison.csv"), help="CSV output path")
    p.add_argument("--out-summary", type=str, default=str(Path(REPORTS_DIR) / "edge_vs_local_summary.txt"), help="Summary output path")
    p.add_argument("--out-fig", type=str, default=str(Path(FIGURES_DIR) / "edge_vs_local_latency.png"), help="Latency figure output path")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(42)

    model, metadata = load_model_artifacts(args.model_path, args.metadata_path)
    sample, source = load_real_sample_from_dataset(metadata)
    print(f"Using sample extracted from {source}")

    rows: List[Dict[str, Any]] = []

    # LOCAL run (allow multi-threading)
    set_thread_limits(single_thread=False)
    local_batch_ms, local_per_sample_ms = time_predictions(model, metadata, sample, args.iterations, args.batch_size, args.warmup)
    local_summary = summarize(local_per_sample_ms)

    for i, (b_ms, s_ms) in enumerate(zip(local_batch_ms, local_per_sample_ms)):
        rows.append({"mode": "local", "iteration": i, "batch_ms": b_ms, "per_sample_ms": s_ms, "batch_size": args.batch_size})

    # SIMULATED EDGE run (constrain threads to approximate Pi)
    set_thread_limits(single_thread=True)
    edge_batch_ms, edge_per_sample_ms = time_predictions(model, metadata, sample, args.iterations, args.batch_size, args.warmup)
    edge_summary = summarize(edge_per_sample_ms)

    for i, (b_ms, s_ms) in enumerate(zip(edge_batch_ms, edge_per_sample_ms)):
        rows.append({"mode": "simulated_edge", "iteration": i, "batch_ms": b_ms, "per_sample_ms": s_ms, "batch_size": args.batch_size})

    out_csv = Path(args.out_csv)
    save_csv(rows, out_csv)

    summary = {"local": local_summary, "simulated_edge": edge_summary}
    save_summary_txt(summary, Path(args.out_summary))

    plot_latency(local_per_sample_ms, edge_per_sample_ms, Path(args.out_fig))

    print("Benchmark complete.")
    print(f"CSV: {out_csv}")
    print(f"Summary: {args.out_summary}")
    print(f"Figure: {args.out_fig}")


if __name__ == "__main__":
    main()
