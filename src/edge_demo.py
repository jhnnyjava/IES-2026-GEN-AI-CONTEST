from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

import pandas as pd

from .predict import _coerce_sample, load_model_artifacts, prompt_for_sample
from .utils import EDGE_LATENCY_CSV_PATH, EDGE_LATENCY_TXT_PATH, ensure_project_dirs, save_text


def run_latency_demo(
    sample: dict[str, Any],
    model_path: str | Path,
    metadata_path: str | Path,
    iterations: int = 50,
    warmup: int = 5,
) -> tuple[pd.DataFrame, Path, Path]:
    """Measure repeated prediction latency in milliseconds for Raspberry Pi testing."""

    ensure_project_dirs()
    model, metadata = load_model_artifacts(model_path, metadata_path)
    sample_frame = _coerce_sample(sample, metadata)

    for _ in range(max(0, warmup)):
        model.predict(sample_frame)

    rows = []
    for index in range(1, iterations + 1):
        start = perf_counter()
        prediction = model.predict(sample_frame)
        latency_ms = (perf_counter() - start) * 1000.0
        rows.append(
            {
                "iteration": index,
                "latency_ms": latency_ms,
                "prediction": float(prediction[0]),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(EDGE_LATENCY_CSV_PATH, index=False)

    summary_text = "\n".join(
        [
            "AgriResilAI+ edge latency summary",
            f"Iterations: {iterations}",
            f"Warmup runs: {warmup}",
            f"Mean latency (ms): {mean(results['latency_ms']) if not results.empty else 0.0:.4f}",
            f"Median latency (ms): {median(results['latency_ms']) if not results.empty else 0.0:.4f}",
            f"Minimum latency (ms): {results['latency_ms'].min() if not results.empty else 0.0:.4f}",
            f"Maximum latency (ms): {results['latency_ms'].max() if not results.empty else 0.0:.4f}",
        ]
    )
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
        sample = prompt_for_sample(metadata)

    results, _, _ = run_latency_demo(sample, args.model_path, args.metadata_path, args.iterations, args.warmup)
    if not results.empty:
        print(f"Mean latency: {results['latency_ms'].mean():.4f} ms")


if __name__ == "__main__":
    main()
