"""Augment the primary Kenya maize dataset with reproducible environmental proxies."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_raw_dataset
from src.feature_engineering import augment_and_persist
from src.preprocessing import standardize_maize_dataset
from src.utils import AUGMENTED_DATA_PATH, DEFAULT_RAW_DATA_PATH


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Augment the Kenya maize dataset with synthetic environmental features.")
    parser.add_argument("--source", type=str, default=str(DEFAULT_RAW_DATA_PATH), help="Path to the primary Kenya maize CSV.")
    parser.add_argument("--output", type=str, default=str(AUGMENTED_DATA_PATH), help="Path to the augmented CSV output.")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    raw = load_raw_dataset(args.source, preserve_copy=True)
    cleaned = standardize_maize_dataset(raw)
    augmented, output_path, _ = augment_and_persist(cleaned, output_path=Path(args.output))
    print(f"✓ Augmented dataset created: {output_path}")
    print(f"  Rows: {len(augmented)}")
    print(f"  Columns: {len(augmented.columns)}")


if __name__ == "__main__":
    main()
