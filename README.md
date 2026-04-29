# AgriResilAI+

A lightweight, reproducible, edge-deployable agricultural decision intelligence system under data-scarce conditions.

This repository supports reproducible experimentation and a compact deployment footprint suitable for Raspberry Pi 4-class devices. The pipeline trains regression models for Kenya maize production under constrained data regimes and evaluates inference latency and throughput both on development hardware and on edge devices.

## Key Terminology
- Deterministic environmental proxies: compact, reproducible augmentations derived from administrative context (not observed sensor data). These proxies simulate missing environmental covariates and are NOT real sensor measurements. Use them only to approximate environmental variation when sensor or gridded weather data are unavailable.

## Overview

The project trains on a Kenya maize production table and focuses on the primary target `TOTMAZPROD` (with `MAZYIELD` treated as a secondary analysis target). The workflow is deterministic and designed for publication: training, evaluation (5-fold CV), artifact preservation, and edge benchmarking.

Workflow highlights:

- Data cleaning and schema validation
- Deterministic feature engineering (administrative + proxy environmental features)
- Model training: Linear Regression, Random Forest, Gradient Boosting
- Hyperparameter tuning with 5-fold CV
- Model selection and artifact saving (`models/`)
- Inference utilities and Raspberry Pi 4 latency benchmarking

## Edge vs Machine Inference Comparison

We evaluate inference in two deployment contexts:

- Local / development machine: representative of a researcher workstation or cloud VM.
- Raspberry Pi 4 (edge): representative of constrained on-farm / in-field compute.

Metrics defined:

- Latency: elapsed time per prediction in milliseconds (ms).
- Throughput: effective predictions per second when running batch inference (batch size > 1).
- Consistency: variability of predictions and latency (median, p95) across repeated runs.

The repository now contains a benchmark comparison utility that runs both contexts with the same preprocessing, model, and input to ensure a fair comparison.

## Experimental Contribution

We conducted an iteration-scaling experiment to test ensemble convergence under data scarcity. Key points:

- Experiment: increase iterative learner budget (n_estimators) from 50 → 200 for tree ensembles.
- Result: Gradient Boosting improved substantially — R² increased and MAE decreased (see `reports/paper_conclusions.md`).
- Conclusion: sufficient iterations materially improve ensemble convergence and downstream accuracy on small tabular datasets.

## Limitations

- Small dataset size (210 rows) limits generalization.
- Temporal aggregation: `YEAR` is a period label rather than a continuous time variable.
- Environmental covariates are deterministic proxies (not sensors) — treat results accordingly.
- `MAZYIELD` showed limited predictive power with currently available features.

## Reproducibility

All experiments are deterministic when run with the provided dataset and `RANDOM_STATE = 42`. Key reproduction commands:

Train and evaluate (default uses 5-fold CV and ensemble iterations = 200):

```bash
python main.py --data-path data/ken_maize_production.csv
```

Run portable inference (extracts a real row by default):

```bash
python -m src.predict
```

Run the edge benchmark (200 iterations, batch size 8):

```bash
python -m src.edge_demo --iterations 200 --batch-size 8
```

Run the new local vs edge comparison (publication-ready):

```bash
python -m src.benchmark_compare --iterations 200 --batch-size 8
```

Outputs are written to `models/`, `reports/`, and `figures/` and include the trained pipeline, latency CSVs, summary text files, and publication-quality plots.

## Outputs for the Paper

Automatically generated artifacts produced by the pipeline:

- `models/best_model.pkl` and `models/best_model_metadata.json`
- `reports/paper_conclusions.md` and `reports/iteration_increase_comparison.md`
- Latency reports: `reports/edge_latency.csv`, `reports/edge_vs_local_comparison.csv`
- Figures: `figures/actual_vs_predicted_*.png`, `figures/feature_importance_*.png`, `figures/edge_vs_local_latency.png`

## Raspberry Pi 4 Notes

The pipeline relies on scikit-learn and numeric Python packages. For fair edge comparisons, the `src.benchmark_compare` script constrains threading when simulating the edge environment and uses identical preprocessing and input data for both modes.

Recommended Pi steps:

```bash
cd ~/AgriResilAI
source .venv/bin/activate
python main.py --skip-edge-demo
python -m src.predict
python -m src.edge_demo --iterations 200 --batch-size 8
python -m src.benchmark_compare --iterations 200 --batch-size 8
```

## Repository Layout

```text
AgriResilAI/
├── data/
├── notebooks/
├── src/
├── models/
├── reports/
├── figures/
├── requirements.txt
├── README.md
└── main.py
```

