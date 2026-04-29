# AgriResilAI+

A climate-aware, edge-deployable agricultural intelligence system with real environmental data integration and validated performance on both local and edge hardware.

This repository supports reproducible experimentation and a compact deployment footprint suitable for Raspberry Pi 4-class devices. The pipeline integrates real historical climate data (rainfall and temperature), trains regression models for Kenya maize production, and evaluates inference latency and throughput on both development hardware and edge devices.

## Key Features

- **Real Environmental Data Integration**: Utilizes real historical rainfall and temperature datasets (1991–2016) aggregated into yearly and multi-year features aligned with maize production periods.
- **Deterministic and Reproducible**: All experiments use `RANDOM_STATE = 42` for deterministic outcomes.
- **Edge-Optimized**: Designed to run efficiently on Raspberry Pi 4 with scikit-learn-only dependencies (no deep learning).
- **Feature Ablation Analysis**: Quantifies the impact of environmental variables on model performance via automated comparison experiments.
- **Publication-Ready**: Generates CSV reports, latency benchmarks, and publication-quality figures.

## Overview

The project trains on a Kenya maize production table integrated with real climate data and focuses on the primary target `TOTMAZPROD` (with `MAZYIELD` treated as a secondary analysis target). The workflow is fully deterministic and designed for publication-grade reproducibility.

Workflow highlights:

- Load and validate Kenya maize production CSV
- Integrate real rainfall and temperature datasets (monthly → yearly aggregation)
- Align climate data with maize production periods (e.g., "86-90" → mean of 1986–1990)
- Merge climate features with production data
- Train models: Linear Regression, Random Forest, Gradient Boosting
- Perform feature ablation (with/without environmental variables)
- Hyperparameter tuning with 5-fold cross-validation
- Model selection and artifact persistence
- Local and edge latency benchmarking

## Environmental Data Integration

Environmental variables are derived from real historical datasets aggregated into yearly features.

For **Rainfall** (monthly totals, 1991–2016):
- `annual_rainfall`: sum of all months
- `long_rains`: sum of March, April, May
- `short_rains`: sum of October, November, December
- `rainfall_std`: standard deviation of monthly values

For **Temperature** (monthly averages, 1991–2016):
- `avg_temp`: yearly mean temperature
- `temp_max`: yearly maximum temperature
- `temp_min`: yearly minimum temperature
- `temp_std`: standard deviation of monthly temperatures

Period alignment: For maize periods like "86-90", the corresponding climate values are computed as the mean of available yearly aggregates for 1986–1990. If years are missing, the nearest available window is used with fallback documented.

## Model Comparison and Feature Ablation

The pipeline automatically performs feature ablation when climate data is integrated:
- **With environmental features**: Full model including real rainfall and temperature
- **Without environmental features**: Baseline using only production metadata

Results are saved to `reports/feature_ablation_results.csv` and `reports/environmental_impact_analysis.txt`.

## Data Files Required

```
data/
├── ken_maize_production.csv          (required; primary production table)
├── rainfall.csv                       (optional; enables climate integration)
├── temperature.csv                    (optional; enables climate integration)
└── processed/
    └── merged_dataset.csv             (auto-generated after integration)
```

Both rainfall and temperature CSVs should have the format:
```
Year,Month Average,Value
1991,Jan Average,38.2847
1991,Feb Average,12.7492
...
```

## Reproducibility

All experiments are deterministic with `RANDOM_STATE = 42`. Key reproduction commands:

Train and evaluate with climate integration (automatic if CSVs present):

```bash
python main.py --data-path data/ken_maize_production.csv
```

View feature ablation results:

```bash
cat reports/environmental_impact_analysis.txt
```

Run portable inference:

```bash
python -m src.predict
```

Run local vs edge latency comparison:

```bash
python -m src.benchmark_compare --iterations 200 --batch-size 8
```

Outputs are written to `models/`, `reports/`, and `figures/` and include trained pipelines, latency CSVs, analysis files, and publication-quality plots.

## Expected Outputs

Automatically generated artifacts:

- `models/best_model.pkl` and `models/best_model_metadata.json`
- `reports/model_results.csv`
- `reports/feature_ablation_results.csv` (when climate data present)
- `reports/environmental_impact_analysis.txt` (when climate data present)
- `figures/actual_vs_predicted_*.png`
- `figures/feature_importance_*.png`
- `figures/edge_vs_local_latency.png`

## Limitations

- Dataset size is limited (210 rows in the challenge dataset).
- Maize production periods are aggregated (e.g., "86-90") rather than continuous annual data.
- Climate features are interpolated from historical aggregates where exact years are unavailable.
- `MAZYIELD` predictive power may be limited with current feature sets.

## Raspberry Pi 4 Notes

The pipeline uses scikit-learn only and is designed to remain lightweight on Raspberry Pi 4.

Recommended Pi steps:

```bash
cd ~/AgriResilAI
source .venv/bin/activate
python main.py --skip-edge-demo
python -m src.predict
python -m src.benchmark_compare --iterations 200 --batch-size 8
```

## Repository Layout

```text
AgriResilAI/
├── data/
│   ├── rainfall.csv
│   ├── temperature.csv
│   ├── ken_maize_production.csv
│   └── processed/
│       └── merged_dataset.csv
├── notebooks/
├── src/
│   ├── data_integration.py
│   ├── benchmark_compare.py
│   ├── train.py
│   ├── predict.py
│   └── ...
├── models/
├── reports/
├── figures/
├── requirements.txt
├── README.md
└── main.py
```
