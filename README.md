# AgriResilAI+

AgriResilAI+ is a reproducible, Raspberry Pi-compatible agricultural decision-intelligence pipeline for the Kenya maize production challenge. It combines a primary maize production table with deterministic synthetic environmental proxies so the system remains lightweight, interpretable, and suitable for an IEEE-style submission.

## Overview

The project trains on the challenge-provided Kenya maize production CSV and supports a primary target of `TOTMAZPROD` with `MAZYIELD` as a secondary target for analysis. The pipeline is cross-validated, tuned, and designed for offline edge deployment on Raspberry Pi 4.

The full workflow:

1. Load the primary Kenya maize dataset.
2. Validate the schema and preserve a raw copy.
3. Clean the records and keep only research-relevant fields.
4. Augment each record with deterministic environmental proxies:
   `rainfall_mm`, `temperature_c`, and `humidity_pct`.
5. Train Linear Regression, Random Forest, and Gradient Boosting models.
6. Tune the tree-based models with 5-fold cross-validation.
7. Save the best pipeline and metadata.
8. Generate decision intelligence for high-risk regions.
9. Produce plots, summary tables, and a Raspberry Pi latency benchmark.

## Dataset Origin

This repository expects the Kenya maize production table distributed in the challenge data package, derived from administrative agricultural statistics aligned with ICPAC/FAO-style reporting. Place it at:

```text
data/ken_maize_production.csv
```

The code preserves a raw copy in `data/raw/`, writes cleaned data to `data/cleaned/`, and writes augmented data to `data/augmented/`.

## Feature Engineering

The base modeling features are:

- `ADLEVEL1`
- `ADLEVEL2`
- `ADLEVEL3`
- `YEAR`
- `AREAHARV`

Synthetic environmental proxies are added deterministically from the administrative context and period label:

- `rainfall_mm`
- `temperature_c`
- `humidity_pct`

These proxy variables are used only to simulate missing environmental covariates. They are not treated as observed sensor data.

## Modeling Approach

The primary research target is `TOTMAZPROD`.

Models compared:

- Linear Regression baseline
- Random Forest Regressor as the main model
- Gradient Boosting Regressor as the comparison model

Validation protocol:

- 80/20 train/test split
- 5-fold cross-validation on the training split
- Hyperparameter tuning for Random Forest and Gradient Boosting via `GridSearchCV`

Reported metrics:

- MAE
- RMSE
- R²
- cross-validation mean and standard deviation

The decision layer classifies low-production forecasts as `HIGH RISK REGION`.

## Reproducibility

The project is deterministic given the input CSV and the fixed random seed `42`. All generated artifacts are written to disk:

- `data/raw/`
- `data/cleaned/`
- `data/augmented/`
- `models/`
- `reports/`
- `figures/`

## Run Training

```bash
python main.py
```

Optional arguments:

```bash
python main.py --data-path data/ken_maize_production.csv
python main.py --skip-edge-demo
python main.py --edge-iterations 100 --edge-batch-size 8
```

## Run Inference

Interactive prompt:

```bash
python -m src.predict
```

Key-value input:

```bash
python -m src.predict --values adlevel1=rift_valley adlevel2=nakuru adlevel3=nakuru year=86-90 areaharv=1200
```

JSON input:

```bash
python -m src.predict --json '{"adlevel1":"rift_valley","adlevel2":"nakuru","adlevel3":"nakuru","year":"86-90","areaharv":1200}'
```

## Run Edge Benchmark

```bash
python -m src.edge_demo --values adlevel1=rift_valley adlevel2=nakuru adlevel3=nakuru year=86-90 areaharv=1200 --iterations 50 --warmup 5 --batch-size 8
```

The benchmark writes latency statistics to `reports/edge_latency.csv` and `reports/edge_latency.txt`.

## Outputs for the Paper

Automatically generated files include:

- `reports/model_results.csv`
- `reports/results_summary.md`
- `reports/paper_discussion_summary.txt`
- `reports/dataset_profile.md`
- `figures/actual_vs_predicted_*.png`
- `figures/residuals_*.png`
- `figures/feature_importance_*.png`

## Raspberry Pi 4 Notes

The pipeline uses scikit-learn only and avoids deep learning frameworks. It is designed to remain lightweight on Raspberry Pi 4 while still supporting reproducible cross-validation, decision logic, and offline inference.

Recommended Pi steps:

```bash
cd ~/IES-2026-GEN-AI-CONTEST
source .venv/bin/activate
python main.py --skip-edge-demo
python -m src.predict --values adlevel1=rift_valley adlevel2=nakuru adlevel3=nakuru year=86-90 areaharv=1200
python -m src.edge_demo --values adlevel1=rift_valley adlevel2=nakuru adlevel3=nakuru year=86-90 areaharv=1200 --iterations 50 --warmup 5 --batch-size 8
```

## Limitations and Future Work

- Real-time weather sensor data is not yet integrated.
- The rainfall, temperature, and humidity fields are synthetic proxies for missing environmental covariates.
- Future work will integrate field telemetry and RFID-based traceability.
- The dataset is aggregated by period, so `YEAR` is a categorical temporal label rather than a continuous time series.
- Final performance depends on the quality of the real Kenya maize CSV supplied to the repository.

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

