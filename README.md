# AgriResilAI+

AgriResilAI+ is a lightweight, reproducible research codebase for the Kenya maize production dataset. The project is structured for an IEEE-style competition paper and focuses on transparent preprocessing, classical regression models, evaluation, and Raspberry Pi-friendly inference.

## Project Overview

The workflow:

1. Loads the raw Kenya maize production CSV.
2. Audits the dataset and preserves a raw copy.
3. Cleans the columns and removes geometry and map metadata.
4. Trains multiple regression models for the available target tasks.
5. Compares the models using MAE, RMSE, and R2.
6. Saves the best pipeline as a joblib artifact.
7. Runs a simple command-line inference demo.
8. Measures edge latency for Raspberry Pi deployment.
9. Writes report-ready summaries and figures.

## Dataset Description

The source dataset is the Kenya maize production statistics CSV. Important source fields include:

- `ADLEVEL1`, `ADLEVEL2`, `ADLEVEL3`
- `TOTMAZPROD`
- `MAZYIELD`
- `AREAHARV`
- `YEAR`

The dataset also contains geometry and metadata columns such as `_id`, `FID`, `the_geom`, `AREA`, `PERIMETER`, `REGIONS_`, `REGIONS_ID`, `SQKM`, `ADMSQKM`, `CODE`, `ADMINID`, and `COUNTRY`. These are removed during preprocessing because they do not support model training.

## Features Used

The default modeling feature set is kept intentionally small and interpretable:

- Categorical features: `ADLEVEL1`, `ADLEVEL2`, `ADLEVEL3`, `YEAR`
- Numeric feature: `AREAHARV`

The targets are evaluated separately for:

- `TOTMAZPROD`
- `MAZYIELD`

`YEAR` is treated as a categorical field because the source data are aggregated into a period label such as `86-90`, not a true yearly time series.

## Preprocessing Steps

The preprocessing pipeline:

- normalizes column names to `snake_case`
- drops geometry and pure metadata columns
- converts numeric-looking values safely
- preserves the raw CSV as a copy in `data/raw_ken_maize_production.csv`
- saves the cleaned dataset to `data/cleaned_maize_data.csv`
- imputes missing numeric values with the median
- imputes missing categorical values with the most frequent category
- one-hot encodes categorical variables
- scales numeric variables for compatibility with linear regression

Zero-production rows are not automatically removed. They are retained if present, and the training code reports the zero share for each target so the paper can discuss whether the target is sparse or stable.

## Model Comparison

The project compares three regression models:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The best model is selected primarily by holdout MAE, with RMSE and R2 reported as supporting metrics.

## How to Train

Place the raw CSV at:

```text
data/ken_maize_production.csv
```

Then run:

```bash
python main.py
```

This will:

- audit the dataset
- clean and save the processed CSV
- train and compare the regression models
- save the best model to `models/best_model.pkl`
- save model metadata to `models/best_model_metadata.json`
- generate plots in `figures/`
- write results to `reports/model_results.csv`
- produce summary text files in `reports/`
- run a small inference demo
- optionally run an edge latency demo

## How to Evaluate

The main workflow generates:

- `reports/model_results.csv`
- `figures/actual_vs_predicted_*.png`
- `figures/residuals_*.png`
- `figures/feature_importance_*.png` when available
- `reports/results_summary.md`
- `reports/paper_discussion_summary.txt`

The evaluation metrics are:

- MAE
- RMSE
- R2

## How to Run Inference

After training, run:

```bash
python -m src.predict
```

You can also pass values explicitly:

```bash
python -m src.predict --values adlevel1=rift_valley adlevel2=nakuru adlevel3=nakuru year=86-90 areaharv=1200
```

Or pass a JSON object:

```bash
python -m src.predict --json '{"adlevel1":"rift_valley","adlevel2":"nakuru","adlevel3":"nakuru","year":"86-90","areaharv":1200}'
```

## Raspberry Pi Setup Notes

The saved model is a standard scikit-learn pipeline and is appropriate for Raspberry Pi 4 deployment because it avoids deep learning and large external dependencies.

Recommended setup:

- use a Raspberry Pi OS Python environment
- install the packages from `requirements.txt`
- copy `models/best_model.pkl` and `models/best_model_metadata.json` to the device
- run `python -m src.predict` for interactive inference
- run `python -m src.edge_demo` to measure latency offline

## Validated Pi4 Runbook (Reproduced)

The following sequence was executed successfully on Raspberry Pi 4.

1. Enter project and activate virtual environment:

```bash
cd ~/IES-2026-GEN-AI-CONTEST
source .venv/bin/activate
```

2. Pull latest code:

```bash
git pull origin main
```

3. Generate a dataset on Pi (for testing/reproducibility when real CSV is unavailable):

```bash
python create_dataset.py
```

4. Train and save model artifacts on Pi:

```bash
python main.py --skip-edge-demo
```

5. Verify model files:

```bash
ls models
```

Expected files:

```text
best_model.pkl
best_model_metadata.json
```

6. Run inference test:

```bash
python -m src.predict --values adlevel1=rift_valley adlevel2=nakuru adlevel3=nakuru year=86-90 areaharv=1200
```

Observed output during validation:

```text
Predicted mazyield: 12.3500
```

7. Run edge latency benchmark:

```bash
python -m src.edge_demo --values adlevel1=rift_valley adlevel2=nakuru adlevel3=nakuru year=86-90 areaharv=1200 --iterations 50 --warmup 5
```

Observed output during validation:

```text
Mean latency: 97.4793 ms
```

Notes:

- If you use a real maize CSV, place it at `data/ken_maize_production.csv` and rerun training.
- The synthetic dataset step is only a fallback for reproducibility and device bring-up.

## Limitations

- The dataset is aggregated by period, so the `YEAR` field is not a true annual time series.
- The project currently uses only the maize CSV; climate covariates can be added later if available.
- If production or yield values are sparse, the results may vary by target task.
- The geometry and administrative metadata are removed because they are not useful for a compact regression baseline.

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
