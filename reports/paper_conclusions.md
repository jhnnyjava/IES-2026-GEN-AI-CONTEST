# AgriResilAI+ — Paper Conclusions and Key Results

This document collects the final experimental conclusions, evaluation numbers, limitations, and reproducibility notes intended for inclusion in the paper.

## Dataset
- Source: Kenya maize production table (challenge-provided CSV).
- Location in repo: `data/ken_maize_production.csv` (raw copy preserved at `data/raw/raw_ken_maize_production.csv`).
- Total rows used: 210 (train 168 / test 42 in current split).

## Feature Set
- Administrative context: `adlevel1`, `adlevel2`, `adlevel3`, `year`.
- Agronomic covariate: `areaharv` (harvested area).
- Deterministic synthetic environmental proxies (engineered): `rainfall_mm`, `temperature_c`, `humidity_pct`.

Notes: environmental proxies are deterministic, reproducible augmentations (see `src/feature_engineering.py`) and are intended to simulate missing weather covariates; they are not sensor measurements.

## Modeling Approach
- Targets: primary = `TOTMAZPROD`, secondary = `MAZYIELD` (analysis only).
- Candidate models: Linear Regression (baseline), Random Forest, Gradient Boosting.
- Training protocol: 80/20 holdout split on the dataset; 5-fold cross-validation on the training set for reporting stability.
- Initial iteration policy: tree ensembles used grid-searched settings; later experiments standardized ensemble iterations to 200 for convergence testing.

## Iteration Increase Experiment (50 → 200)

Motivation: ensure iterative learners (tree ensembles) have sufficient rounds to converge and to check whether limited iterations contributed to poor fit.

Changes applied:
- RandomForest: fixed `n_estimators=200` (previous grid contained 120 and 200).
- GradientBoosting: fixed `n_estimators=200` (previous grid ranged 120–180).
- Edge benchmark iterations increased to 200.

## Results Summary (primary target: `TOTMAZPROD`)

### Before (baseline settings)
- Linear Regression: MAE = 11,389.31 | RMSE = 14,145.84 | R² = 0.5670
- Gradient Boosting: MAE = 11,985.63 | RMSE = 15,346.09 | R² = 0.4904
- Random Forest: MAE = 13,103.90 | RMSE = 15,036.19 | R² = 0.5107

### After (200 iterations)
- Gradient Boosting (best): MAE = 9,435.71 | RMSE = 12,569.16 | R² = 0.6834
- Linear Regression: MAE = 10,987.11 | RMSE = 13,683.60 | R² = 0.6248
- Random Forest: MAE = 13,807.60 | RMSE = 16,302.31 | R² = 0.4674

Observations:
- Increasing ensemble iterations to 200 improved performance substantially for `TOTMAZPROD` in our Pi run: Gradient Boosting moved from R²≈0.49 to R²≈0.68 and reduced MAE by ~21% relative to the prior GB run.
- The best model after the experiment was Gradient Boosting (not Linear Regression as earlier), with a meaningful margin over the linear baseline.

## Secondary Target (`MAZYIELD`)
- The secondary target remains difficult to predict; models showed negative R² in earlier runs and volatile performance in grid searches. This indicates limited signal and/or insufficient explanatory features for this target.

## Edge Benchmark (Raspberry Pi 4)
- Single-sample mean latency: 8.22 ms (median 7.56 ms, p95 12.06 ms).
- Batch (size 8) mean total latency: 8.16 ms → per-sample ≈ 1.02 ms (p95 per-sample 1.51 ms).
- All benchmarks recorded to `reports/edge_latency.csv` and `reports/edge_latency.txt`.

## Interpretation and Recommendations for Paper
1. Iteration count matters for iterative ensemble learners; increasing to 200 unlocked substantially better GB performance on our dataset and hardware.
2. Even with improved iterations, results are constrained by dataset size (210 rows). Collecting additional labeled observations will likely yield larger gains than additional iterations.
3. Deterministic environmental proxies are a practical, reproducible compromise for edge deployment when sensor data are not available; however, the paper should explicitly note their synthetic nature and related limitations.
4. For future work: integrate sensor-based or gridded weather covariates, increase sample size, and explore localized models per administrative region to capture heterogeneity.

## Reproducibility and Artifacts
- Code: `src/` (training, preprocessing, feature engineering, inference, and edge demo).
- Trained model artifact: `models/best_model.pkl` and metadata `models/best_model_metadata.json`.
- Key figures (available in `figures/`): actual vs predicted, residuals, and feature importance for the selected model.
- Reports: `reports/results_summary.md`, `reports/paper_discussion_summary.txt`, `reports/dataset_profile.md`, and `reports/iteration_increase_comparison.md`.

## Exact Commands to Reproduce (recommended for Methods section)
```bash
# clone and prepare environment
git clone https://github.com/jhnnyjava/IES-2026-GEN-AI-CONTEST.git AgriResilAI
cd AgriResilAI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# train and evaluate (defaults use 200 iterations for ensembles and 5-fold CV)
python main.py --data-path data/ken_maize_production.csv

# run inference (uses a real dataset row by default)
python -m src.predict

# run edge benchmark (200 iterations)
python -m src.edge_demo --iterations 200 --batch-size 8
```

## Short author-ready conclusion paragraph
Increasing the iterative training budget for ensemble models (to 200 rounds) substantially improved Gradient Boosting performance on the Kenya maize production challenge, increasing explanatory power to R²≈0.68 and reducing MAE by ≈21% relative to the previous configuration. However, limited sample size and the use of synthetic environmental proxies constrain generalization; addressing these data limitations (more samples and observed environmental covariates) is the most promising next step to further improve predictive accuracy and decision reliability at the edge.
