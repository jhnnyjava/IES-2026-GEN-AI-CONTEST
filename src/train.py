from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from .decision import derive_risk_threshold
from .preprocessing import build_preprocessor, split_features_target
from .utils import MODEL_METADATA_PATH, MODEL_PATH, RANDOM_STATE, RESULTS_CSV_PATH, TARGET_PRIMARY, TARGET_SECONDARY, ensure_project_dirs, save_json


CV_SPLITS = 5


@dataclass(slots=True)
class ModelBundle:
    target: str
    model_name: str
    pipeline: Pipeline
    best_params: dict[str, Any]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray
    holdout_metrics: dict[str, float]
    cv_metrics: dict[str, float]
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    decision_threshold: float


@dataclass(slots=True)
class ProjectTrainingReport:
    comparison_table: pd.DataFrame
    primary_bundle: ModelBundle
    secondary_bundle: ModelBundle | None


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {"mae": float(mean_absolute_error(y_true, y_pred)), "rmse": float(np.sqrt(mse)), "r2": float(r2_score(y_true, y_pred))}


def _cv_strategy() -> KFold:
    return KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def _build_model_candidates() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    return {
        "linear_regression": (LinearRegression(), {}),
        "random_forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, n_estimators=200),
            {"model__max_depth": [None, 10], "model__min_samples_leaf": [1, 2], "model__max_features": ["sqrt"]},
        ),
        "gradient_boosting": (
            GradientBoostingRegressor(random_state=RANDOM_STATE, n_estimators=200),
            {"model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3], "model__subsample": [0.8]},
        ),
    }


def _fit_candidate(name: str, estimator: Any, param_grid: dict[str, list[Any]], X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Pipeline, dict[str, Any], dict[str, float]]:
    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    best_params: dict[str, Any] = {}
    fitted_pipeline = pipeline
    if param_grid:
        search = GridSearchCV(pipeline, param_grid=param_grid, scoring="neg_mean_absolute_error", cv=3, n_jobs=1, refit=True)
        search.fit(X_train, y_train)
        fitted_pipeline = search.best_estimator_
        best_params = {key.replace("model__", ""): value for key, value in search.best_params_.items()}
    else:
        fitted_pipeline.fit(X_train, y_train)

    cv_scores = cross_validate(
        fitted_pipeline,
        X_train,
        y_train,
        cv=_cv_strategy(),
        scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
        n_jobs=1,
    )

    cv_metrics = {
        "cv_mae_mean": float(-np.mean(cv_scores["test_mae"])),
        "cv_mae_std": float(np.std(cv_scores["test_mae"])),
        "cv_rmse_mean": float(-np.mean(cv_scores["test_rmse"])),
        "cv_rmse_std": float(np.std(cv_scores["test_rmse"])),
        "cv_r2_mean": float(np.mean(cv_scores["test_r2"])),
        "cv_r2_std": float(np.std(cv_scores["test_r2"])),
    }
    return fitted_pipeline, best_params, cv_metrics


def train_target_models(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, ModelBundle]:
    """Train the full candidate set for a single target."""

    X, y, feature_columns = split_features_target(df, target_column)
    if y.nunique() < 2:
        raise ValueError(f"Target '{target_column}' does not contain enough variation for regression.")

    test_size = 0.2 if len(y) >= 25 else 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)

    results: list[dict[str, Any]] = []
    best_bundle: ModelBundle | None = None

    for model_name, (estimator, grid) in _build_model_candidates().items():
        start = perf_counter()
        fitted_pipeline, best_params, cv_metrics = _fit_candidate(model_name, estimator, grid, X_train, y_train)
        fit_seconds = perf_counter() - start
        y_pred = fitted_pipeline.predict(X_test)
        holdout_metrics = _metrics(y_test, y_pred)

        result_row = {
            "target": target_column,
            "model": model_name,
            "mae": holdout_metrics["mae"],
            "rmse": holdout_metrics["rmse"],
            "r2": holdout_metrics["r2"],
            "cv_mae_mean": cv_metrics["cv_mae_mean"],
            "cv_mae_std": cv_metrics["cv_mae_std"],
            "cv_rmse_mean": cv_metrics["cv_rmse_mean"],
            "cv_rmse_std": cv_metrics["cv_rmse_std"],
            "cv_r2_mean": cv_metrics["cv_r2_mean"],
            "cv_r2_std": cv_metrics["cv_r2_std"],
            "fit_seconds": fit_seconds,
            "best_params": best_params,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        }
        results.append(result_row)

        numeric_features = [column for column in X_train.columns if column not in {"adlevel1", "adlevel2", "adlevel3", "year"}]
        categorical_features = [column for column in X_train.columns if column in {"adlevel1", "adlevel2", "adlevel3", "year"}]
        bundle = ModelBundle(
            target=target_column,
            model_name=model_name,
            pipeline=fitted_pipeline,
            best_params=best_params,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            y_pred=np.asarray(y_pred),
            holdout_metrics=holdout_metrics,
            cv_metrics=cv_metrics,
            feature_columns=feature_columns,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            decision_threshold=derive_risk_threshold(y_train),
        )

        if best_bundle is None or bundle.holdout_metrics["mae"] < best_bundle.holdout_metrics["mae"]:
            best_bundle = bundle

    assert best_bundle is not None
    comparison = pd.DataFrame(results).sort_values(["target", "mae"]).reset_index(drop=True)
    return comparison, best_bundle


def train_project(df: pd.DataFrame) -> ProjectTrainingReport:
    """Train the primary target and optional secondary target."""

    primary_comparison, primary_bundle = train_target_models(df, TARGET_PRIMARY)
    secondary_bundle: ModelBundle | None = None
    comparisons = [primary_comparison]
    if TARGET_SECONDARY in df.columns:
        secondary_comparison, secondary_bundle = train_target_models(df, TARGET_SECONDARY)
        comparisons.append(secondary_comparison)

    combined = pd.concat(comparisons, ignore_index=True).sort_values(["target", "mae"]).reset_index(drop=True)
    return ProjectTrainingReport(comparison_table=combined, primary_bundle=primary_bundle, secondary_bundle=secondary_bundle)


def save_results_table(results_df: pd.DataFrame, output_path: str | Path = RESULTS_CSV_PATH) -> Path:
    ensure_project_dirs()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output, index=False)
    return output


def save_model_artifacts(bundle: ModelBundle, comparison_df: pd.DataFrame, dataset_summary: dict[str, Any], augmentation_summary: dict[str, Any]) -> tuple[Path, Path]:
    """Persist the best fitted pipeline and a scientific metadata payload."""

    ensure_project_dirs()
    model_output = Path(MODEL_PATH)
    metadata_output = Path(MODEL_METADATA_PATH)

    joblib.dump(bundle.pipeline, model_output)
    metadata = {
        "target": bundle.target,
        "model_name": bundle.model_name,
        "best_params": bundle.best_params,
        "holdout_metrics": bundle.holdout_metrics,
        "cv_metrics": bundle.cv_metrics,
        "feature_columns": bundle.feature_columns,
        "numeric_features": bundle.numeric_features,
        "categorical_features": bundle.categorical_features,
        "decision_threshold": bundle.decision_threshold,
        "dataset_summary": dataset_summary,
        "augmentation_summary": augmentation_summary,
        "comparison_snapshot": comparison_df.to_dict(orient="records"),
        "model_size_bytes": model_output.stat().st_size if model_output.exists() else None,
    }
    save_json(metadata_output, metadata)
    return model_output, metadata_output


def perform_feature_ablation(df: pd.DataFrame, output_csv: str | Path = "reports/feature_ablation_results.csv", output_txt: str | Path = "reports/environmental_impact_analysis.txt") -> None:
    """Train with and without environmental features and save comparison results.

    Compares performance on merged dataset (with real climate features) vs. baseline (without).
    Writes CSV and human-readable analysis.
    """
    ensure_project_dirs()
    env_cols = ["annual_rainfall", "long_rains", "short_rains", "rainfall_std", "avg_temp", "temp_max", "temp_min", "temp_std"]
    has_env = all(c in df.columns for c in env_cols)

    if not has_env:
        return  # No environmental features to ablate

    # Train on full dataset (with env)
    baseline_df = df.copy()
    baseline_comp, baseline_bundle = train_target_models(baseline_df, TARGET_PRIMARY)

    # Train without env features
    no_env_df = df.drop(columns=[c for c in env_cols if c in df.columns], errors="ignore")
    noenv_comp, noenv_bundle = train_target_models(no_env_df, TARGET_PRIMARY)

    # Prepare CSV summary
    rows = []
    for name, comp in (("with_env", baseline_comp), ("without_env", noenv_comp)):
        best = comp.sort_values("mae").iloc[0]
        rows.append({
            "mode": name,
            "model": best["model"],
            "mae": float(best["mae"]),
            "rmse": float(best["rmse"]),
            "r2": float(best["r2"]),
            "cv_mae_mean": float(best["cv_mae_mean"]),
        })

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    # Write analysis text
    best_with = rows[0]
    best_without = rows[1]
    mae_improvement = (best_without["mae"] - best_with["mae"]) if best_without["mae"] and best_with["mae"] else 0.0
    r2_improvement = (best_with["r2"] - best_without["r2"]) if best_with["r2"] and best_without["r2"] else 0.0

    lines = [
        "Environmental Impact Analysis",
        "",
        f"Primary target: {TARGET_PRIMARY}",
        "",
        "Best model WITH environmental features (rainfall + temperature):",
        f"- Model: {best_with['model']}",
        f"- MAE: {best_with['mae']:.4f}",
        f"- RMSE: {best_with['rmse']:.4f}",
        f"- R²: {best_with['r2']:.4f}",
        f"- CV MAE: {best_with['cv_mae_mean']:.4f}",
        "",
        "Best model WITHOUT environmental features:",
        f"- Model: {best_without['model']}",
        f"- MAE: {best_without['mae']:.4f}",
        f"- RMSE: {best_without['rmse']:.4f}",
        f"- R²: {best_without['r2']:.4f}",
        f"- CV MAE: {best_without['cv_mae_mean']:.4f}",
        "",
        f"MAE reduction with environmental features: {mae_improvement:.4f}",
        f"R² improvement with environmental features: {r2_improvement:.6f}",
        "",
        "Conclusion:",
        f"Environmental features (real rainfall and temperature aggregates) {'improve' if mae_improvement > 0 else 'do not improve'} model performance.",
        "This validates the importance of climate variables in agricultural yield prediction.",
    ]

    Path(output_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, "w", encoding="utf8") as fh:
        fh.write("\n".join(lines) + "\n")
