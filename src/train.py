from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import build_model_metadata, build_preprocessor, infer_feature_types, prepare_task_data
from .utils import (
    MODEL_METADATA_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    RESULTS_CSV_PATH,
    TARGET_COLUMNS,
    ensure_project_dirs,
    save_json,
)


@dataclass
class TrainingBundle:
    """Container for the best fitted pipeline and its holdout data."""

    target: str
    model_name: str
    pipeline: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray
    mae: float
    rmse: float
    r2: float
    cv_mae_mean: float | None
    cv_rmse_mean: float | None
    cv_r2_mean: float | None
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    target_rows: int
    target_zero_share: float


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _candidate_models() -> dict[str, Any]:
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def _safe_cross_validation(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float | None]:
    if len(y) < 8 or y.nunique() < 2:
        return {"cv_mae_mean": None, "cv_rmse_mean": None, "cv_r2_mean": None}

    folds = min(5, max(3, len(y) // 5))
    folds = min(folds, len(y))
    if folds < 2:
        return {"cv_mae_mean": None, "cv_rmse_mean": None, "cv_r2_mean": None}

    try:
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=folds,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
            n_jobs=None,
            error_score="raise",
        )
    except Exception:
        return {"cv_mae_mean": None, "cv_rmse_mean": None, "cv_r2_mean": None}

    return {
        "cv_mae_mean": float(-np.mean(scores["test_mae"])),
        "cv_rmse_mean": float(-np.mean(scores["test_rmse"])),
        "cv_r2_mean": float(np.mean(scores["test_r2"])),
    }


def train_models_for_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, TrainingBundle]:
    """Train and compare the candidate regressors for one target."""

    X, y, feature_columns = prepare_task_data(df, target_column)
    if y.nunique() < 2:
        raise ValueError(f"Target '{target_column}' does not contain enough variation for regression.")

    test_size = 0.2 if len(y) >= 20 else 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(X_train)
    numeric_features, categorical_features = infer_feature_types(X_train)

    rows: list[dict[str, Any]] = []
    best_bundle: TrainingBundle | None = None

    for model_name, estimator in _candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = _compute_metrics(y_test, predictions)
        cv_metrics = _safe_cross_validation(pipeline, X, y)

        row = {
            "target": target_column,
            "model": model_name,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "cv_mae_mean": cv_metrics["cv_mae_mean"],
            "cv_rmse_mean": cv_metrics["cv_rmse_mean"],
            "cv_r2_mean": cv_metrics["cv_r2_mean"],
            "rows": int(len(y)),
            "test_rows": int(len(y_test)),
            "target_zero_share": float((y == 0).mean()) if len(y) else 0.0,
        }
        rows.append(row)

        candidate_bundle = TrainingBundle(
            target=target_column,
            model_name=model_name,
            pipeline=pipeline,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            y_pred=np.asarray(predictions),
            mae=metrics["mae"],
            rmse=metrics["rmse"],
            r2=metrics["r2"],
            cv_mae_mean=cv_metrics["cv_mae_mean"],
            cv_rmse_mean=cv_metrics["cv_rmse_mean"],
            cv_r2_mean=cv_metrics["cv_r2_mean"],
            feature_columns=feature_columns,
            numeric_features=list(numeric_features),
            categorical_features=list(categorical_features),
            target_rows=int(len(y)),
            target_zero_share=float((y == 0).mean()) if len(y) else 0.0,
        )

        if best_bundle is None or candidate_bundle.mae < best_bundle.mae:
            best_bundle = candidate_bundle

    if best_bundle is None:
        raise RuntimeError(f"No model could be trained for target '{target_column}'.")

    return pd.DataFrame(rows), best_bundle


def train_all_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, TrainingBundle]:
    """Train all valid target tasks and select the best overall model."""

    all_results = []
    best_bundle: TrainingBundle | None = None

    for target in TARGET_COLUMNS:
        if target not in df.columns:
            continue
        target_results, target_bundle = train_models_for_target(df, target)
        all_results.append(target_results)
        if best_bundle is None or target_bundle.mae < best_bundle.mae:
            best_bundle = target_bundle

    if not all_results:
        raise ValueError("Neither TOTMAZPROD nor MAZYIELD was available for training.")

    if best_bundle is None:
        raise RuntimeError("Failed to select a best model.")

    return pd.concat(all_results, ignore_index=True), best_bundle


def save_results_table(results_df: pd.DataFrame, output_path: str | Path = RESULTS_CSV_PATH) -> Path:
    ensure_project_dirs()
    output = Path(output_path)
    results_df.sort_values(["mae", "rmse"], ascending=True).to_csv(output, index=False)
    return output


def save_best_model(bundle: TrainingBundle, model_path: str | Path = MODEL_PATH, metadata_path: str | Path = MODEL_METADATA_PATH) -> tuple[Path, Path]:
    """Persist the best fitted pipeline and a small JSON metadata file."""

    ensure_project_dirs()
    model_output = Path(model_path)
    metadata_output = Path(metadata_path)

    joblib.dump(bundle.pipeline, model_output)
    metadata = build_model_metadata(bundle.target, bundle.feature_columns, bundle.X_train, bundle.y_train)
    metadata.update(
        {
            "model_name": bundle.model_name,
            "mae": bundle.mae,
            "rmse": bundle.rmse,
            "r2": bundle.r2,
            "cv_mae_mean": bundle.cv_mae_mean,
            "cv_rmse_mean": bundle.cv_rmse_mean,
            "cv_r2_mean": bundle.cv_r2_mean,
        }
    )
    save_json(metadata_output, metadata)
    return model_output, metadata_output
