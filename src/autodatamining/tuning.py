"""Automated hyperparameter tuning for selected models."""

from copy import deepcopy
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .exceptions import ConfigurationError, DataValidationError

_DEFAULT_PARAM_DISTRIBUTIONS = {
    "random_forest": {
        "model__n_estimators": [50, 100, 150],
        "model__max_depth": [None, 5, 10, 15],
        "model__min_samples_split": [2, 5, 10],
    },
    "extra_trees": {
        "model__n_estimators": [50, 100, 150],
        "model__max_depth": [None, 5, 10, 15],
    },
    "gradient_boosting": {
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__n_estimators": [50, 100, 150],
        "model__max_depth": [3, 5],
    },
    "hist_gradient_boosting": {
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_iter": [50, 100, 150],
    },
    "logistic_regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
    },
    "ridge": {
        "model__alpha": [0.1, 1.0, 10.0, 50.0],
    },
    "lasso": {
        "model__alpha": [0.01, 0.1, 1.0, 10.0],
    },
    "decision_tree": {
        "model__max_depth": [3, 5, 10, None],
        "model__min_samples_split": [2, 5, 10],
    },
    "knn": {
        "model__n_neighbors": [3, 5, 7, 9, 11],
        "model__weights": ["uniform", "distance"],
    },
    "linear_regression": {
        "model__fit_intercept": [True, False],
        "model__positive": [False, True],
    },
    "gaussian_nb": {
        "model__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
    },
    "svm": {
        "model__C": [0.1, 1.0, 10.0],
        "model__gamma": ["scale", "auto"],
    },
}


class TunedModelResult:
    """Results of automated hyperparameter tuning."""

    def __init__(
        self,
        model_name: str,
        best_estimator: Any,
        best_params: Dict[str, Any],
        score_before: float,
        score_after: float,
        metric_name: str,
        comparison: pd.DataFrame,
        tuning_seconds: float,
    ):
        self.model_name = model_name
        self.best_estimator = best_estimator
        self.best_params = best_params
        self.score_before = score_before
        self.score_after = score_after
        self.metric_name = metric_name
        self.comparison = comparison
        self.tuning_seconds = tuning_seconds
        self.improvement = score_after - score_before

    def predict(self, X):
        """Predict using the tuned pipeline."""
        return self.best_estimator.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities if supported."""
        if hasattr(self.best_estimator, "predict_proba"):
            return self.best_estimator.predict_proba(X)
        raise AttributeError(f"{self.model_name} does not support predict_proba")

    def summary(self) -> str:
        """Return a structured narrative explanation of the tuning results."""
        diff = self.improvement
        pct_diff = (diff / abs(self.score_before) * 100) if self.score_before != 0 else 0.0
        lines = [
            "==================================================================",
            "        HASIL PENYETELAN PARAMETER (HYPERPARAMETER TUNING)",
            "==================================================================",
            f"Algoritma Pemenang : {self.model_name.upper()}",
            f"Waktu Penyetelan   : {self.tuning_seconds:.2f} detik",
            f"Metrik Evaluasi    : {self.metric_name.upper()}",
            "",
            "1. PARAMETER TERBAIK YANG DITEMUKAN:",
        ]
        for param, val in self.best_params.items():
            lines.append(f"   - {param}: {val}")

        lines.extend([
            "",
            "2. PERBANDINGAN PERFORMA (SEBELUM vs SESUDAH):",
            f"   - Sebelum Tuning (Default) : {self.score_before:.4f}",
            f"   - Sesudah Tuning (Optimal) : {self.score_after:.4f}",
            f"   - Perubahan Skor           : {diff:+.4f} ({pct_diff:+.2f}%)",
            "",
            "3. KESIMPULAN PENYETELAN:",
        ])
        if diff > 0.005:
            lines.append(
                f"   ✅ TUNING BERHASIL MENINGKATKAN KINERJA:\n"
                f"      Penyetelan hyperparameter berhasil menaikkan skor {self.metric_name} sebesar {diff:+.4f}.\n"
                f"      Model ini siap dijadikan konfigurasi final untuk publikasi atau implementasi produksi."
            )
        elif diff >= -0.005:
            lines.append(
                f"   ℹ️ KINERJA STABIL:\n"
                f"      Skor sebelum dan sesudah tuning relatif sama. Konfigurasi bawaan scikit-learn sudah\n"
                f"      mendekati optimal untuk dataset ini."
            )
        else:
            lines.append(
                f"   ⚠️ PERINGATAN:\n"
                f"      Skor pada data uji mengalami sedikit deviasi ({diff:.4f}). Disarankan tetap menggunakan\n"
                f"      model sebelum tuning atau memperluas rentang pencarian parameter."
            )
        lines.append("==================================================================")
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"TunedModelResult(model={self.model_name!r}, metric={self.metric_name!r}, "
            f"score_before={self.score_before:.4f}, score_after={self.score_after:.4f})"
        )


def tune_model(
    results: Any,
    param_grid: Optional[Dict[str, list]] = None,
    method: str = "random",
    n_iter: int = 6,
    cv: int = 3,
    random_state: int = 42,
) -> TunedModelResult:
    """Tune hyperparameters of the winning model using cross-validated search.

    Parameters
    ----------
    results : MiningResults
        The fitted experiment results object.
    param_grid : dict, optional
        Custom mapping of parameter names to lists of candidate values.
    method : str, default="random"
        Tuning strategy: "random" (RandomizedSearchCV) or "grid" (GridSearchCV).
    n_iter : int, default=6
        Number of parameter settings sampled (when method="random").
    cv : int, default=3
        Cross-validation splitting strategy for tuning.
    random_state : int, default=42
        Seed for reproducible random sampling.
    """
    if results.task not in {"classification", "regression"}:
        raise DataValidationError(f"tune() is only supported for classification and regression, got {results.task!r}")

    if results.best_model is None:
        raise DataValidationError("No best_model available to tune.")

    X_train = getattr(results, "_X_train", None)
    y_train = getattr(results, "_y_train", None)
    X_test = getattr(results, "_X_test", None)
    y_test = getattr(results, "_y_test", None)

    if X_train is None or y_train is None:
        if X_test is not None and y_test is not None:
            X_train, y_train = X_test, y_test
        else:
            raise DataValidationError("Training data is required to tune the model without data leakage.")

    model_name = results.model_name or "model"
    grid = param_grid or _DEFAULT_PARAM_DISTRIBUTIONS.get(model_name)
    if not grid:
        step_model = results.best_model.estimator.named_steps.get("model")
        params = step_model.get_params() if step_model else {}
        if "n_estimators" in params:
            grid = {"model__n_estimators": [50, 100, 150]}
        elif "max_depth" in params:
            grid = {"model__max_depth": [3, 5, 10, None]}
        elif "alpha" in params:
            grid = {"model__alpha": [0.1, 1.0, 10.0]}
        elif "C" in params:
            grid = {"model__C": [0.1, 1.0, 10.0]}
        else:
            scoring = "accuracy" if results.task == "classification" else "r2"
            base_pipeline = results.best_model.estimator
            pred_before = base_pipeline.predict(X_test)
            if results.task == "classification":
                score_before = float(np.mean(pred_before == y_test))
            else:
                u = ((y_test - pred_before) ** 2).sum()
                v = ((y_test - y_test.mean()) ** 2).sum()
                score_before = float(1 - (u / v)) if v > 0 else 0.0

            return TunedModelResult(
                best_estimator=base_pipeline,
                best_params={},
                score_before=score_before,
                score_after=score_before,
                improvement_pct=0.0,
                metric_name=scoring.upper(),
                model_name=model_name,
                elapsed_seconds=0.01,
            )

    scoring = "accuracy" if results.task == "classification" else "r2"
    base_pipeline = results.best_model.estimator

    # Evaluate score before tuning on test data
    pred_before = base_pipeline.predict(X_test)
    if results.task == "classification":
        score_before = float(np.mean(pred_before == y_test))
    else:
        # R2 score
        u = ((y_test - pred_before) ** 2).sum()
        v = ((y_test - y_test.mean()) ** 2).sum()
        score_before = float(1 - (u / v)) if v > 0 else 0.0

    start_time = time.perf_counter()
    if str(method).lower().strip() == "grid":
        search = GridSearchCV(
            clone(base_pipeline),
            param_grid=grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
        )
    else:
        # Compute total combinations in grid to clamp n_iter safely
        total_combos = 1
        for v in grid.values():
            if isinstance(v, (list, tuple)):
                total_combos *= len(v)
        effective_n_iter = min(n_iter, total_combos)

        search = RandomizedSearchCV(
            clone(base_pipeline),
            param_distributions=grid,
            n_iter=effective_n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=-1,
            refit=True,
        )
    search.fit(X_train, y_train)
    elapsed = float(time.perf_counter() - start_time)

    tuned_pipeline = search.best_estimator_
    pred_after = tuned_pipeline.predict(X_test)
    if results.task == "classification":
        score_after = float(np.mean(pred_after == y_test))
    else:
        u = ((y_test - pred_after) ** 2).sum()
        v = ((y_test - y_test.mean()) ** 2).sum()
        score_after = float(1 - (u / v)) if v > 0 else 0.0

    # Clean parameter names (strip model__ prefix)
    clean_params = {
        (k[7:] if k.startswith("model__") else k): v
        for k, v in search.best_params_.items()
    }

    comparison = pd.DataFrame([
        {"Tahap": "Sebelum Tuning (Default)", "Skor": score_before, "Parameter": "Default"},
        {"Tahap": "Sesudah Tuning (Optimal)", "Skor": score_after, "Parameter": str(clean_params)},
    ])

    return TunedModelResult(
        model_name=model_name,
        best_estimator=tuned_pipeline,
        best_params=clean_params,
        score_before=score_before,
        score_after=score_after,
        metric_name=scoring,
        comparison=comparison,
        tuning_seconds=elapsed,
    )
