"""Main AutoDataMining experiment runner."""

import platform
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from .config import MiningConfig
from .detection import as_frame, profile_data, resolve_task
from .evaluation import (
    anomaly_metrics,
    classification_metrics,
    clustering_metrics,
    regression_metrics,
)
from .exceptions import DataValidationError, NoModelsSucceededError
from .preprocessing import make_preprocessor
from .registry import DEFAULT_REGISTRY
from .results import MiningResults, ValidatedModel
from .version import __version__

_DEFAULT_SCORING = {"classification": "accuracy", "regression": "r2"}


class AutoDataMining:
    """Run a reproducible benchmark over compatible scikit-learn estimators.

    Parameters mirror the quick-start API. ``model_params`` maps estimator names to sklearn
    parameters. For anomaly detection, optional ``target`` labels use 1/True for anomalies and
    0/False for normal observations.
    """

    def __init__(
        self,
        task="auto",
        target=None,
        models=None,
        test_size=0.2,
        cv=5,
        random_state=42,
        scoring=None,
        registry=None,
        model_params=None,
    ):
        self.config = MiningConfig(task=task, target=target, test_size=test_size, cv=cv,
                                   random_state=random_state, scoring=scoring)
        self.task = task
        self.target = target
        self.models = [models] if isinstance(models, str) else (list(models) if models is not None else None)
        self.registry = registry or DEFAULT_REGISTRY
        self.model_params = model_params or {}
        if not isinstance(self.model_params, dict):
            raise DataValidationError("model_params must be a mapping of model name to parameters")
        self.results_ = None
        if models is not None and task in {"classification", "regression", "clustering", "anomaly"}:
            self._selected_names(task)

    @staticmethod
    def profile(data, target=None):
        """Summarize a dataset without fitting models."""
        return profile_data(data, target=target)

    def fit(self, data):
        """Fit, evaluate, and compare estimators; return a :class:`MiningResults` object."""
        frame = as_frame(data)
        if self.target is not None and self.target not in frame.columns and self.task != "clustering":
            raise DataValidationError(f"target column {self.target!r} was not found")
        target_values = frame[self.target] if self.target is not None and self.target in frame.columns else None
        resolved = resolve_task(self.task, target_values)
        self.resolved_task_ = resolved
        if resolved in {"classification", "regression"}:
            outcome = self._fit_supervised(frame, resolved)
        elif resolved == "clustering":
            outcome = self._fit_clustering(frame)
        else:
            outcome = self._fit_anomaly(frame, target_values)
        self.results_ = outcome
        return outcome

    def _selected_names(self, task):
        names = self.models if self.models is not None else self.registry.names(task)
        if isinstance(names, str):
            names = [names]
        names = list(names)
        if not names:
            raise DataValidationError(f"no estimators selected for {task}")
        # Resolve compatibility before any model fitting, so typos fail fast.
        for name in names:
            self.registry.get(name, task)
        return names

    def _params(self, name):
        params = self.model_params.get(name, {})
        if not isinstance(params, dict):
            raise DataValidationError(f"parameters for model {name!r} must be a mapping")
        return params

    def _pipeline(self, frame, name, task):
        spec = self.registry.get(name, task)
        estimator = spec.instantiate(self.config.random_state, self._params(name))
        return Pipeline([("preprocessor", make_preprocessor(frame)), ("model", estimator)])

    def _fit_supervised(self, frame, task):
        if self.target not in frame.columns:
            raise DataValidationError(f"target column {self.target!r} was not found")
        clean = frame.loc[frame[self.target].notna()].copy()
        dropped = len(frame) - len(clean)
        if dropped:
            warnings.warn(f"Dropped {dropped} row(s) with missing target values.", UserWarning, stacklevel=2)
        y = clean.pop(self.target)
        if y.nunique() < 2:
            raise DataValidationError("target must contain at least two distinct non-missing values")
        X = clean
        if X.shape[1] == 0:
            raise DataValidationError("at least one feature column is required")
        stratify = None
        if task == "classification":
            counts = y.value_counts()
            if counts.min() >= 2:
                stratify = y
            else:
                warnings.warn("Stratified split is unavailable because a class has fewer than two rows.",
                              UserWarning, stacklevel=2)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=self.config.random_state,
                stratify=stratify,
            )
        except ValueError as exc:
            raise DataValidationError(f"could not create train/test split: {exc}") from exc

        scoring = self.config.scoring or _DEFAULT_SCORING[task]
        try:
            from sklearn.metrics import get_scorer
            get_scorer(scoring)
        except Exception as exc:
            raise DataValidationError(f"unsupported scikit-learn scoring name {scoring!r}") from exc
        cv_splits, cv_kind = self._cv_splitter(task, y_train)
        names = self._selected_names(task)
        rows, models = [], {}
        for name in names:
            row = {"model": name, "cv_score": float("nan"), "training_seconds": float("nan"), "status": "failed"}
            try:
                pipeline = self._pipeline(X_train, name, task)
                if cv_splits is not None:
                    scores = cross_val_score(clone(pipeline), X_train, y_train, cv=cv_splits,
                                             scoring=scoring, n_jobs=1, error_score="raise")
                    row["cv_score"] = float(np.mean(scores))
                    row["cv_std"] = float(np.std(scores))
                start = time.perf_counter()
                pipeline.fit(X_train, y_train)
                row["training_seconds"] = float(time.perf_counter() - start)
                prediction = pipeline.predict(X_test)
                if task == "classification":
                    scores_for_metrics = None
                    if hasattr(pipeline, "predict_proba"):
                        try:
                            scores_for_metrics = pipeline.predict_proba(X_test)
                        except (ValueError, AttributeError):
                            pass
                    metrics = classification_metrics(y_test, prediction, scores_for_metrics)
                else:
                    metrics = regression_metrics(y_test, prediction)
                row.update(metrics)
                row["status"] = "ok"
                models[name] = pipeline
            except Exception as exc:  # noqa: BLE001 - estimator failures are isolated by design
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        successes = [row for row in rows if row["status"] == "ok"]
        if not successes:
            details = "; ".join(f"{row['model']}: {row.get('error', 'failed')}" for row in rows)
            raise NoModelsSucceededError(f"all requested models failed: {details}")
        valid_cv = [row for row in successes if np.isfinite(row.get("cv_score", np.nan))]
        if valid_cv:
            winner = max(valid_cv, key=lambda row: row["cv_score"])
            selection_note = f"Selected by mean {scoring} across training-set {cv_kind} CV."
        else:
            winner = successes[0]
            selection_note = "CV was unavailable; selected the first successful estimator in request order."
            warnings.warn(selection_note, UserWarning, stacklevel=2)
        leaderboard = pd.DataFrame(rows)
        if valid_cv:
            leaderboard["rank"] = leaderboard["cv_score"].rank(method="min", ascending=False, na_option="bottom").astype(int)
        else:
            leaderboard["rank"] = np.arange(1, len(leaderboard) + 1)
        leaderboard = leaderboard.sort_values(["rank", "model"], kind="stable").reset_index(drop=True)
        model = ValidatedModel(models[winner["model"]], list(X.columns))
        best_metrics = {key: winner[key] for key in winner if key not in {"model", "status", "error"}}
        experiment = self._metadata(
            task, X, y, scoring, cv_kind, len(X_train), len(X_test),
            split_strategy="stratified_random" if stratify is not None else "random",
        )
        experiment["selected_model"] = winner["model"]
        experiment["selected_model_params"] = models[winner["model"]].named_steps["model"].get_params(deep=False)
        return MiningResults(
            task=task, leaderboard=leaderboard, best_model=model, metrics=best_metrics,
            selection_note=selection_note, experiment=experiment,
            X_test=X_test, y_test=y_test, model_name=winner["model"],
        )

    def _cv_splitter(self, task, y_train):
        folds = min(self.config.cv, len(y_train))
        if task == "classification":
            least = int(y_train.value_counts().min())
            folds = min(folds, least)
            if folds < 2:
                warnings.warn("Cross-validation skipped: a training class has fewer than two rows.",
                              UserWarning, stacklevel=2)
                return None, "unavailable"
            return StratifiedKFold(n_splits=folds, shuffle=True,
                                  random_state=self.config.random_state), f"stratified_{folds}_fold"
        if folds < 2:
            warnings.warn("Cross-validation skipped: fewer than two training rows.", UserWarning, stacklevel=2)
            return None, "unavailable"
        return KFold(n_splits=folds, shuffle=True, random_state=self.config.random_state), f"{folds}_fold"

    def _fit_clustering(self, frame):
        X = frame
        names = self._selected_names("clustering")
        preprocessor = make_preprocessor(X)
        transformed = preprocessor.fit_transform(X)
        rows, fitted = [], {}
        for name in names:
            row = {"model": name, "status": "failed", "training_seconds": float("nan")}
            try:
                spec = self.registry.get(name, "clustering")
                estimator = spec.instantiate(self.config.random_state, self._params(name))
                start = time.perf_counter()
                labels = estimator.fit_predict(transformed)
                row["training_seconds"] = float(time.perf_counter() - start)
                row.update(clustering_metrics(transformed, labels))
                row["clusters"] = int(len(set(labels)) - (1 if -1 in set(labels) else 0))
                row["noise_count"] = int(np.sum(np.asarray(labels) == -1))
                row["status"] = "ok"
                fitted[name] = (estimator, labels)
            except Exception as exc:  # noqa: BLE001 - estimator failures are isolated by design
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        successes = [r for r in rows if r["status"] == "ok"]
        if not successes:
            raise NoModelsSucceededError("all requested clustering models failed: " + "; ".join(
                f"{r['model']}: {r.get('error', 'failed')}" for r in rows))
        valid = [r for r in successes if np.isfinite(r.get("silhouette", np.nan))]
        if valid:
            winner = max(valid, key=lambda r: r["silhouette"])
            selection_note = "Selected by silhouette score; internal scores do not measure ground-truth accuracy."
            for row in rows:
                row["rank"] = (
                    1 + sum(np.isfinite(other.get("silhouette", np.nan)) and
                            other["silhouette"] > row.get("silhouette", np.nan)
                            for other in rows)
                    if np.isfinite(row.get("silhouette", np.nan)) else len(successes) + 1
                )
        else:
            winner = successes[0]
            selection_note = "Internal cluster metrics were undefined; selected the first successful estimator."
            for i, row in enumerate(rows, 1):
                row["rank"] = i
        board = pd.DataFrame(rows).sort_values(["rank", "model"], kind="stable").reset_index(drop=True)
        estimator, labels = fitted[winner["model"]]
        experiment = self._metadata("clustering", X, None, None, None, len(X), 0)
        experiment["selected_model"] = winner["model"]
        experiment["selected_model_params"] = estimator.get_params(deep=False)
        # Store the fitted transformer and clusterer in a pipeline-shaped object for persistence.
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        result = MiningResults(
            task="clustering", leaderboard=board, best_model=ValidatedModel(pipeline, list(X.columns)),
            metrics={k: v for k, v in winner.items() if k not in {"model", "status", "error"}},
            labels=np.asarray(labels), selection_note=selection_note,
            experiment=experiment,
            X_test=X, model_name=winner["model"],
        )
        return result

    def _fit_anomaly(self, frame, target_values):
        y_true = target_values
        if y_true is not None:
            valid = y_true.notna()
            if not valid.all():
                warnings.warn(f"Dropped {int((~valid).sum())} row(s) with missing anomaly labels.",
                              UserWarning, stacklevel=2)
                frame = frame.loc[valid].copy()
                y_true = y_true.loc[valid]
            X = frame.drop(columns=[self.target])
            if y_true.nunique() < 2:
                raise DataValidationError("ground-truth anomaly target must contain both normal and anomaly labels")
            counts = y_true.value_counts()
            stratify = y_true if counts.min() >= 2 else None
            if stratify is None:
                warnings.warn("Stratified anomaly holdout is unavailable because a label has fewer than two rows.",
                              UserWarning, stacklevel=2)
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_true, test_size=self.config.test_size,
                    random_state=self.config.random_state, stratify=stratify,
                )
            except ValueError as exc:
                raise DataValidationError(f"could not create anomaly holdout: {exc}") from exc
            split_strategy = "stratified_random" if stratify is not None else "random"
            cv_splits = min(self.config.cv, len(X_train), int(y_train.value_counts().min()))
            if cv_splits >= 2:
                cv = StratifiedKFold(n_splits=cv_splits, shuffle=True,
                                     random_state=self.config.random_state)
                cv_kind = f"stratified_{cv_splits}_fold"
            else:
                cv = None
                cv_kind = "unavailable"
                warnings.warn("Anomaly cross-validation skipped: a training label has fewer than two rows.",
                              UserWarning, stacklevel=2)
        else:
            X = frame
            X_train = X_test = X
            y_train = y_test = None
            cv = None
            cv_kind = None
            split_strategy = "none"
        if X.shape[1] == 0:
            raise DataValidationError("anomaly detection requires at least one feature column")
        names = self._selected_names("anomaly")
        rows, fitted = [], {}
        for name in names:
            row = {"model": name, "status": "failed", "training_seconds": float("nan")}
            try:
                start = time.perf_counter()
                if y_true is not None and cv is not None:
                    fold_f1 = []
                    fold_errors = []
                    for fit_indices, validation_indices in cv.split(X_train, y_train):
                        X_fold_train = X_train.iloc[fit_indices]
                        X_fold_valid = X_train.iloc[validation_indices]
                        y_fold_valid = y_train.iloc[validation_indices]
                        try:
                            fold_pipeline = self._pipeline(X_fold_train, name, "anomaly")
                            fold_pipeline.fit(X_fold_train)
                            fold_labels = fold_pipeline.predict(X_fold_valid)
                            fold_scores = None
                            if hasattr(fold_pipeline, "decision_function"):
                                fold_scores = -np.asarray(fold_pipeline.decision_function(X_fold_valid))
                            fold_f1.append(anomaly_metrics(fold_labels, y_fold_valid, fold_scores)["f1"])
                        except Exception as exc:  # noqa: BLE001 - record fold-level estimator failures
                            fold_errors.append(f"{type(exc).__name__}: {exc}")
                    if fold_f1:
                        row["cv_score"] = float(np.mean(fold_f1))
                        row["cv_std"] = float(np.std(fold_f1))
                    if fold_errors:
                        row["cv_error"] = "; ".join(fold_errors)
                estimator = self.registry.get(name, "anomaly").instantiate(
                    self.config.random_state, self._params(name)
                )
                if y_true is not None:
                    pipeline = Pipeline([("preprocessor", make_preprocessor(X_train)), ("model", estimator)])
                    pipeline.fit(X_train)
                    labels = pipeline.predict(X_test)
                    score_method = getattr(pipeline, "decision_function", None)
                    scores = -np.asarray(score_method(X_test)) if score_method is not None else None
                else:
                    preprocessor = make_preprocessor(X)
                    transformed = preprocessor.fit_transform(X)
                    if name.lower() == "lof" and estimator.get_params(deep=False).get("novelty", False):
                        estimator.fit(transformed)
                        # novelty=True is required for future-row prediction; training labels are
                        # recovered from LOF's fitted negative-outlier-factor threshold.
                        labels = np.where(estimator.negative_outlier_factor_ < estimator.offset_, -1, 1)
                    else:
                        labels = estimator.fit_predict(transformed)
                    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
                    score_method = getattr(estimator, "decision_function", None)
                    scores = -np.asarray(score_method(transformed)) if score_method is not None else None
                row["training_seconds"] = float(time.perf_counter() - start)
                row.update(anomaly_metrics(labels, y_test, scores))
                row["status"] = "ok"
                fitted[name] = (pipeline, labels)
            except Exception as exc:  # noqa: BLE001 - estimator failures are isolated by design
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        successes = [r for r in rows if r["status"] == "ok"]
        if not successes:
            raise NoModelsSucceededError("all requested anomaly models failed: " + "; ".join(
                f"{r['model']}: {r.get('error', 'failed')}" for r in rows))
        if y_true is not None:
            valid = [r for r in successes if np.isfinite(r.get("cv_score", np.nan))]
            if valid:
                winner = max(valid, key=lambda r: r["cv_score"])
                selection_note = (
                    f"Selected by mean training-set {cv_kind} F1 against ground-truth anomaly labels; "
                    "holdout metrics are reported separately."
                )
            else:
                winner = successes[0]
                selection_note = "Anomaly CV was unavailable; selected the first successful estimator in request order."
        else:
            winner = successes[0]
            selection_note = (
                "No ground-truth anomaly labels were supplied. Anomaly rate describes detections but cannot "
                "rank detection quality; selected the first successful estimator in request order."
            )
            warnings.warn("Anomaly detection has no labeled quality metric; leaderboard is not a quality ranking.",
                          UserWarning, stacklevel=2)
        if y_true is not None:
            valid = [r for r in successes if np.isfinite(r.get("cv_score", np.nan))]
            if valid:
                for row in rows:
                    row["rank"] = (
                        1 + sum(other["cv_score"] > row.get("cv_score", np.nan) for other in valid)
                        if np.isfinite(row.get("cv_score", np.nan)) else len(valid) + 1
                    )
        board = pd.DataFrame(rows)
        if "rank" in board:
            board = board.sort_values(["rank", "model"], kind="stable").reset_index(drop=True)
        pipeline, labels = fitted[winner["model"]]
        experiment = self._metadata(
            "anomaly", X, y_true, "f1" if y_true is not None else None, cv_kind,
            len(X_train), len(X_test) if y_true is not None else 0,
            split_strategy=split_strategy,
        )
        experiment["selected_model"] = winner["model"]
        experiment["selected_model_params"] = pipeline.named_steps["model"].get_params(deep=False)
        return MiningResults(
            task="anomaly", leaderboard=board, best_model=ValidatedModel(pipeline, list(X.columns)),
            metrics={k: v for k, v in winner.items() if k not in {"model", "status", "error"}},
            labels=np.asarray(labels), selection_note=selection_note,
            experiment=experiment,
            X_test=X_test, y_test=y_test, model_name=winner["model"],
        )

    def _metadata(self, task, X, y, scoring, cv_kind, train_rows, test_rows, split_strategy=None):
        if split_strategy is None:
            split_strategy = (
                "stratified_random" if task == "classification" else
                "random" if task == "regression" else "none"
            )
        return {
            "task": task,
            "target": self.target,
            "random_state": self.config.random_state,
            "test_size": self.config.test_size if task in {"classification", "regression"} or (task == "anomaly" and test_rows) else None,
            "cv": self.config.cv if task in {"classification", "regression"} or (task == "anomaly" and test_rows) else None,
            "cv_strategy": cv_kind,
            "split_strategy": split_strategy,
            "split_random_state": self.config.random_state,
            "scoring": scoring,
            "n_features": int(X.shape[1]),
            "train_rows": int(train_rows),
            "test_rows": int(test_rows),
            "feature_columns": [str(column) for column in X.columns],
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "python_version": platform.python_version(),
            "autodatamining_version": __version__,
        }
