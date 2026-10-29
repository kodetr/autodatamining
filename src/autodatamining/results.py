"""Experiment result object and schema-validated prediction facade."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report as sklearn_classification_report

from .exceptions import DataValidationError
from .explainability import explain_model
from .persistence import save_model
from .reporting import write_report
from .visualization import plot_results


class ValidatedModel:
    """A fitted estimator that validates feature names and order on prediction."""

    def __init__(self, estimator, feature_names):
        self.estimator = estimator
        self.feature_names = list(feature_names)

    def _prepare(self, X):
        if isinstance(X, pd.DataFrame):
            missing = [name for name in self.feature_names if name not in X.columns]
            extra = [name for name in X.columns if name not in self.feature_names]
            if missing or extra:
                raise DataValidationError(f"feature columns do not match; missing={missing}, unexpected={extra}")
            return X.loc[:, self.feature_names]
        values = np.asarray(X)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise DataValidationError("prediction input must be a 2D table")
        frame = pd.DataFrame(values)
        if frame.shape[1] != len(self.feature_names):
            raise DataValidationError(f"expected {len(self.feature_names)} features, got {frame.shape[1]}")
        frame.columns = self.feature_names
        return frame

    def predict(self, X):
        return self.estimator.predict(self._prepare(X))

    def predict_proba(self, X):
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError("the selected estimator does not support predict_proba")
        return self.estimator.predict_proba(self._prepare(X))

    def decision_function(self, X):
        if not hasattr(self.estimator, "decision_function"):
            raise AttributeError("the selected estimator does not support decision_function")
        return self.estimator.decision_function(self._prepare(X))

    def fit_predict(self, X):
        if not hasattr(self.estimator, "fit_predict"):
            raise AttributeError("the selected estimator does not support fit_predict")
        return self.estimator.fit_predict(self._prepare(X))

    def __getattr__(self, name):
        # Expose fitted sklearn attributes such as named_steps, classes_, and feature_importances_.
        # Dunder lookup must stay on the wrapper: delegating __getstate__ would serialize its
        # underlying sklearn Pipeline as this object's state.
        if name.startswith("__"):
            raise AttributeError(name)
        state = object.__getattribute__(self, "__dict__")
        estimator = state.get("estimator")
        if estimator is None:
            raise AttributeError(name)
        return getattr(estimator, name)


class MiningResults:
    """Leaderboards, holdout metrics, labels, model, and experiment metadata."""

    def __init__(self, task, leaderboard, best_model, metrics, selection_note, experiment,
                 labels=None, X_test=None, y_test=None, model_name=None):
        self.task = task
        self.leaderboard = leaderboard
        self.best_model = best_model
        self.metrics = metrics
        self.selection_note = selection_note
        self.experiment = experiment
        self.labels = labels
        self._X_test = X_test
        self._y_test = y_test
        self.model_name = model_name

    def predict(self, X):
        """Predict with the selected fitted pipeline after validating feature columns."""
        return self.best_model.predict(X)

    def explain(self, method="auto", scoring=None, n_repeats=5):
        """Return feature importance; unsupported estimators get a clear explanation."""
        return explain_model(
            self.best_model, self._X_test, self._y_test, task=self.task,
            method=method, scoring=scoring, n_repeats=n_repeats,
            random_state=self.experiment.get("random_state", 42),
        )

    def classification_report(self, output_dict=True):
        """Return holdout per-class precision, recall, and F1 for classification runs."""
        if self.task != "classification" or self._y_test is None:
            raise DataValidationError("classification_report is only available for classification runs")
        report = sklearn_classification_report(
            self._y_test, self.predict(self._X_test), output_dict=output_dict, zero_division=0,
        )
        if not output_dict:
            return report
        rows = {name: value if isinstance(value, dict) else {"score": value}
                for name, value in report.items()}
        return pd.DataFrame.from_dict(rows, orient="index")

    def plot(self, kind="leaderboard", ax=None, show=False):
        """Create a matplotlib plot and return its axes."""
        return plot_results(self, kind=kind, ax=ax, show=show)

    def report(self, path):
        """Write a self-contained HTML report without embedding source dataset rows."""
        return write_report(self, path)

    def to_csv(self, path):
        """Export the model leaderboard to CSV and return the path."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.leaderboard.to_csv(destination, index=False)
        return destination

    def save(self, path):
        """Save the fitted preprocessing + model pipeline and experiment metadata."""
        return save_model(self.best_model, path, self.experiment)

    def __repr__(self):
        return (f"MiningResults(task={self.task!r}, model={self.model_name!r}, "
                f"metrics={self.metrics!r})")
