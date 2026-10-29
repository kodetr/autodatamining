import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

from autodatamining import (
    AutoDataMining,
    ConfigurationError,
    EstimatorRegistry,
    ModelCompatibilityError,
)


class BrokenClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        raise RuntimeError("intentional test failure")


def test_custom_estimator_registry():
    registry = EstimatorRegistry()
    registry.register("sgd_custom", SGDClassifier, tasks="classification", params={"max_iter": 1000})
    with pytest.raises(ConfigurationError, match="already registered"):
        registry.register("sgd_custom", SGDClassifier, tasks="classification")
    X, y = make_classification(n_samples=80, n_features=4, random_state=2)
    frame = pd.DataFrame(X, columns=list("abcd"))
    frame["label"] = y
    result = AutoDataMining(task="classification", target="label", registry=registry,
                            models=["sgd_custom"], cv=3).fit(frame)
    assert result.leaderboard.iloc[0]["status"] == "ok"


def test_task_compatibility_is_validated():
    registry = EstimatorRegistry()
    with pytest.raises(ModelCompatibilityError, match="not registered"):
        AutoDataMining(task="regression", target="y", registry=registry,
                       models=["gaussian_nb"])


def test_one_failed_estimator_does_not_hide_successful_results():
    registry = EstimatorRegistry()
    registry.register("broken", BrokenClassifier, tasks="classification")
    X, y = make_classification(n_samples=80, n_features=4, random_state=2)
    frame = pd.DataFrame(X, columns=list("abcd"))
    frame["label"] = y
    result = AutoDataMining(task="classification", target="label", registry=registry,
                            models=["broken", "logistic_regression"], cv=3).fit(frame)
    broken = result.leaderboard.set_index("model").loc["broken"]
    assert broken["status"] == "failed"
    assert "intentional test failure" in broken["error"]
    assert result.best_model is not None
