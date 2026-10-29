import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification, make_regression
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


def test_builtin_voting_and_stacking_work_for_classification_and_regression():
    registry = EstimatorRegistry()
    params = {
        "voting": {
            "random_forest__n_estimators": 10,
            "extra_trees__n_estimators": 10,
        },
        "stacking": {
            "random_forest__n_estimators": 10,
            "extra_trees__n_estimators": 10,
        },
    }

    classification_X, classification_y = make_classification(
        n_samples=90, n_features=5, n_informative=3, n_redundant=0, random_state=3
    )
    classification_data = pd.DataFrame(classification_X, columns=[f"x{i}" for i in range(5)])
    classification_data["label"] = classification_y
    classification_result = AutoDataMining(
        task="classification", target="label", registry=registry,
        models=["voting", "stacking"], model_params=params, cv=2, random_state=42,
    ).fit(classification_data)
    assert set(classification_result.leaderboard["model"]) == {"voting", "stacking"}
    assert classification_result.leaderboard["status"].eq("ok").all()

    regression_X, regression_y = make_regression(
        n_samples=90, n_features=5, noise=8, random_state=3
    )
    regression_data = pd.DataFrame(regression_X, columns=[f"x{i}" for i in range(5)])
    regression_data["value"] = regression_y
    regression_result = AutoDataMining(
        task="regression", target="value", registry=registry,
        models=["voting", "stacking"], model_params=params, cv=2, random_state=42,
    ).fit(regression_data)
    assert set(regression_result.leaderboard["model"]) == {"voting", "stacking"}
    assert regression_result.leaderboard["status"].eq("ok").all()

    seeded = registry.get("voting", "classification").instantiate(random_state=17)
    assert seeded.get_params()["random_forest__random_state"] == 17
    assert seeded.get_params()["extra_trees__random_state"] == 17
