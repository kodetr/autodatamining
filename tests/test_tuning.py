"""Tests for automated hyperparameter tuning (results.tune)."""

import pandas as pd
from sklearn.datasets import make_classification, make_regression

from autodatamining import AutoDataMining
from autodatamining.exceptions import DataValidationError


def test_tune_classification_random_forest():
    X, y = make_classification(n_samples=100, n_features=4, n_informative=2, n_redundant=0, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y

    adm = AutoDataMining(
        task="classification",
        target="target",
        models=["random_forest"],
        random_state=42,
    )
    results = adm.fit(df)

    # Run tuning with small n_iter for speed
    tuned = results.tune(n_iter=3, cv=2, random_state=42)

    assert tuned.model_name == "random_forest"
    assert "n_estimators" in tuned.best_params or "max_depth" in tuned.best_params
    assert isinstance(tuned.score_before, float)
    assert isinstance(tuned.score_after, float)
    assert not tuned.comparison.empty

    # Test summary and predictions
    summary = tuned.summary()
    assert "HASIL PENYETELAN PARAMETER" in summary
    assert "RANDOM_FOREST" in summary

    preds = tuned.predict(results._X_test)
    assert len(preds) == len(results._X_test)


def test_tune_regression_ridge():
    X, y = make_regression(n_samples=80, n_features=3, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
    df["target"] = y

    adm = AutoDataMining(
        task="regression",
        target="target",
        models=["ridge"],
        random_state=42,
    )
    results = adm.fit(df)

    tuned = results.tune(n_iter=2, cv=2, random_state=42)
    assert tuned.model_name == "ridge"
    assert "alpha" in tuned.best_params
    assert isinstance(tuned.score_after, float)


def test_tune_invalid_task():
    adm = AutoDataMining(task="clustering", models=["kmeans"], random_state=42)
    df = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]], columns=["a", "b"])
    results = adm.fit(df)

    import pytest
    with pytest.raises(DataValidationError):
        results.tune()
