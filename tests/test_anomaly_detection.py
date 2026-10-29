import numpy as np
import pandas as pd

from autodatamining import AutoDataMining, DataValidationError


def test_anomaly_with_ground_truth():
    rng = np.random.RandomState(8)
    X = np.r_[rng.normal(size=(90, 2)), rng.normal(loc=5, size=(10, 2))]
    y = np.r_[np.zeros(90, dtype=int), np.ones(10, dtype=int)]
    data = pd.DataFrame(X, columns=["x", "y"])
    data["truth"] = y
    results = AutoDataMining(task="anomaly", target="truth", models=["isolation_forest"],
                             random_state=8).fit(data)
    assert len(results.labels) == results.experiment["test_rows"]
    assert 0 <= results.metrics["anomaly_rate"] <= 1
    assert {"precision", "recall", "f1"}.issubset(results.metrics)
    assert "cv_score" in results.leaderboard
    assert "holdout metrics are reported separately" in results.selection_note


def test_unlabeled_anomaly_warns_that_rate_is_not_quality():
    data = pd.DataFrame(np.random.RandomState(2).normal(size=(40, 2)), columns=["x", "y"])
    import pytest
    with pytest.warns(UserWarning, match="no labeled quality metric"):
        result = AutoDataMining(task="anomaly", models=["isolation_forest"], random_state=2).fit(data)
    assert "anomaly_rate" in result.leaderboard
    assert len(result.labels) == len(data)
    assert result.experiment["test_rows"] == 0


def test_missing_anomaly_target_column_is_rejected():
    data = pd.DataFrame({"x": range(10), "y": range(10)})
    import pytest
    with pytest.raises(DataValidationError, match="was not found"):
        AutoDataMining(task="anomaly", target="truth", models=["isolation_forest"]).fit(data)
