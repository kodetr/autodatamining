import pandas as pd
from sklearn.datasets import make_regression

from autodatamining import AutoDataMining


def test_regression_scores_and_reproducibility():
    X, y = make_regression(n_samples=100, n_features=5, noise=7, random_state=2)
    data = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    data["target"] = y
    params = {"task": "regression", "target": "target", "models": ["ridge"], "cv": 3, "random_state": 9}
    first = AutoDataMining(**params).fit(data)
    second = AutoDataMining(**params).fit(data)
    assert first.leaderboard.iloc[0]["cv_score"] == second.leaderboard.iloc[0]["cv_score"]
    assert {"mae", "mse", "rmse", "r2"}.issubset(first.metrics)
    assert len(first.predict(data.drop(columns="target").iloc[:3])) == 3
