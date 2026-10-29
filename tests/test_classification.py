import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from autodatamining import AutoDataMining, DataValidationError, load_model


def _data():
    X, y = make_classification(n_samples=100, n_features=5, random_state=3)
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    frame["category"] = np.where(X[:, 0] > 0, "high", "low")
    frame.loc[0, "x1"] = np.nan
    frame["label"] = y
    return frame


def test_classification_mixed_preprocessing_and_predictions(tmp_path):
    data = _data()
    results = AutoDataMining(task="classification", target="label", models=["logistic_regression"],
                             cv=3, scoring="f1_weighted", random_state=7).fit(data)
    assert results.leaderboard.iloc[0]["status"] == "ok"
    assert "f1_weighted" in results.metrics
    features = data.drop(columns="label").iloc[:4]
    reordered = features[features.columns[::-1]]
    assert len(results.predict(reordered)) == 4
    with pytest.raises(DataValidationError, match="feature columns"):
        results.predict(features.assign(unexpected=1))
    path = tmp_path / "model.joblib"
    results.save(path)
    with pytest.warns(UserWarning, match="execute code"):
        loaded = load_model(path)
    np.testing.assert_array_equal(loaded.predict(features), results.predict(features))


def test_auto_detection_and_missing_target_rows():
    data = _data()
    data.loc[1, "label"] = np.nan
    with pytest.warns(UserWarning, match="Dropped"):
        result = AutoDataMining(task="auto", target="label", models=["logistic_regression"], cv=3).fit(data)
    assert result.task == "classification"


def test_auto_without_target_does_not_silently_cluster():
    with pytest.raises(DataValidationError, match="requires a target"):
        AutoDataMining().fit(pd.DataFrame({"x": [1, 2, 3]}))


def test_native_and_permutation_explanations():
    data = _data()
    results = AutoDataMining(task="classification", target="label", models=["random_forest"],
                             cv=3, random_state=7).fit(data)
    native = results.explain(method="native")
    permutation = results.explain(method="permutation", n_repeats=2)
    assert "importance" in native
    assert set(permutation["feature"]) == set(data.columns) - {"label"}
