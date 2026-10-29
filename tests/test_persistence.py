import pandas as pd
import pytest
from sklearn.datasets import make_regression

from autodatamining import AutoDataMining, load_model


def test_saved_pipeline_predicts_new_data(tmp_path):
    X, y = make_regression(n_samples=60, n_features=3, random_state=1)
    data = pd.DataFrame(X, columns=["a", "b", "c"])
    data["y"] = y
    results = AutoDataMining(task="regression", target="y", models=["ridge"], cv=3).fit(data)
    path = results.save(tmp_path / "saved.joblib")
    with pytest.warns(UserWarning, match="trusted source"):
        model = load_model(path)
    prediction = model.predict(data[["c", "a", "b"]].iloc[:2])
    assert len(prediction) == 2
