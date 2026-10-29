import numpy as np
import pandas as pd

from autodatamining import AutoDataMining, reduce_dimensions, select_features


def test_profile_feature_selection_and_reduction():
    X = pd.DataFrame({"a": np.arange(12), "b": np.arange(12) ** 2, "constant": 1})
    y = np.array([0, 1] * 6)
    profile = AutoDataMining.profile(X)
    assert profile["rows"] == 12
    selected, _selector = select_features(X, method="variance")
    assert selected.shape[1] == 2
    reduced, _transformer = reduce_dimensions(X, n_components=2)
    assert reduced.shape == (12, 2)
    selected, _selector = select_features(X.drop(columns="constant"), y, method="kbest", k=1)
    assert selected.shape == (12, 1)
    svd_embedding, _svd = reduce_dimensions(X.drop(columns="constant"), method="svd", n_components=1)
    assert svd_embedding.shape == (12, 1)
