"""Tests for pipeline diagram plotting (results.plot(kind='pipeline'))."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification

from autodatamining import AutoDataMining


def test_pipeline_plot():
    X, y = make_classification(n_samples=50, n_features=3, n_informative=2, n_redundant=0, random_state=42)
    df = pd.DataFrame(X, columns=["f0", "f1", "f2"])
    df["target"] = y

    adm = AutoDataMining(
        task="classification",
        target="target",
        models=["logistic_regression"],
        random_state=42,
    )
    results = adm.fit(df)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    returned_ax = results.plot(kind="pipeline", ax=ax, show=False)

    assert returned_ax is ax
    assert "Diagram Alur Pipeline" in ax.get_title()
    plt.close(fig)
