import pandas as pd
from sklearn.datasets import make_classification

from autodatamining import AutoDataMining


def test_leaderboard_and_confusion_plots():
    X, y = make_classification(n_samples=60, n_features=4, random_state=5)
    data = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    data["target"] = y
    result = AutoDataMining(task="classification", target="target", models=["logistic_regression"],
                            cv=3).fit(data)
    assert result.plot(kind="leaderboard").get_title()
    assert result.plot(kind="confusion_matrix").get_xlabel()
    assert result.plot(kind="roc_curve").get_title()
    assert result.plot(kind="precision_recall_curve").get_title()
    assert "precision" in result.classification_report()
