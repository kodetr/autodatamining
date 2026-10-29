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
    assert result.plot(kind="confusion_matrix", model_name="logistic_regression").get_xlabel()
    assert result.plot(kind="roc_curve").get_title()
    assert result.plot(kind="precision_recall_curve").get_title()
    assert "precision" in result.classification_report()


def test_regression_plots():
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=50, n_features=3, random_state=42)
    data = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    data["target"] = y
    result = AutoDataMining(task="regression", target="target", models=["linear_regression"], cv=3).fit(data)
    assert result.plot(kind="residuals").get_title()
    assert result.plot(kind="actual_vs_predicted").get_title()
    assert result.plot(kind="residuals", model_name="linear_regression").get_title()

