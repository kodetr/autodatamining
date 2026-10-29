import pandas as pd
from sklearn.datasets import make_classification

from autodatamining import AutoDataMining


def test_report_and_csv_exclude_dataset_values(tmp_path):
    X, y = make_classification(n_samples=60, n_features=4, random_state=5)
    data = pd.DataFrame(X, columns=["private-value-should-not-appear", "b", "c", "d"])
    data["target"] = y
    result = AutoDataMining(task="classification", target="target", models=["logistic_regression"],
                            cv=3).fit(data)
    report = result.report(tmp_path / "report.html")
    csv = result.to_csv(tmp_path / "leaderboard.csv")
    html = report.read_text()
    assert "Leaderboard" in html
    assert "private-value-should-not-appear" in html  # feature name metadata is safe
    assert str(data.iloc[0, 0]) not in html
    assert csv.exists()
