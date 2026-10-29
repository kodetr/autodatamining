"""Small, reproducible classification example."""
import pandas as pd
from sklearn.datasets import make_classification

from autodatamining import AutoDataMining

X, y = make_classification(n_samples=180, n_features=8, random_state=42)
data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
data["label"] = y
results = AutoDataMining(task="classification", target="label", models=["logistic_regression", "random_forest"],
                         cv=3, scoring="f1_weighted", random_state=42).fit(data)
print(results.leaderboard)
print(results.metrics)
