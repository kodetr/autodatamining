"""Small, reproducible regression example."""
import pandas as pd
from sklearn.datasets import make_regression

from autodatamining import AutoDataMining

X, y = make_regression(n_samples=160, n_features=7, noise=12, random_state=42)
data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
data["target"] = y
results = AutoDataMining(task="regression", target="target", models=["ridge", "random_forest"],
                         cv=3, random_state=42).fit(data)
print(results.leaderboard)
print(results.metrics)
