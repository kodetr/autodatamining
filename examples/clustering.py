"""Small, reproducible clustering example."""
import pandas as pd
from sklearn.datasets import make_blobs

from autodatamining import AutoDataMining

X, _ = make_blobs(n_samples=120, centers=3, random_state=42)
features = pd.DataFrame(X, columns=["x", "y"])
results = AutoDataMining(task="clustering", models=["kmeans", "agglomerative"],
                         random_state=42).fit(features)
print(results.leaderboard)
print(results.labels)
