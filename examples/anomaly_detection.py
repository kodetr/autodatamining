"""Small, reproducible anomaly-detection example."""
import numpy as np
import pandas as pd

from autodatamining import AutoDataMining

rng = np.random.RandomState(42)
normal = rng.normal(size=(140, 2))
outliers = rng.uniform(low=-6, high=6, size=(12, 2))
X = np.vstack([normal, outliers])
y = np.r_[np.zeros(len(normal), dtype=int), np.ones(len(outliers), dtype=int)]
data = pd.DataFrame(X, columns=["x", "y"])
data["is_anomaly"] = y
results = AutoDataMining(task="anomaly", target="is_anomaly", models=["isolation_forest", "lof"],
                         random_state=42).fit(data)
print(results.leaderboard)
print(results.labels)
