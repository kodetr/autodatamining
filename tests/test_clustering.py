import pandas as pd
from sklearn.datasets import make_blobs

from autodatamining import AutoDataMining


def test_clustering_labels_and_internal_metrics():
    X, _ = make_blobs(n_samples=70, centers=3, random_state=4)
    data = pd.DataFrame(X, columns=["x", "y"])
    results = AutoDataMining(task="clustering", models=["kmeans", "agglomerative"], random_state=5).fit(data)
    assert len(results.labels) == len(data)
    assert "silhouette" in results.leaderboard
    assert results.selection_note.startswith("Selected by silhouette")
