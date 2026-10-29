# Clustering

Use `AutoDataMining(task="clustering", models=[...])`. Preprocessing is fit on the full unsupervised dataset. The leaderboard reports internal metrics only; these do not measure agreement with ground-truth labels. The winner is selected by silhouette when defined. DBSCAN noise observations have label `-1` and are excluded from internal cluster scores.

```python
results = AutoDataMining(task="clustering", models=["kmeans", "dbscan"]).fit(features)
print(results.leaderboard)
labels = results.labels
```
