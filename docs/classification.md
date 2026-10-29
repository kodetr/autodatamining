# Classification

Use `AutoDataMining(task="classification", target="label")`. The default model catalog compares nine scikit-learn classifiers. `scoring` accepts a scikit-learn scorer name such as `f1_weighted` or `balanced_accuracy`. CV on training folds determines the winner; holdout scores are reported separately. Stratification is used when class counts permit.

```python
results = AutoDataMining(task="classification", target="label", scoring="f1_weighted").fit(data)
print(results.leaderboard)
print(results.metrics)
```

Weighted precision, recall, and F1 are returned alongside accuracy and balanced accuracy. ROC-AUC and PR-AUC are included only when score outputs and labels make them defined.
