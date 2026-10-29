# Regression

Use `AutoDataMining(task="regression", target="value")`. The default training CV score is R². Holdout metrics report MAE, MSE, RMSE, and R². A negative R² is possible and means the evaluated predictions were worse than a constant mean baseline on that holdout.

```python
results = AutoDataMining(task="regression", target="price", models=["ridge", "random_forest"]).fit(data)
print(results.leaderboard)
results.plot(kind="residuals")
```
