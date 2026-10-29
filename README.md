# AutoDataMining

**AutoDataMining** is a Python library for running consistent, inspectable data mining experiments with scikit-learn. It brings preprocessing, model comparison, evaluation, simple explanations, reports, and model persistence behind one API. It integrates existing estimators; it does not introduce new machine learning algorithms.

> **Status: alpha (0.1.1, unreleased).** The public API may change. Benchmark results depend on the dataset, split, metric, and estimator settings, and are not a guarantee of future performance. Review results before using a model in a consequential setting.

## Installation

```bash
python -m pip install autodatamining
```

For plots and optional SHAP explanations:

```bash
python -m pip install "autodatamining[visualization,explainability]"
```

## Quick start: classification

```python
import pandas as pd
from autodatamining import AutoDataMining

data = pd.read_csv("dataset.csv")
results = AutoDataMining(
    task="classification",
    target="label",
    test_size=0.2,
    cv=5,
    random_state=42,
    scoring="f1_weighted",
    models=["logistic_regression", "random_forest"],
).fit(data)

print(results.leaderboard)
print(results.best_model)
print(results.metrics)  # holdout metrics, plus CV score and fit time
results.plot()
results.report("classification-report.html")
results.save("best-model.joblib")
```

`leaderboard.cv_score` is calculated only on training folds. Other task metrics such as `accuracy`, `f1_weighted`, and `roc_auc` are holdout test values. The holdout is not used for selection when valid cross-validation scores exist.

## Other tasks

### Regression

```python
results = AutoDataMining(
    task="regression", target="price", cv=5, random_state=42,
    models=["ridge", "random_forest", "hist_gradient_boosting"],
).fit(data)
print(results.leaderboard[["model", "cv_score", "mae", "rmse", "r2", "status"]])
```

### Clustering

```python
results = AutoDataMining(
    task="clustering", models=["kmeans", "dbscan", "agglomerative"], random_state=42,
).fit(features)
print(results.leaderboard)
print(results.labels)  # labels from the selected model
```

The selected clustering model is ranked by silhouette score when that score is defined. Silhouette, Davies–Bouldin, and Calinski–Harabasz are internal measures, not accuracy against known classes. DBSCAN noise points have label `-1`.

### Anomaly detection

```python
results = AutoDataMining(
    task="anomaly", models=["isolation_forest", "lof"], random_state=42,
).fit(features)
print(results.leaderboard)  # anomaly_rate and anomaly_count
print(results.labels)  # -1 is an anomaly; +1 is an inlier
```

Without ground-truth labels, anomaly rate describes how many rows were flagged; it cannot rank model quality. If labels are available, pass their column as `target` and encode anomalies as `1`/`True` and normal rows as `0`/`False`. The runner excludes that target from the features, selects using stratified training-fold F1, and reports holdout precision, recall, F1, and AUC where defined. In labeled mode, `results.labels` contains selected-model holdout predictions; without labels, it contains fit-set detections.

## Algorithms and metrics

| Task | Built-in estimators | Default comparison score |
| --- | --- | --- |
| Classification | Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, KNN, Gaussian Naive Bayes, SVM | Accuracy (training CV) |
| Regression | Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, KNN, SVR | R² (training CV) |
| Clustering | K-Means, MiniBatch K-Means, DBSCAN, Agglomerative, Gaussian Mixture | Silhouette |
| Anomaly | Isolation Forest, Local Outlier Factor, One-Class SVM | Labeled F1 when labels are supplied; otherwise no quality ranking |

Classification holdout metrics include accuracy, balanced accuracy, weighted precision/recall/F1, and ROC-AUC / PR-AUC when the estimator and test labels support them. Regression reports MAE, MSE, RMSE, and R². Clustering reports silhouette, Davies–Bouldin, and Calinski–Harabasz when defined. Individual estimator failures are recorded in the leaderboard; successful models remain usable.

## Automatic task detection and data profile

`task="auto"` requires a target column. Text-like targets are treated as classification; small-cardinality integer targets are treated as classification; other numeric targets are treated as regression. Ambiguous low-cardinality numeric targets emit a warning. Pass `task` explicitly when the heuristic is not appropriate. Auto mode never silently launches clustering.

```python
from autodatamining import AutoDataMining

profile = AutoDataMining.profile(data, target="label")
print(profile["potential_issues"])
```

The profile includes row and column counts, per-column type, missing count and cardinality, duplicate rows, and a target distribution. It does not diagnose leakage or establish that a metric is suitable for a research question.

## Advanced usage

### Custom estimators

```python
from sklearn.linear_model import SGDClassifier
from autodatamining import AutoDataMining, EstimatorRegistry

registry = EstimatorRegistry()
registry.register("fast_sgd", SGDClassifier, tasks="classification",
                  params={"max_iter": 2000, "tol": 1e-3})
results = AutoDataMining(
    task="classification", target="label", registry=registry,
    models=["fast_sgd"], cv=3,
).fit(data)
```

Use unique names and register only estimators compatible with the declared task. Estimators are cloned before fitting. Model-specific parameters can be supplied with `model_params={"fast_sgd": {"alpha": 0.001}}`.

The module-level `register_estimator(name, estimator, tasks, params=None)` adds an estimator to the shared default registry for later runs.

### Feature selection and dimensionality reduction

```python
from autodatamining import select_features, reduce_dimensions

selected, selector = select_features(numeric_features, y, method="mutual_information",
                                     task="classification", k=5)
embedding, pca = reduce_dimensions(numeric_features, method="pca", n_components=2)
```

These convenience helpers fit directly to the data passed to them. For scientific evaluation, put selection or reduction inside a cross-validation `Pipeline` so each fold learns its transform from training data only.

### Flexible preprocessing and diagnostic plots

Preprocessing defaults to median imputation and standard scaling for numeric features, plus most-frequent imputation and one-hot encoding for categorical features. Override those choices per experiment:

```python
preprocessing = {
    "numeric_imputer": "median",       # mean, median, most_frequent, constant
    "numeric_scaler": "robust",        # standard, minmax, robust, none
    "categorical_imputer": "constant", # most_frequent, constant
    "categorical_fill_value": "unknown",
    "categorical_encoder": "ordinal", # onehot, ordinal
}
results = AutoDataMining(
    task="classification", target="label", preprocessing=preprocessing,
).fit(data)
```

Advanced users can pass a cloneable scikit-learn transformer as `preprocessing=...` to replace the complete default transformer, or use `numeric_transformer` and `categorical_transformer` in the options dictionary to replace just one branch. The custom transformer is cloned and fitted inside each training fold. See [the preprocessing guide](docs/preprocessing.md) for examples and supported options.

Inspect the raw input before fitting. The plots show missing-value counts and heatmaps, numeric correlation heatmaps, feature distributions, boxplots for potential outliers, and target distributions:

```python
fig, axes = AutoDataMining.plot_preprocessing(data, kind="overview", target="label")
fig.savefig("preprocessing-overview.png", dpi=150, bbox_inches="tight")

fig, ax = AutoDataMining.plot_preprocessing(data, kind="correlation", target="label")
fig.savefig("correlations.png", dpi=150, bbox_inches="tight")
```

`plot_preprocessing(data, kind=...)` is also available at package level. Supported kinds are `overview`, `missing_heatmap`, `heatmap`, `correlation`, `distributions`, `outliers`, and `target`. Plots need `autodatamining[visualization]`; functions return `(figure, axes)` and display only when `show=True`. These are exploratory diagnostics of the data passed in, not plots of a fitted transformation.

### Explainability

```python
importance = results.explain(method="auto")
```

Auto mode uses native tree importance or absolute coefficient magnitude where available, then falls back to holdout permutation importance for supervised tasks. `method="shap"` requires the optional extra. Importance describes model behavior and is not causal evidence. Unsupervised feature importance is not inferred.

### Plots, reports, and CSV

```python
results.plot(kind="leaderboard")
results.plot(kind="confusion_matrix")  # classification
results.plot(kind="roc_curve")          # binary classification
results.plot(kind="precision_recall_curve")  # binary classification
results.plot(kind="residuals")          # regression
results.plot(kind="clusters")           # clustering
print(results.classification_report())
results.to_csv("leaderboard.csv")
results.report("experiment.html")
```

Plots need `autodatamining[visualization]`. HTML reports include experiment metadata, metrics, leaderboard, and evaluation limitations; dataset rows are not included.

### Save and load

```python
from autodatamining import load_model

results.save("model.joblib")
model = load_model("model.joblib")
predictions = model.predict(new_data)
```

Prediction validates feature names and permits incoming DataFrame columns in a different order. Missing or unexpected columns raise an error. A joblib/pickle artifact can execute code when loaded: only load files from a trusted source. Artifacts record the scikit-learn version; cross-version compatibility is not guaranteed.

## Evaluation notes

- Supervised evaluation uses a train/test split and cross-validation on the training partition. Preprocessing is in the estimator pipeline and is fitted independently inside each fold.
- Classification uses stratification where class counts allow. Very small classes can prevent stratification or CV; the runner warns and reports what it could evaluate.
- CV score selects the supervised model when available. Holdout metrics are reported separately and should not be used repeatedly to tune the experiment.
- Clustering and unlabeled anomaly detection do not have ground-truth accuracy. Internal scores and anomaly rates have narrower meanings.
- Compare models using a metric tied to the intended use, and assess class imbalance, data leakage, temporal or grouped splits, calibration, latency, and deployment data drift separately.
- A top score in one experiment is not necessarily the best model for all datasets or real-world use.

## Troubleshooting

- **`task='auto' requires a target column`**: pass a target or explicitly choose clustering/anomaly.
- **A requested estimator failed**: inspect its `status` and `error` in `results.leaderboard`; try fewer parameters or another model.
- **Feature columns do not match**: supply the same feature names as training; DataFrame column order may differ.
- **A score is missing/NaN**: some metrics need multiple target classes or at least two non-noise clusters.
- **Plotting or SHAP import error**: install the corresponding optional extra.

## Development

```bash
python -m pip install -e ".[dev,visualization]"
python -m compileall -q src tests
python -m ruff check src tests examples
python -m pytest -q
python -m build
python -m twine check dist/*
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md). Contributions should include tests for changed behavior and preserve the distinction between training CV and holdout evaluation.

## PyPI release checklist

1. Review source, dependency licenses, project ownership, name availability, and version.
2. Run the test suite and build checks above.
3. Inspect wheel and source distribution contents.
4. Upload to TestPyPI and install into a clean virtual environment.
5. Publish to PyPI using an API token held outside the source tree or trusted publishing.
6. Verify the published install and create a version tag.

Version 0.1.0 was published to PyPI. Version 0.1.1 is a local unreleased update and has not been uploaded. Check the project owner, metadata, and release configuration again before publishing another version. No credentials belong in the repository.

## License

MIT; see [LICENSE](LICENSE). Review ownership and attribution before the first release.
