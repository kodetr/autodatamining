# Explainability

`results.explain()` uses native tree importance or absolute coefficient magnitude where supported, with holdout permutation importance as the automatic fallback. Request `method="permutation"` explicitly to use holdout permutation importance. SHAP is optional via `pip install "autodatamining[explainability]"` and `results.explain(method="shap")`.

Importance describes model behavior in the evaluated data. It does not demonstrate causal effects. The first release does not invent global feature importance for clustering or unlabeled anomaly detection.
