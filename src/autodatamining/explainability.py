"""Feature importance helpers with explicit fallbacks and limitations."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .exceptions import DataValidationError


def explain_model(model, X, y=None, task=None, method="auto", scoring=None,
                  n_repeats=5, random_state=42):
    """Return a descending feature-importance DataFrame.

    Native importances are used when available. Permutation importance evaluates the entire
    fitted pipeline on the stored holdout data. Neither result implies a causal relationship.
    """
    if X is None:
        raise DataValidationError("explanations require retained evaluation features")
    pipeline = getattr(model, "estimator", model)
    if task in {"clustering", "anomaly"}:
        raise DataValidationError(
            "feature importance for unsupervised models is not defined by this API; use a domain-" 
            "specific explanation method or compare perturbations explicitly"
        )
    if method in {"auto", "native"}:
        final = pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else pipeline
        importances = getattr(final, "feature_importances_", None)
        if importances is None and hasattr(final, "coef_"):
            coef = np.asarray(final.coef_)
            importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        if importances is not None:
            preprocessor = pipeline.named_steps.get("preprocessor") if hasattr(pipeline, "named_steps") else None
            if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
                try:
                    names = list(preprocessor.get_feature_names_out())
                except (AttributeError, TypeError, ValueError):
                    names = []
            else:
                names = []
            if len(names) != len(importances):
                names = [f"feature_{i}" for i in range(len(importances))]
            return pd.DataFrame({"feature": names, "importance": np.asarray(importances, dtype=float)}).sort_values(
                "importance", ascending=False, ignore_index=True
            )
        if method == "native":
            raise DataValidationError(
                f"{type(final).__name__} does not expose native feature importance; try method='permutation'"
            )
    if method in {"auto", "permutation"}:
        if y is None:
            raise DataValidationError("permutation importance requires holdout target values")
        result = permutation_importance(
            model, X, y, scoring=scoring, n_repeats=n_repeats,
            random_state=random_state, n_jobs=1,
        )
        return pd.DataFrame({
            "feature": list(X.columns),
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance", ascending=False, ignore_index=True)
    if method == "shap":
        try:
            import shap
        except ImportError as exc:
            raise ImportError("SHAP support requires `pip install autodatamining[explainability]`") from exc
        try:
            explainer = shap.Explainer(model.predict, X)
            values = explainer(X)
            magnitudes = np.asarray(values.values)
            if magnitudes.ndim > 2:
                magnitudes = np.mean(np.abs(magnitudes), axis=tuple(range(magnitudes.ndim - 1)))
            else:
                magnitudes = np.mean(np.abs(magnitudes), axis=0)
            return pd.DataFrame({"feature": list(X.columns), "importance": magnitudes}).sort_values(
                "importance", ascending=False, ignore_index=True
            )
        except Exception as exc:
            raise DataValidationError(f"SHAP could not explain this fitted estimator: {exc}") from exc
    raise DataValidationError("method must be 'auto', 'native', 'permutation', or 'shap'")
