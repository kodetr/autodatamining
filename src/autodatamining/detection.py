"""Task inference and lightweight tabular data profiling."""

import warnings

import numpy as np
import pandas as pd

from .exceptions import DataValidationError


def as_frame(data):
    """Convert supported tabular inputs to a defensive DataFrame copy."""
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, (np.ndarray, list, tuple)):
        array = np.asarray(data)
        if array.ndim != 2:
            raise DataValidationError("data must be a two-dimensional table")
        frame = pd.DataFrame(array, columns=[f"feature_{i}" for i in range(array.shape[1])])
    else:
        raise DataValidationError("data must be a pandas DataFrame or a 2D array-like")
    if frame.empty:
        raise DataValidationError("data must contain at least one row and one column")
    if frame.columns.duplicated().any():
        raise DataValidationError("column names must be unique")
    if len(frame) < 2:
        raise DataValidationError("at least two rows are required")
    return frame


def detect_task(target):
    """Infer classification versus regression with a conservative cardinality heuristic."""
    series = pd.Series(target).dropna()
    if series.empty:
        raise DataValidationError("target contains no non-missing values")
    unique = int(series.nunique())
    if unique < 2:
        raise DataValidationError("target must contain at least two distinct values")
    if not pd.api.types.is_numeric_dtype(series.dtype):
        return "classification"
    # Integer/boolean targets with a small number of classes are commonly encoded as numbers.
    if pd.api.types.is_bool_dtype(series.dtype) or pd.api.types.is_integer_dtype(series.dtype):
        if unique <= min(20, max(2, int(np.sqrt(len(series))))):
            if unique > 10:
                warnings.warn(
                    "Numeric target has several distinct integer values; inferred classification. "
                    "Set task explicitly if these values are continuous measurements.",
                    UserWarning,
                    stacklevel=2,
                )
            return "classification"
        return "regression"
    # Floating targets are only classified when values are clearly discrete and low-cardinality.
    ratio = unique / max(1, len(series))
    if unique <= 10 and ratio <= 0.05:
        warnings.warn(
            "Low-cardinality numeric target inferred as classification. Set task explicitly if "
            "this is a regression target.",
            UserWarning,
            stacklevel=2,
        )
        return "classification"
    if unique <= 2:
        return "classification"
    return "regression"


def resolve_task(task, target):
    """Resolve explicit or automatic task; never silently choose unsupervised learning."""
    aliases = {
        "anomaly_detection": "anomaly",
        "outlier_detection": "anomaly",
        "association_rules": "association",
        "association_rule_mining": "association",
    }
    task = aliases.get(task, task)
    if task != "auto":
        if task in {"classification", "regression"} and target is None:
            raise DataValidationError(f"target is required for {task}")
        if task in {"clustering", "anomaly", "association"}:
            return task
        if task not in {"classification", "regression"}:
            raise DataValidationError(f"unsupported task: {task}")
        return task
    if target is None:
        raise DataValidationError(
            "task='auto' requires a target column; set task='clustering' or task='anomaly' "
            "explicitly for unsupervised analysis"
        )
    return detect_task(target)


def profile_data(data, target=None):
    """Return row/column counts, dtypes, null counts, cardinality, and target summary."""
    frame = as_frame(data)
    columns = {}
    for name in frame.columns:
        series = frame[name]
        columns[str(name)] = {
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "missing_fraction": float(series.isna().mean()),
            "unique": int(series.nunique(dropna=True)),
        }
    summary = {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_details": columns,
        "duplicate_rows": int(frame.duplicated().sum()),
        "potential_issues": [],
    }
    if frame.isna().any().any():
        summary["potential_issues"].append("Missing values are present; preprocessing will impute them.")
    if frame.duplicated().any():
        summary["potential_issues"].append("Duplicate rows are present.")
    if target is not None:
        if target not in frame.columns:
            raise DataValidationError(f"target column {target!r} was not found")
        y = frame[target]
        summary["target"] = {
            "name": str(target),
            "dtype": str(y.dtype),
            "missing": int(y.isna().sum()),
            "unique": int(y.nunique(dropna=True)),
            "value_counts": {str(k): int(v) for k, v in y.value_counts(dropna=False).items()},
        }
        if y.nunique(dropna=True) <= 1:
            summary["potential_issues"].append("Target contains fewer than two distinct non-missing values.")
    return summary
