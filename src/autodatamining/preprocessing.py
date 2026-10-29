"""Leakage-safe preprocessing and feature transformation helpers."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .detection import as_frame
from .exceptions import DataValidationError


def make_preprocessor(frame):
    """Build a sklearn transformer; it is fitted only inside each training pipeline."""
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    transformers = []
    if numeric:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipe, numeric))
    if categorical:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_pipe, categorical))
    if not transformers:
        raise DataValidationError("no usable feature columns were found")
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def select_features(X, y=None, method="variance", k=10, task="classification"):
    """Fit a feature selector and return (transformed_X, fitted_selector).

    SelectKBest and mutual information require a target; preprocessing is intentionally left to
    callers so this helper can be composed into their own validation pipeline.
    """
    frame = as_frame(X)
    values = frame.select_dtypes(include=[np.number]).copy()
    if values.shape[1] != frame.shape[1]:
        raise DataValidationError("feature selection helpers currently require numeric input")
    values = values.fillna(values.median()).fillna(0)
    if method == "variance":
        selector = VarianceThreshold()
    elif method in {"kbest", "mutual_information"}:
        if y is None:
            raise DataValidationError(f"{method} feature selection requires y")
        if method == "kbest":
            selector = SelectKBest(k=min(int(k), values.shape[1]))
        else:
            score = mutual_info_classif if task == "classification" else mutual_info_regression
            selector = SelectKBest(score_func=score, k=min(int(k), values.shape[1]))
    else:
        raise DataValidationError("method must be 'variance', 'kbest', or 'mutual_information'")
    transformed = selector.fit_transform(values, y)
    return transformed, selector


def reduce_dimensions(X, method="pca", n_components=2, random_state=42):
    """Fit PCA or TruncatedSVD on numeric tabular data and return (embedding, transformer)."""
    frame = as_frame(X)
    values = frame.select_dtypes(include=[np.number]).copy()
    if values.shape[1] != frame.shape[1]:
        raise DataValidationError("dimensionality reduction helpers currently require numeric input")
    values = values.fillna(values.median()).fillna(0)
    components = min(int(n_components), values.shape[1], max(1, len(values) - 1))
    if components < 1:
        raise DataValidationError("n_components must be positive")
    if method == "pca":
        transformer = PCA(n_components=components, random_state=random_state)
    elif method == "svd":
        transformer = TruncatedSVD(n_components=components, random_state=random_state)
    else:
        raise DataValidationError("method must be 'pca' or 'svd'")
    return transformer.fit_transform(values), transformer
