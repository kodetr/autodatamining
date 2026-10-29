"""Leakage-safe preprocessing and feature transformation helpers."""

import numpy as np
from sklearn.base import clone
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
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from .detection import as_frame
from .exceptions import DataValidationError


def make_preprocessor(frame, config=None):
    """Build a configurable sklearn transformer without fitting it.

    ``config`` may be a strategy dictionary or any cloneable sklearn transformer. Dictionary
    options are ``numeric_imputer``, ``numeric_fill_value``, ``numeric_scaler``,
    ``categorical_imputer``, ``categorical_fill_value``, ``categorical_encoder``,
    ``numeric_transformer``, and ``categorical_transformer``. Custom transformers replace the
    corresponding default pipeline. The returned transformer is fitted inside each training fold.
    """
    if config is not None and not isinstance(config, dict):
        if not hasattr(config, "fit") or not hasattr(config, "transform"):
            raise DataValidationError("preprocessing must be a strategy dictionary or sklearn transformer")
        try:
            return clone(config)
        except (TypeError, RuntimeError) as exc:
            raise DataValidationError("custom preprocessing transformer must be sklearn-cloneable") from exc

    settings = dict(config or {})
    allowed = {
        "numeric_imputer", "numeric_fill_value", "numeric_scaler",
        "categorical_imputer", "categorical_fill_value", "categorical_encoder",
        "numeric_transformer", "categorical_transformer",
    }
    unknown = sorted(set(settings) - allowed)
    if unknown:
        raise DataValidationError(f"unknown preprocessing option(s): {', '.join(unknown)}")
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    transformers = []
    if numeric:
        custom_numeric = settings.get("numeric_transformer")
        if custom_numeric is not None:
            if any(key in settings for key in ("numeric_imputer", "numeric_fill_value", "numeric_scaler")):
                raise DataValidationError(
                    "numeric_transformer cannot be combined with numeric imputer/scaler options"
                )
            try:
                numeric_pipe = clone(custom_numeric)
            except (TypeError, RuntimeError) as exc:
                raise DataValidationError("numeric_transformer must be sklearn-cloneable") from exc
        else:
            imputer = settings.get("numeric_imputer", "median")
            if not isinstance(imputer, str) or imputer not in {
                "mean", "median", "most_frequent", "constant"
            }:
                raise DataValidationError(
                    "numeric_imputer must be 'mean', 'median', 'most_frequent', or 'constant'"
                )
            numeric_steps = [("imputer", SimpleImputer(
                strategy=imputer, fill_value=settings.get("numeric_fill_value", 0)
            ))]
            scaler = settings.get("numeric_scaler", "standard")
            scaler_factories = {
                "standard": StandardScaler,
                "minmax": MinMaxScaler,
                "robust": RobustScaler,
                None: None,
                "none": None,
            }
            if scaler is not None and not isinstance(scaler, str):
                raise DataValidationError(
                    "numeric_scaler must be 'standard', 'minmax', 'robust', or None"
                )
            if scaler not in scaler_factories:
                raise DataValidationError("numeric_scaler must be 'standard', 'minmax', 'robust', or None")
            scaler_factory = scaler_factories[scaler]
            if scaler_factory is not None:
                numeric_steps.append(("scaler", scaler_factory()))
            numeric_pipe = Pipeline(numeric_steps)
        transformers.append(("numeric", numeric_pipe, numeric))
    if categorical:
        custom_categorical = settings.get("categorical_transformer")
        if custom_categorical is not None:
            if any(key in settings for key in (
                "categorical_imputer", "categorical_fill_value", "categorical_encoder"
            )):
                raise DataValidationError(
                    "categorical_transformer cannot be combined with categorical imputer/encoder options"
                )
            try:
                categorical_pipe = clone(custom_categorical)
            except (TypeError, RuntimeError) as exc:
                raise DataValidationError("categorical_transformer must be sklearn-cloneable") from exc
        else:
            imputer = settings.get("categorical_imputer", "most_frequent")
            if not isinstance(imputer, str) or imputer not in {"most_frequent", "constant"}:
                raise DataValidationError("categorical_imputer must be 'most_frequent' or 'constant'")
            encoder = settings.get("categorical_encoder", "onehot")
            if not isinstance(encoder, str):
                raise DataValidationError("categorical_encoder must be 'onehot' or 'ordinal'")
            if encoder == "onehot":
                encoder_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            elif encoder == "ordinal":
                encoder_transformer = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
            else:
                raise DataValidationError("categorical_encoder must be 'onehot' or 'ordinal'")
            categorical_pipe = Pipeline([
                ("imputer", SimpleImputer(
                    strategy=imputer, fill_value=settings.get("categorical_fill_value", "missing")
                )),
                ("encoder", encoder_transformer),
            ])
        transformers.append(("categorical", categorical_pipe, categorical))
    if not transformers:
        raise DataValidationError("no usable feature columns were found")
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def plot_preprocessing(data, kind="overview", target=None, max_features=8, show=False, figsize=None):
    """Plot preprocessing diagnostics and return ``(figure, axes)``.

    ``kind`` supports ``overview``, ``missing_heatmap``, ``heatmap`` (correlation when
    numeric, missingness otherwise), ``correlation``, ``distributions``, ``outliers``,
    and ``target``. Matplotlib is an optional dependency.
    """
    from .preprocessing_visualization import plot_preprocessing as _plot_preprocessing

    return _plot_preprocessing(
        data, kind=kind, target=target, max_features=max_features, show=show, figsize=figsize
    )


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
