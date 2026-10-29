"""Matplotlib diagnostics for tabular preprocessing."""

import math

import numpy as np
import pandas as pd

from .detection import as_frame
from .exceptions import DataValidationError

_KINDS = {"overview", "missing_heatmap", "heatmap", "correlation", "distributions", "outliers", "target"}


def plot_preprocessing(data, kind="overview", target=None, max_features=8, show=False, figsize=None):
    """Create a diagnostic figure before fitting and return ``(figure, axes)``."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError as exc:
        raise ImportError("preprocessing plots require `pip install autodatamining[visualization]`") from exc
    if kind not in _KINDS:
        raise DataValidationError(f"kind must be one of: {', '.join(sorted(_KINDS))}")
    try:
        max_features = int(max_features)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("max_features must be a positive integer") from exc
    if max_features < 1:
        raise DataValidationError("max_features must be at least 1")
    frame = as_frame(data)
    if frame.empty:
        raise DataValidationError("preprocessing plots require at least one data row")
    if target is not None and target not in frame.columns:
        raise DataValidationError(f"target column {target!r} was not found")
    target_series = frame[target] if target is not None else None
    features = frame.drop(columns=[target]) if target is not None else frame
    numeric = features.select_dtypes(include=[np.number])
    categorical = features.select_dtypes(exclude=[np.number])

    def message(ax, text):
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    def missing_heatmap(ax, subset):
        subset = subset.iloc[:, :max_features]
        if subset.shape[1] == 0:
            message(ax, "No feature columns to inspect")
            return
        sample_count = min(len(subset), 1000)
        sample_indices = np.linspace(0, len(subset) - 1, sample_count, dtype=int)
        values = subset.iloc[sample_indices].isna().to_numpy().T
        image = ax.imshow(
            values,
            aspect="auto",
            interpolation="nearest",
            cmap=ListedColormap(["#eef2f7", "#d1495b"]),
            vmin=0,
            vmax=1,
        )
        ax.set_yticks(np.arange(subset.shape[1]), labels=[str(c) for c in subset.columns])
        ax.set_xlabel("Sample rows (up to 1,000 sampled)")
        ax.set_title("Missing-value heatmap")
        colorbar = ax.figure.colorbar(image, ax=ax, ticks=[0, 1])
        colorbar.set_label("Missingness")
        colorbar.set_ticklabels(["present", "missing"])

    def correlation(ax, subset):
        values = subset.iloc[:, :max_features]
        if values.shape[1] == 0:
            message(ax, "No numeric features for correlation")
            return
        matrix = values.corr()
        image = ax.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        names = [str(c) for c in matrix.columns]
        ax.set_xticks(np.arange(len(names)), labels=names, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(names)), labels=names)
        if len(names) <= 12:
            for row in range(len(names)):
                for column in range(len(names)):
                    value = matrix.iat[row, column]
                    label = "nan" if pd.isna(value) else f"{value:.2f}"
                    ax.text(column, row, label, ha="center", va="center", fontsize=8)
        ax.set_title("Numeric correlation heatmap")
        ax.figure.colorbar(image, ax=ax, label="Correlation")

    def distribution(ax, name, series):
        if pd.api.types.is_numeric_dtype(series.dtype):
            series.dropna().hist(ax=ax, bins="auto", color="#2878b5", edgecolor="white")
            ax.set_ylabel("Count")
        else:
            counts = series.astype("object").fillna("<missing>").value_counts().head(10)
            ax.bar([str(value) for value in counts.index], counts.values, color="#2878b5")
            ax.tick_params(axis="x", labelrotation=35)
            ax.set_ylabel("Count")
        ax.set_title(str(name))

    if kind == "overview":
        fig, axes = plt.subplots(2, 2, figsize=figsize or (12, 8))
        axes_map = {
            "missing": axes[0, 0],
            "correlation": axes[0, 1],
            "feature_distribution": axes[1, 0],
            "target": axes[1, 1],
        }
        missing_counts = features.isna().sum().sort_values(ascending=False).head(max_features)
        if missing_counts.empty:
            message(axes_map["missing"], "No feature columns to inspect")
        else:
            axes_map["missing"].bar(
                [str(c) for c in missing_counts.index], missing_counts.values, color="#d1495b"
            )
            axes_map["missing"].tick_params(axis="x", labelrotation=35)
            axes_map["missing"].set_title("Missing values by feature")
            axes_map["missing"].set_ylabel("Missing rows")
        correlation(axes_map["correlation"], numeric)
        if not numeric.empty:
            distribution(axes_map["feature_distribution"], numeric.columns[0], numeric.iloc[:, 0])
        elif not categorical.empty:
            distribution(axes_map["feature_distribution"], categorical.columns[0], categorical.iloc[:, 0])
        else:
            message(axes_map["feature_distribution"], "No features to inspect")
        if target_series is not None:
            distribution(axes_map["target"], f"Target: {target}", target_series)
        elif not categorical.empty:
            distribution(axes_map["target"], categorical.columns[0], categorical.iloc[:, 0])
        else:
            message(axes_map["target"], "Pass target=... to plot target distribution")
        fig.suptitle(f"Preprocessing overview — {len(frame):,} rows, {features.shape[1]} features")
        fig.tight_layout()
        axes = axes_map
    elif kind in {"missing_heatmap", "heatmap"}:
        if kind == "heatmap" and not numeric.empty:
            fig, ax = plt.subplots(figsize=figsize or (8, 6))
            correlation(ax, numeric)
        else:
            fig, ax = plt.subplots(figsize=figsize or (10, 5))
            missing_heatmap(ax, features)
        axes = ax
    elif kind == "correlation":
        if numeric.empty:
            raise DataValidationError("correlation heatmaps require at least one numeric feature")
        fig, ax = plt.subplots(figsize=figsize or (8, 6))
        correlation(ax, numeric)
        axes = ax
    elif kind == "distributions":
        columns = list(numeric.columns) + list(categorical.columns)
        columns = columns[:max_features]
        if not columns:
            raise DataValidationError("no features are available to plot")
        ncols = min(2, len(columns))
        nrows = math.ceil(len(columns) / ncols)
        fig, axes_grid = plt.subplots(nrows, ncols, figsize=figsize or (12, 4 * nrows), squeeze=False)
        for ax, name in zip(axes_grid.flat, columns):
            distribution(ax, name, features[name])
        for ax in list(axes_grid.flat)[len(columns):]:
            ax.set_visible(False)
        fig.suptitle("Feature distributions")
        fig.tight_layout()
        axes = axes_grid
    elif kind == "outliers":
        columns = list(numeric.columns[:max_features])
        if not columns:
            raise DataValidationError("outlier boxplots require numeric features")
        fig, ax = plt.subplots(figsize=figsize or (max(8, len(columns) * 1.3), 5))
        ax.boxplot([numeric[column].dropna().to_numpy() for column in columns])
        ax.set_xticks(np.arange(1, len(columns) + 1), labels=[str(c) for c in columns])
        ax.tick_params(axis="x", labelrotation=35)
        ax.set_title("Numeric feature ranges and potential outliers")
        ax.set_ylabel("Value")
        axes = ax
    else:  # target
        if target_series is None:
            raise DataValidationError("kind='target' requires target=column_name")
        fig, ax = plt.subplots(figsize=figsize or (8, 5))
        distribution(ax, f"Target: {target}", target_series)
        axes = ax

    if show:
        plt.show()
    return fig, axes
