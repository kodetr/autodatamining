import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from autodatamining import AutoDataMining, plot_preprocessing, reduce_dimensions, select_features


def test_profile_feature_selection_and_reduction():
    X = pd.DataFrame({"a": np.arange(12), "b": np.arange(12) ** 2, "constant": 1})
    y = np.array([0, 1] * 6)
    profile = AutoDataMining.profile(X)
    assert profile["rows"] == 12
    selected, _selector = select_features(X, method="variance")
    assert selected.shape[1] == 2
    reduced, _transformer = reduce_dimensions(X, n_components=2)
    assert reduced.shape == (12, 2)
    selected, _selector = select_features(X.drop(columns="constant"), y, method="kbest", k=1)
    assert selected.shape == (12, 1)
    svd_embedding, _svd = reduce_dimensions(X.drop(columns="constant"), method="svd", n_components=1)
    assert svd_embedding.shape == (12, 1)


def test_configurable_preprocessing_strategies_are_applied_and_recorded():
    data = pd.DataFrame({
        "amount": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                   11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
        "kind": ["a", "b"] * 10,
        "label": [0, 1] * 10,
    })
    results = AutoDataMining(
        task="classification", target="label", models=["logistic_regression"], cv=2,
        preprocessing={"numeric_scaler": "robust", "categorical_encoder": "ordinal"},
    ).fit(data)

    preprocessor = results.best_model.estimator.named_steps["preprocessor"]
    assert type(preprocessor.named_transformers_["numeric"].named_steps["scaler"]).__name__ == "RobustScaler"
    assert type(preprocessor.named_transformers_["categorical"].named_steps["encoder"]).__name__ == "OrdinalEncoder"
    assert results.experiment["preprocessing"] == {
        "mode": "strategies",
        "options": {"numeric_scaler": "robust", "categorical_encoder": "ordinal"},
    }


def test_custom_sklearn_transformer_can_replace_default_preprocessing():
    data = pd.DataFrame({"first": np.arange(20.0), "second": np.arange(20.0) ** 2,
                         "label": [0, 1] * 10})
    results = AutoDataMining(
        task="classification", target="label", models=["logistic_regression"], cv=2,
        preprocessing=StandardScaler(),
    ).fit(data)
    assert type(results.best_model.estimator.named_steps["preprocessor"]).__name__ == "StandardScaler"
    assert results.experiment["preprocessing"]["mode"] == "custom_transformer"

    mixed = pd.DataFrame({
        "amount": np.arange(20.0),
        "kind": ["a", "b"] * 10,
        "label": [0, 1] * 10,
    })
    branch_results = AutoDataMining(
        task="classification", target="label", models=["logistic_regression"], cv=2,
        preprocessing={"numeric_transformer": StandardScaler()},
    ).fit(mixed)
    assert type(
        branch_results.best_model.estimator.named_steps["preprocessor"]
        .named_transformers_["numeric"]
    ).__name__ == "StandardScaler"


def test_preprocessing_diagnostic_plots_return_figures_and_axes():
    data = pd.DataFrame({
        "amount": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        "other": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        "kind": ["a", "b", "a", "b", "a", "b"],
        "label": [0, 1, 0, 1, 0, 1],
    })
    try:
        fig, axes = AutoDataMining.plot_preprocessing(data, target="label")
        assert set(axes) == {"missing", "correlation", "feature_distribution", "target"}
        plt.close(fig)

        fig, ax = plot_preprocessing(data, kind="missing_heatmap", target="label")
        assert ax.get_title() == "Missing-value heatmap"
        plt.close(fig)

        fig, ax = plot_preprocessing(data, kind="heatmap", target="label")
        assert ax.get_title() == "Numeric correlation heatmap"
        plt.close(fig)

        fig, axes = plot_preprocessing(data, kind="distributions", max_features=3)
        assert sum(ax.get_visible() for ax in axes.flat) == 3
        plt.close(fig)

        fig, ax = plot_preprocessing(data, kind="outliers", target="label")
        assert ax.get_title() == "Numeric feature ranges and potential outliers"
        plt.close(fig)

        fig, ax = plot_preprocessing(data, kind="target", target="label")
        assert ax.get_title() == "Target: label"
        plt.close(fig)
    finally:
        plt.close("all")
