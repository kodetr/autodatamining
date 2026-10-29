"""Built-in estimator catalog and public custom estimator registry."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from sklearn.base import clone
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, LocalOutlierFactor
from sklearn.svm import SVC, SVR, OneClassSVM
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .exceptions import ConfigurationError, ModelCompatibilityError

_VALID_TASKS = {"classification", "regression", "clustering", "anomaly"}


@dataclass
class EstimatorSpec:
    name: str
    estimator: Any
    tasks: tuple[str, ...]
    default_params: dict[str, Any] = field(default_factory=dict)
    built_in: bool = False

    def instantiate(self, random_state=None, params=None):
        base = self.estimator() if isinstance(self.estimator, type) else clone(self.estimator)
        config = deepcopy(self.default_params)
        if random_state is not None:
            shallow_params = base.get_params(deep=False)
            if "random_state" in shallow_params:
                config["random_state"] = random_state
            for name, value in base.get_params(deep=True).items():
                if name.endswith("__random_state") and value is None:
                    config[name] = random_state
        if params:
            config.update(params)
        try:
            base.set_params(**config)
            return clone(base)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"invalid parameters for estimator {self.name!r}: {exc}") from exc


class EstimatorRegistry:
    """Task-aware estimator registry. Registrations are copied per registry instance."""

    def __init__(self, include_defaults=True):
        self._specs = {}
        if include_defaults:
            self._add_builtins()

    def register(self, name, estimator, tasks, params=None, replace=False):
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError("estimator name must be a non-empty string")
        key = name.strip().lower()
        existing = [index for index in self._specs if index[0] == key]
        if existing and not replace:
            raise ConfigurationError(f"estimator {key!r} is already registered")
        if isinstance(tasks, str):
            tasks = (tasks,)
        tasks = tuple(dict.fromkeys(tasks))
        if not tasks or any(task not in _VALID_TASKS for task in tasks):
            raise ConfigurationError("tasks must contain valid task names")
        try:
            instance = estimator() if isinstance(estimator, type) else clone(estimator)
        except Exception as exc:
            raise ModelCompatibilityError("estimator must be a scikit-learn compatible estimator") from exc
        if not hasattr(instance, "fit") or not hasattr(instance, "get_params"):
            raise ModelCompatibilityError("estimator must implement fit() and get_params()")
        if params:
            try:
                instance.set_params(**params)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(f"invalid default estimator parameters: {exc}") from exc
        if existing:
            for index in existing:
                del self._specs[index]
        spec = EstimatorSpec(key, estimator, tasks, dict(params or {}), False)
        for task in tasks:
            self._specs[(key, task)] = spec
        return self

    def get(self, name, task):
        key = str(name).strip().lower()
        index = (key, task)
        if index not in self._specs:
            if any(existing_name == key for existing_name, _ in self._specs):
                raise ModelCompatibilityError(f"estimator {key!r} is not registered for task {task!r}")
            raise ConfigurationError(f"unknown estimator {name!r}; available: {', '.join(self.names(task))}")
        return self._specs[index]

    def names(self, task=None):
        return [name for (name, registered_task), _ in self._specs.items()
                if task is None or task == registered_task]

    def _add(self, name, cls, task, params=None):
        self._specs[(name, task)] = EstimatorSpec(name, cls, (task,), params or {}, True)

    def _add_builtins(self):
        classifiers = {
            "logistic_regression": (LogisticRegression, {"max_iter": 1000}),
            "decision_tree": (DecisionTreeClassifier, {}),
            "random_forest": (RandomForestClassifier, {"n_estimators": 100, "n_jobs": -1}),
            "extra_trees": (ExtraTreesClassifier, {"n_estimators": 100, "n_jobs": -1}),
            "gradient_boosting": (GradientBoostingClassifier, {}),
            "hist_gradient_boosting": (HistGradientBoostingClassifier, {}),
            "knn": (KNeighborsClassifier, {"n_neighbors": 5}),
            "gaussian_nb": (GaussianNB, {}),
            "svm": (SVC, {"probability": True}),
            "voting": (
                VotingClassifier(
                    estimators=[
                        ("logistic", LogisticRegression(max_iter=1000)),
                        ("random_forest", RandomForestClassifier(n_estimators=100, n_jobs=-1)),
                        ("extra_trees", ExtraTreesClassifier(n_estimators=100, n_jobs=-1)),
                    ],
                    voting="soft",
                ),
                {},
            ),
            "stacking": (
                StackingClassifier(
                    estimators=[
                        ("logistic", LogisticRegression(max_iter=1000)),
                        ("random_forest", RandomForestClassifier(n_estimators=100, n_jobs=-1)),
                        ("extra_trees", ExtraTreesClassifier(n_estimators=100, n_jobs=-1)),
                    ],
                    final_estimator=LogisticRegression(max_iter=1000),
                    cv=3,
                ),
                {},
            ),
        }
        regressors = {
            "linear_regression": (LinearRegression, {}),
            "ridge": (Ridge, {}),
            "lasso": (Lasso, {"max_iter": 5000}),
            "decision_tree": (DecisionTreeRegressor, {}),
            "random_forest": (RandomForestRegressor, {"n_estimators": 100, "n_jobs": -1}),
            "extra_trees": (ExtraTreesRegressor, {"n_estimators": 100, "n_jobs": -1}),
            "gradient_boosting": (GradientBoostingRegressor, {}),
            "hist_gradient_boosting": (HistGradientBoostingRegressor, {}),
            "knn": (KNeighborsRegressor, {"n_neighbors": 5}),
            "svm": (SVR, {}),
            "voting": (
                VotingRegressor(
                    estimators=[
                        ("ridge", Ridge()),
                        ("random_forest", RandomForestRegressor(n_estimators=100, n_jobs=-1)),
                        ("extra_trees", ExtraTreesRegressor(n_estimators=100, n_jobs=-1)),
                    ]
                ),
                {},
            ),
            "stacking": (
                StackingRegressor(
                    estimators=[
                        ("ridge", Ridge()),
                        ("random_forest", RandomForestRegressor(n_estimators=100, n_jobs=-1)),
                        ("extra_trees", ExtraTreesRegressor(n_estimators=100, n_jobs=-1)),
                    ],
                    final_estimator=Ridge(),
                    cv=3,
                ),
                {},
            ),
        }
        for name, (cls, params) in classifiers.items():
            self._add(name, cls, "classification", params)
        for name, (cls, params) in regressors.items():
            self._add(name, cls, "regression", params)
        for name, cls, params in [
            ("kmeans", KMeans, {"n_clusters": 3, "n_init": 10}),
            ("minibatch_kmeans", MiniBatchKMeans, {"n_clusters": 3, "n_init": 10}),
            ("dbscan", DBSCAN, {"eps": 0.5, "min_samples": 5}),
            ("agglomerative", AgglomerativeClustering, {"n_clusters": 3}),
            ("gaussian_mixture", GaussianMixture, {"n_components": 3}),
        ]:
            self._add(name, cls, "clustering", params)
        for name, cls, params in [
            ("isolation_forest", IsolationForest, {"n_estimators": 100, "contamination": "auto"}),
            ("lof", LocalOutlierFactor, {"n_neighbors": 20, "novelty": True}),
            ("one_class_svm", OneClassSVM, {"nu": 0.05}),
        ]:
            self._add(name, cls, "anomaly", params)


DEFAULT_REGISTRY = EstimatorRegistry()


def register_estimator(name, estimator, tasks, params=None, replace=False):
    """Register an estimator globally for subsequent AutoDataMining instances."""
    DEFAULT_REGISTRY.register(name, estimator, tasks, params=params, replace=replace)
    return DEFAULT_REGISTRY
