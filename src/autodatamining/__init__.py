"""AutoDataMining: consistent, transparent workflows over scikit-learn estimators."""

from .association import TransactionEncoder, apriori, association_rules, fpgrowth
from .core import AutoDataMining
from .datasets import list_datasets, load_dataset
from .detection import detect_task, profile_data
from .exceptions import (
    AutoDataMiningError,
    ConfigurationError,
    DataValidationError,
    ModelCompatibilityError,
    NoModelsSucceededError,
)
from .glossary import explain_concept, list_concepts
from .persistence import load_model
from .preprocessing import plot_preprocessing, reduce_dimensions, select_features
from .registry import EstimatorRegistry, register_estimator
from .results import MiningResults, ValidatedModel
from .ui import launch_ui
from .version import __version__

__all__ = [
    "AutoDataMining",
    "AutoDataMiningError",
    "ConfigurationError",
    "DataValidationError",
    "EstimatorRegistry",
    "MiningResults",
    "ModelCompatibilityError",
    "NoModelsSucceededError",
    "TransactionEncoder",
    "ValidatedModel",
    "__version__",
    "apriori",
    "association_rules",
    "detect_task",
    "explain_concept",
    "fpgrowth",
    "launch_ui",
    "list_concepts",
    "list_datasets",
    "load_dataset",
    "load_model",
    "plot_preprocessing",
    "profile_data",
    "reduce_dimensions",
    "register_estimator",
    "select_features",
]
