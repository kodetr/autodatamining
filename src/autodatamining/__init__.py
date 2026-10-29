"""AutoDataMining: consistent, transparent workflows over scikit-learn estimators."""

from .core import AutoDataMining
from .detection import detect_task, profile_data
from .exceptions import (
    AutoDataMiningError,
    ConfigurationError,
    DataValidationError,
    ModelCompatibilityError,
    NoModelsSucceededError,
)
from .persistence import load_model
from .preprocessing import reduce_dimensions, select_features
from .registry import EstimatorRegistry, register_estimator
from .results import MiningResults, ValidatedModel
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
    "ValidatedModel",
    "__version__",
    "detect_task",
    "load_model",
    "profile_data",
    "reduce_dimensions",
    "register_estimator",
    "select_features",
]
