"""Exceptions raised by AutoDataMining."""


class AutoDataMiningError(Exception):
    """Base exception for library errors."""


class ConfigurationError(AutoDataMiningError, ValueError):
    """An invalid task or configuration was supplied."""


class DataValidationError(AutoDataMiningError, ValueError):
    """Input data cannot be used for the requested operation."""


class NoModelsSucceededError(AutoDataMiningError, RuntimeError):
    """All requested estimators failed."""


class ModelCompatibilityError(AutoDataMiningError, TypeError):
    """A model is incompatible with the requested task."""
