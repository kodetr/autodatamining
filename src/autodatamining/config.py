"""Validated user-facing configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .exceptions import ConfigurationError


@dataclass(frozen=True)
class MiningConfig:
    task: str = "auto"
    target: Optional[str] = None  # noqa: UP045 - keep Python 3.9 syntax support
    test_size: float = 0.2
    cv: int = 5
    random_state: int = 42
    scoring: Optional[str] = None  # noqa: UP045 - keep Python 3.9 syntax support
    min_support: float = 0.1
    min_confidence: float = 0.5
    max_len: Optional[int] = None  # noqa: UP045 - keep Python 3.9 syntax support
    metric: str = "confidence"

    def __post_init__(self):
        aliases = {
            "anomaly_detection": "anomaly",
            "outlier_detection": "anomaly",
            "association_rules": "association",
            "association_rule_mining": "association",
        }
        object.__setattr__(self, "task", aliases.get(self.task, self.task))
        allowed = {"auto", "classification", "regression", "clustering", "anomaly", "association"}
        if self.task not in allowed:
            raise ConfigurationError("task must be one of: " + ", ".join(sorted(allowed)))
        if not 0 < self.test_size < 1:
            raise ConfigurationError("test_size must be greater than 0 and less than 1")
        if self.cv < 2:
            raise ConfigurationError("cv must be at least 2")
        if self.task in {"classification", "regression"} and not self.target:
            raise ConfigurationError(f"target is required for {self.task}")
        if not 0 < self.min_support <= 1.0:
            raise ConfigurationError("min_support must be between 0 (exclusive) and 1.0 (inclusive)")
        if not 0 < self.min_confidence <= 1.0:
            raise ConfigurationError("min_confidence must be between 0 (exclusive) and 1.0 (inclusive)")
