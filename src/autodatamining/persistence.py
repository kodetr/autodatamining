"""Joblib persistence for fitted model pipelines and minimal experiment metadata."""

import warnings
from pathlib import Path

import joblib
import sklearn


def save_model(model, path, metadata=None):
    """Save a fitted validated model and metadata; do not include training records."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "format_version": 1,
        "model": model,
        "metadata": dict(metadata or {}),
    }
    joblib.dump(bundle, destination, compress=3)
    return destination


def load_model(path):
    """Load a joblib model artifact.

    Only load files from a trusted source: joblib/pickle deserialization can execute arbitrary code.
    """
    warnings.warn(
        "Loading joblib/pickle artifacts can execute code. Only load files from a trusted source.",
        UserWarning,
        stacklevel=2,
    )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("format_version") != 1 or "model" not in bundle:
        raise ValueError("unsupported or invalid AutoDataMining model artifact")
    saved = bundle.get("metadata", {}).get("sklearn_version")
    current = sklearn.__version__
    if saved and saved.split(".")[:2] != current.split(".")[:2]:
        warnings.warn(
            f"Artifact was saved with scikit-learn {saved}, current version is {current}; "
            "cross-version compatibility is not guaranteed.",
            UserWarning,
            stacklevel=2,
        )
    return bundle["model"]
