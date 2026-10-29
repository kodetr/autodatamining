# Installation

Install the alpha package from PyPI after it is published:

```bash
python -m pip install autodatamining
```

For local development:

```bash
python -m pip install -e ".[dev,visualization]"
```

Optional dependencies: `visualization` installs matplotlib; `explainability` installs SHAP. Core install uses NumPy, pandas, and scikit-learn only.
