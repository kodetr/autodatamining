# Contributing

1. Create a virtual environment and install `python -m pip install -e ".[dev,visualization]"`.
2. Keep task-specific code modular and preserve train/test separation and fold-local preprocessing.
3. Add or update tests for behavior and edge cases.
4. Run `python -m compileall -q src tests`, `python -m ruff check src tests examples`, `python -m pytest -q`, `python -m build`, and `python -m twine check dist/*`.
5. Update README/docs and CHANGELOG when public behavior changes.

Please check third-party dependency licenses and preserve attribution. Do not include user datasets, credentials, or generated model artifacts in pull requests. The current API is alpha and may change before a stable release.
