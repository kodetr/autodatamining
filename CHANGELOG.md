# Changelog

## 0.2.0 — 2026-10-30

- Added interactive No-Code Web Dashboard GUI (`app.py` and `python -m autodatamining.ui`) with complete Streamlit interface ready for Hugging Face Spaces.
- Added Hugging Face Spaces deployment metadata and documentation (`spaces/huggingface/` and `docs/huggingface-spaces.md`).
- Added 5 structured practical Jupyter Notebooks in `notebooks/` ready for Google Colab.
- Added automated hyperparameter tuning (`results.tune()`) using RandomizedSearchCV and GridSearchCV without data leakage.
- Added visual pipeline flowchart diagram (`results.plot(kind="pipeline")`) for theses and slide presentations.
- Added built-in educational datasets (`load_dataset`, `list_datasets`) covering all 5 data mining tasks with zero-friction offline availability.
- Added academic table and publication-ready LaTeX exporters (`results.to_academic_table()`, `results.to_latex()`) formatted for theses and journal papers.
- Added interactive concept glossary and explainer (`explain_concept()`, `list_concepts()`) providing plain-language definitions and everyday analogies.
- Added automated narrative conclusions via `results.summary()` and `results.kesimpulan()` featuring overfitting diagnosis, plain-language metric interpretation, and draft academic discussion paragraphs.
- Embedded narrative conclusions directly inside HTML experiment reports.
- Added native Association Rule Mining module with Apriori, FP-Growth, and rule metrics (support, confidence, lift, leverage, conviction).
- Added `task="association"` in AutoDataMining with automated algorithm benchmarking and rule scatter plot visualization (`plot(kind="rules")`).
- Added `TransactionEncoder` for transforming raw transaction sequences into boolean DataFrames.
- Added built-in voting and stacking ensembles for classification and regression.
- Propagated the experiment random seed to nested ensemble estimators.

## 0.1.1 — 2026-10-30

- Added configurable numeric and categorical preprocessing strategies and custom scikit-learn transformers.
- Added preprocessing diagnostics for missing values, correlations, feature distributions, outliers, and targets.
- Recorded the selected preprocessing configuration in experiment metadata.
- Added an Indonesian beginner guide with examples for all supported task types.

## 0.1.0

- Initial modular library for classification, regression, clustering, and anomaly detection.
- Added registry, mixed tabular preprocessing, evaluation, reports, explainability, and persistence.
- Initial alpha release.
