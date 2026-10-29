"""Optional matplotlib visualizations for experiment results."""

import numpy as np

from .exceptions import DataValidationError


def plot_results(results, kind="leaderboard", ax=None, show=False):
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.metrics import (
            ConfusionMatrixDisplay,
            precision_recall_curve,
            roc_curve,
        )
    except ImportError as exc:
        raise ImportError("plotting requires `pip install autodatamining[visualization]`") from exc
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    if kind == "leaderboard":
        board = results.leaderboard
        score_col = next((column for column in ("cv_score", "silhouette", "f1", "anomaly_rate")
                          if column in board and board[column].notna().any()), None)
        if score_col is None:
            raise DataValidationError("leaderboard has no finite numeric scores to plot")
        visible = board[board[score_col].notna()]
        ax.barh(visible["model"], visible[score_col])
        label = "anomaly_rate (not a quality score)" if results.task == "anomaly" and score_col == "anomaly_rate" else score_col
        ax.set_xlabel(label)
        title = "Detected anomaly rate (not model quality)" if results.task == "anomaly" and score_col == "anomaly_rate" else f"{results.task.title()} model comparison"
        ax.set_title(title)
        ax.invert_yaxis()
    elif kind == "confusion_matrix":
        if results.task != "classification" or results._y_test is None:
            raise DataValidationError("confusion_matrix plots require a supervised classification run")
        prediction = results.predict(results._X_test)
        ConfusionMatrixDisplay.from_predictions(results._y_test, prediction, ax=ax, colorbar=False)
    elif kind == "residuals":
        if results.task != "regression" or results._y_test is None:
            raise DataValidationError("residual plots require a supervised regression run")
        prediction = results.predict(results._X_test)
        residuals = np.asarray(results._y_test) - np.asarray(prediction)
        ax.scatter(prediction, residuals, alpha=0.7)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel("Predicted value")
        ax.set_ylabel("Residual")
        ax.set_title("Holdout residuals")
    elif kind in {"roc_curve", "precision_recall_curve"}:
        if results.task != "classification" or results._y_test is None:
            raise DataValidationError(f"{kind} plots require a supervised classification run")
        classes = list(results.best_model.classes_)
        if len(classes) != 2:
            raise DataValidationError(f"{kind} currently requires a binary classification task")
        if hasattr(results.best_model, "predict_proba"):
            scores = results.best_model.predict_proba(results._X_test)[:, 1]
        elif hasattr(results.best_model, "decision_function"):
            scores = results.best_model.decision_function(results._X_test)
        else:
            raise DataValidationError("the selected estimator has no score method for curve plotting")
        positive = np.asarray(results._y_test) == classes[1]
        if kind == "roc_curve":
            fpr, tpr, _ = roc_curve(positive, scores)
            ax.plot(fpr, tpr, label="ROC")
            ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("Holdout ROC curve")
        else:
            precision, recall, _ = precision_recall_curve(positive, scores)
            ax.plot(recall, precision, label="Precision–recall")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Holdout precision–recall curve")
    elif kind in {"clusters", "anomalies"}:
        if results.task not in {"clustering", "anomaly"} or results.labels is None:
            raise DataValidationError(f"{kind} plots require a clustering or anomaly run")
        pipeline = results.best_model.estimator
        transformed = pipeline.named_steps["preprocessor"].transform(results._X_test)
        values = np.asarray(transformed)
        if values.ndim != 2:
            raise DataValidationError("could not create a 2D view of the transformed features")
        if values.shape[1] >= 2:
            coords = PCA(n_components=2, random_state=results.experiment.get("random_state", 42)).fit_transform(values)
        else:
            coords = np.column_stack([values[:, 0], np.zeros(len(values))])
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=results.labels, cmap="tab10", alpha=0.8)
        ax.set_xlabel("Projection 1")
        ax.set_ylabel("Projection 2")
        ax.set_title("Cluster projection" if kind == "clusters" else "Anomaly labels (-1 = anomaly)")
        if kind == "clusters":
            ax.figure.colorbar(scatter, ax=ax, label="Cluster label")
    else:
        raise DataValidationError(
            "kind must be 'leaderboard', 'confusion_matrix', 'residuals', 'roc_curve', "
            "'precision_recall_curve', 'clusters', or 'anomalies'"
        )
    if show:
        plt.show()
    return ax
