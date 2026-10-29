"""Optional matplotlib visualizations for experiment results."""

import numpy as np

from .exceptions import DataValidationError


def plot_results(results, kind="leaderboard", ax=None, show=False, save_path=None):
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
        score_col = next((column for column in ("cv_score", "silhouette", "f1", "anomaly_rate", "itemsets_count", "rules_count")
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
    elif kind == "rules":
        if results.task != "association" or results.rules is None or results.rules.empty:
            raise DataValidationError("rules plots require an association run with non-empty rules")
        scatter = ax.scatter(
            results.rules["support"],
            results.rules["confidence"],
            c=results.rules["lift"],
            cmap="viridis",
            alpha=0.8,
            s=60,
        )
        ax.set_xlabel("Support")
        ax.set_ylabel("Confidence")
        ax.set_title("Association Rules (Support vs. Confidence)")
        ax.figure.colorbar(scatter, ax=ax, label="Lift")
    elif kind == "pipeline":
        from matplotlib.patches import FancyBboxPatch
        ax.axis("off")
        samples_info = f"Sampel: {results.experiment.get('train_samples', '-')}"
        cv_info = str(results.experiment.get("cv_strategy", "5-Fold CV")).replace("_", " ").title()
        model_display = str(results.model_name or "Estimator").replace("_", " ").title()
        steps = [
            ("1. Data Input", f"{results.task.title()}\n{samples_info}", "#e3f2fd", "#1565c0"),
            ("2. Prapemrosesan", "Imputasi Median/Modus\nScaling & Encoding", "#e8f5e9", "#2e7d32"),
            ("3. Validasi Silang", f"{cv_info}\nEvaluasi Metodologis", "#fff3e0", "#e65100"),
            ("4. Model Terbaik", f"{model_display}\nPeringkat #1 Leaderboard", "#f3e5f5", "#7b1fa2"),
            ("5. Output Prediksi", "Metrik Evaluasi Uji\nLaporan & Kesimpulan", "#fce4ec", "#c2185b"),
        ]
        n_steps = len(steps)
        box_width = 0.16
        box_height = 0.58
        spacing = 0.20
        start_x = 0.02
        y = 0.20

        for i, (title, desc, fill_color, border_color) in enumerate(steps):
            x = start_x + i * spacing
            box = FancyBboxPatch(
                (x, y), box_width, box_height,
                boxstyle="round,pad=0.015,rounding_size=0.03",
                facecolor=fill_color,
                edgecolor=border_color,
                linewidth=1.8,
                transform=ax.transAxes,
            )
            ax.add_patch(box)
            ax.text(
                x + box_width / 2, y + box_height - 0.12, title,
                ha="center", va="center", weight="bold", fontsize=9.5, color=border_color,
                transform=ax.transAxes,
            )
            ax.text(
                x + box_width / 2, y + box_height / 2 - 0.08, desc,
                ha="center", va="center", fontsize=8, color="#2d3748",
                transform=ax.transAxes,
            )
            if i < n_steps - 1:
                arrow_start_x = x + box_width + 0.005
                arrow_end_x = start_x + (i + 1) * spacing - 0.005
                arrow_y = y + box_height / 2
                ax.annotate(
                    "",
                    xy=(arrow_end_x, arrow_y),
                    xytext=(arrow_start_x, arrow_y),
                    xycoords="axes fraction",
                    textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=2.0, color="#4a5568"),
                )
        ax.set_title(f"Diagram Alur Pipeline AutoDataMining ({results.task.title()})", fontsize=11.5, pad=12, weight="bold")
    else:
        raise DataValidationError(
            "kind must be 'leaderboard', 'confusion_matrix', 'residuals', 'roc_curve', "
            "'precision_recall_curve', 'clusters', 'anomalies', 'rules', or 'pipeline'"
        )
    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return ax
