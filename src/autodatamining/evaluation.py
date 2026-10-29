"""Task-specific metric computation and score direction handling."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    get_scorer,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)


def classification_metrics(y_true, y_pred, y_score=None, labels=None):
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_score is not None:
        try:
            if np.asarray(y_score).ndim == 1 or np.asarray(y_score).shape[1] == 2:
                pos = np.asarray(y_score) if np.asarray(y_score).ndim == 1 else np.asarray(y_score)[:, -1]
                result["roc_auc"] = float(roc_auc_score(y_true, pos))
                result["pr_auc"] = float(average_precision_score(y_true, pos))
            else:
                result["roc_auc"] = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted"))
        except (ValueError, IndexError):
            pass
    return result


def regression_metrics(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def clustering_metrics(X_transformed, labels):
    labels = np.asarray(labels)
    # DBSCAN's -1 noise label is not a meaningful cluster for internal metrics.
    mask = labels != -1
    score_labels = labels[mask]
    score_X = np.asarray(X_transformed)[mask]
    if len(set(score_labels.tolist())) < 2 or len(set(score_labels.tolist())) >= len(score_labels):
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"), "calinski_harabasz": float("nan")}
    return {
        "silhouette": float(silhouette_score(score_X, score_labels)),
        "davies_bouldin": float(davies_bouldin_score(score_X, score_labels)),
        "calinski_harabasz": float(calinski_harabasz_score(score_X, score_labels)),
    }


def anomaly_metrics(labels, y_true=None, anomaly_scores=None):
    detected = np.asarray(labels) == -1
    result = {"anomaly_rate": float(np.mean(detected)), "anomaly_count": int(detected.sum())}
    if y_true is None:
        return result
    truth = np.asarray(y_true)
    def is_anomaly(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return value == 1
        return str(value).strip().lower() in {"1", "true", "anomaly", "outlier", "yes"}
    positive = np.asarray([is_anomaly(value) for value in truth], dtype=bool)
    result.update({
        "precision": float(precision_score(positive, detected, zero_division=0)),
        "recall": float(recall_score(positive, detected, zero_division=0)),
        "f1": float(f1_score(positive, detected, zero_division=0)),
    })
    if anomaly_scores is not None and len(np.unique(positive)) == 2:
        try:
            result["roc_auc"] = float(roc_auc_score(positive, anomaly_scores))
            result["pr_auc"] = float(average_precision_score(positive, anomaly_scores))
        except ValueError:
            pass
    return result


def scorer_direction(scoring):
    """Return +1 if larger scores are better and -1 if smaller values are better."""
    if not scoring:
        return 1
    return int(get_scorer(scoring)._sign)
