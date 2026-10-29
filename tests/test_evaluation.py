import numpy as np

from autodatamining.evaluation import (
    anomaly_metrics,
    classification_metrics,
    regression_metrics,
    scorer_direction,
)


def test_metric_direction_and_task_metrics():
    assert scorer_direction("accuracy") == 1
    assert scorer_direction("neg_mean_squared_error") == -1
    classification = classification_metrics([0, 1, 1], [0, 1, 0], [0.1, 0.9, 0.4])
    assert classification["accuracy"] == 2 / 3
    regression = regression_metrics([1, 2], [1, 3])
    assert regression["rmse"] > 0
    anomaly = anomaly_metrics([-1, 1, -1], [1, 0, 1], np.array([0.9, 0.1, 0.8]))
    assert anomaly["f1"] == 1.0
