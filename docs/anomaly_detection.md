# Anomaly detection

Use `AutoDataMining(task="anomaly")`. Labels returned by supported scikit-learn estimators use `-1` for anomaly and `+1` for inlier. Without ground truth, the report includes detection count/rate only and does not claim a quality ranking.

When ground truth is available, pass `target="is_anomaly"`; encode anomalies as 1/True and normal rows as 0/False. The target is excluded from the features. A stratified holdout reports precision, recall, F1, and AUC where defined; training-fold F1 selects the estimator so holdout scores are not used for selection. In labeled mode `results.labels` contains holdout predictions. Without labels, it contains fit-set detections.
