# Custom estimators

Register a cloneable scikit-learn estimator and declare compatible task names:

```python
from sklearn.linear_model import SGDClassifier
from autodatamining import AutoDataMining, EstimatorRegistry

registry = EstimatorRegistry()
registry.register("sgd", SGDClassifier, tasks="classification", params={"max_iter": 2000})
results = AutoDataMining(task="classification", target="label", registry=registry,
                         models=["sgd"]).fit(data)
```

Names must be unique within a registry. Estimators are cloned before fitting; task compatibility and parameters are checked. Per-model failures appear in the leaderboard. Use `model_params={"sgd": {"alpha": 0.001}}` to override parameters for an experiment.
