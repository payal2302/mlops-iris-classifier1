# src/compare_tuning_results.py

"""
Compare Baseline, Grid Search and Random Search results
from the MLflow experiment.
"""

import mlflow
import pandas as pd


mlflow.set_tracking_uri("sqlite:///mlflow.db")

experiment = mlflow.get_experiment_by_name(
    "iris-hyperparameter-tuning"
)

if experiment is None:
    print("Experiment not found.")
    exit()

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id]
)

columns = [
    "tags.mlflow.runName",
    "metrics.cv_f1_macro_mean",
    "metrics.best_cv_f1_macro",
    "metrics.test_accuracy",
]

available_columns = [
    col for col in columns
    if col in runs.columns
]

results = runs[available_columns].copy()

results = results.sort_values(
    by="metrics.test_accuracy",
    ascending=False
)

print("\n===== MLflow Tuning Comparison =====\n")

print(
    results.to_string(index=False)
)

print("\n====================================")