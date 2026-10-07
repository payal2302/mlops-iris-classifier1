# src/random_search_tuning.py

"""
Random Search over the same hyperparameter space as Grid Search.
"""

import mlflow
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder


FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]


PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(50, 201),
    "max_depth": [3, 5, 10, None],
    "min_samples_split": randint(2, 11),
    "max_features": ["sqrt", "log2"],
}


def load_dataset(path: str):

    df = pd.read_csv(path)

    le = LabelEncoder()
    y = le.fit_transform(df["species"])

    X = df[FEATURE_COLS].fillna(
        df[FEATURE_COLS].median()
    )

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def run_random_search(data_path: str):

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment(
        "iris-hyperparameter-tuning"
    )

    X_train, X_test, y_train, y_test = load_dataset(
        data_path
    )

    with mlflow.start_run(
        run_name="random_search_random_forest"
    ):

        n_iter = 72

        mlflow.log_param(
            "search_type",
            "RandomizedSearchCV"
        )

        mlflow.log_param(
            "n_iter",
            n_iter
        )

        mlflow.log_param(
            "cv_folds",
            5
        )

        random_search = RandomizedSearchCV(
            RandomForestClassifier(
                random_state=42
            ),
            param_distributions=PARAM_DISTRIBUTIONS,
            n_iter=n_iter,
            cv=5,
            scoring="f1_macro",
            random_state=42,
            n_jobs=-1,
        )

        random_search.fit(
            X_train,
            y_train
        )

        mlflow.log_metric(
            "best_cv_f1_macro",
            random_search.best_score_
        )

        for param, value in random_search.best_params_.items():

            mlflow.log_param(
                f"best_{param}",
                value
            )

        test_score = random_search.best_estimator_.score(
            X_test,
            y_test
        )

        mlflow.log_metric(
            "test_accuracy",
            test_score
        )

        results_df = pd.DataFrame(
            random_search.cv_results_
        )[
            [
                "params",
                "mean_test_score",
                "std_test_score",
                "rank_test_score"
            ]
        ].sort_values(
            "rank_test_score"
        )

        results_df.to_csv(
            "random_search_all_candidates.csv",
            index=False
        )

        mlflow.log_artifact(
            "random_search_all_candidates.csv"
        )

        print(
            f"Random Search evaluated "
            f"{n_iter} combinations x 5 folds = "
            f"{n_iter * 5} total fits"
        )

        print(
            f"Best params: "
            f"{random_search.best_params_}"
        )

        print(
            f"Best CV f1_macro: "
            f"{random_search.best_score_:.4f}"
        )

        print(
            f"Test accuracy: "
            f"{test_score:.4f}"
        )

        return (
            random_search.best_score_,
            random_search.best_params_
        )


if __name__ == "__main__":

    run_random_search(
        "data/processed/iris_features.csv"
    )