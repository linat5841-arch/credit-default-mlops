"""Сравнение конфигураций PD-модели с MLflow."""

import mlflow
import mlflow.sklearn

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score

from train import (
    MLFLOW_EXPERIMENT_NAME,
    PROJECT_ROOT,
    RANDOM_STATE,
    TEST_SIZE,
    build_pipeline,
    evaluate_model,
    load_processed_data,
    split_data,
)

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

CV_FOLDS = 5

EXPERIMENTS = [
    {
        "run_name": "logreg_C_0.01_none",
        "C": 0.01,
        "class_weight": None,
    },
    {
        "run_name": "logreg_C_0.1_none",
        "C": 0.1,
        "class_weight": None,
    },
    {
        "run_name": "logreg_C_1_none",
        "C": 1.0,
        "class_weight": None,
    },
    {
        "run_name": "logreg_C_1_balanced",
        "C": 1.0,
        "class_weight": "balanced",
    },
    {
        "run_name": "logreg_C_10_balanced",
        "C": 10.0,
        "class_weight": "balanced",
    },
]


def configure_mlflow() -> None:
    """Настраиваю локальное хранилище MLflow."""

    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def create_cv() -> StratifiedKFold:
    """Создаю воспроизводимую стратифицированную CV."""

    return StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def evaluate_configuration(
    base_pipeline,
    config: dict,
    X_train,
    y_train,
) -> dict:
    """Оцениваю конфигурацию только на train посредством CV."""

    model = clone(base_pipeline)

    model.set_params(
        model__C=config["C"],
        model__class_weight=config["class_weight"],
    )

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=create_cv(),
        scoring="roc_auc",
        n_jobs=-1,
    )

    return {
        "config": config,
        "cv_roc_auc": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
    }


def log_cv_experiment(result: dict) -> None:
    """Сохраняю результаты CV отдельной конфигурации."""

    config = result["config"]

    with mlflow.start_run(run_name=config["run_name"]):
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "C": config["C"],
                "class_weight": str(config["class_weight"]),
                "cv_folds": CV_FOLDS,
                "cv_strategy": "StratifiedKFold",
                "cv_shuffle": True,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
            }
        )

        mlflow.log_metrics(
            {
                "cv_roc_auc": result["cv_roc_auc"],
                "cv_roc_auc_std": result["cv_roc_auc_std"],
            }
        )

        mlflow.set_tag("evaluation_stage", "cross_validation")

        print(
            f"MLflow: {config['run_name']} | "
            f"CV ROC-AUC: {result['cv_roc_auc']:.4f} "
            f"+/- {result['cv_roc_auc_std']:.4f}"
        )


def select_best_configuration(results: list) -> dict:
    """Выбираю конфигурацию по максимальному CV ROC-AUC."""

    return max(
        results,
        key=lambda result: result["cv_roc_auc"],
    )


def train_and_evaluate_best(
    base_pipeline,
    best_result: dict,
    X_train,
    X_test,
    y_train,
    y_test,
) -> None:
    """Обучаю победителя и однократно оцениваю на test."""

    config = best_result["config"]

    best_model = clone(base_pipeline)

    best_model.set_params(
        model__C=config["C"],
        model__class_weight=config["class_weight"],
    )

    best_model.fit(X_train, y_train)

    # Test используется только после выбора конфигурации.
    test_metrics = evaluate_model(
        best_model,
        X_test,
        y_test,
    )

    with mlflow.start_run(
        run_name="logreg_selected_final",
    ):
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "C": config["C"],
                "class_weight": str(config["class_weight"]),
                "cv_folds": CV_FOLDS,
                "cv_strategy": "StratifiedKFold",
                "cv_shuffle": True,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "selection_metric": "cv_roc_auc",
                "selected_configuration": config["run_name"],
            }
        )

        mlflow.log_metrics(
            {
                "cv_roc_auc": best_result["cv_roc_auc"],
                "cv_roc_auc_std": best_result["cv_roc_auc_std"],
            }
        )

        mlflow.log_metrics(
            {f"test_{name}": value for name, value in test_metrics.items()}
        )

        mlflow.set_tag(
            "evaluation_stage",
            "final_test",
        )

        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            input_example=X_train.head(5),
            serialization_format="cloudpickle",
        )

        run_id = mlflow.active_run().info.run_id

        print("\nФинальная модель сохранена в MLflow.")
        print(f"Выбранная конфигурация: {config['run_name']}")
        print(f"CV ROC-AUC: {best_result['cv_roc_auc']:.4f}")
        print(f"MLflow Run ID: {run_id}")


def main() -> None:
    """Выполняю сравнение моделей и финальную оценку."""

    configure_mlflow()

    df = load_processed_data()

    X_train, X_test, y_train, y_test = split_data(df)

    base_pipeline = build_pipeline(X_train)

    results = []

    for config in EXPERIMENTS:
        print("\n" + "=" * 60)
        print(f"CV-эксперимент: {config['run_name']}")
        print("=" * 60)

        result = evaluate_configuration(
            base_pipeline,
            config,
            X_train,
            y_train,
        )

        log_cv_experiment(result)

        results.append(result)

    best_result = select_best_configuration(results)

    print("\nЛучшая конфигурация по CV:")
    print(best_result["config"])
    print(f"CV ROC-AUC: " f"{best_result['cv_roc_auc']:.4f}")

    train_and_evaluate_best(
        base_pipeline,
        best_result,
        X_train,
        X_test,
        y_train,
        y_test,
    )


if __name__ == "__main__":
    main()
