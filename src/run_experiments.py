import mlflow
import mlflow.sklearn
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

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

# Задаю путь к локальной базе MLflow
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"


# Определяю четыре дополнительные конфигурации модели
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
]


def configure_mlflow() -> None:
    """Настраиваю MLflow для сравнительных экспериментов."""

    # Формирую URI локальной SQLite-базы
    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

    # Подключаюсь к локальному MLflow Tracking
    mlflow.set_tracking_uri(tracking_uri)

    # Использую тот же эксперимент, что и при основном обучении
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def run_experiment(
    base_pipeline,
    config: dict,
    X_train,
    X_test,
    y_train,
    y_test,
) -> None:
    """Обучаю и логирую одну конфигурацию модели."""

    # Создаю независимую копию Pipeline
    model = clone(base_pipeline)

    # Устанавливаю параметры текущего эксперимента
    model.set_params(
        model__C=config["C"],
        model__class_weight=config["class_weight"],
    )

    # Оцениваю модель кросс-валидацией только на train
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    # Обучаю модель на всей обучающей выборке
    model.fit(
        X_train,
        y_train,
    )

    # Оцениваю модель на неизменной тестовой выборке
    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    # Начинаю отдельный MLflow run
    with mlflow.start_run(
        run_name=config["run_name"],
    ):
        # Логирую тип модели
        mlflow.log_param(
            "model_type",
            "LogisticRegression",
        )

        # Логирую коэффициент регуляризации
        mlflow.log_param(
            "C",
            config["C"],
        )

        # Логирую способ учёта дисбаланса классов
        mlflow.log_param(
            "class_weight",
            str(config["class_weight"]),
        )

        # Логирую количество фолдов кросс-валидации
        mlflow.log_param(
            "cv_folds",
            5,
        )

        # Логирую размер тестовой выборки
        mlflow.log_param(
            "test_size",
            TEST_SIZE,
        )

        # Логирую random state
        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
        )

        # Логирую средний ROC-AUC кросс-валидации
        mlflow.log_metric(
            "cv_roc_auc",
            cv_scores.mean(),
        )

        # Логирую метрики на тестовой выборке
        mlflow.log_metrics(metrics)

        # Создаю пример входных данных
        input_example = X_train.head(5)

        # Логирую обученный Pipeline
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example,
            serialization_format="cloudpickle",
        )

        # Получаю идентификатор текущего run
        run_id = mlflow.active_run().info.run_id

        print(f"\nЭксперимент {config['run_name']} " f"успешно сохранён в MLflow.")
        print(f"Средний CV ROC-AUC: " f"{cv_scores.mean():.4f}")
        print(f"MLflow Run ID: {run_id}")


if __name__ == "__main__":
    # Настраиваю MLflow
    configure_mlflow()

    # Загружаю подготовленный датасет
    df = load_processed_data()

    # Использую одинаковое разделение train/test
    X_train, X_test, y_train, y_test = split_data(df)

    # Создаю базовый Pipeline
    base_pipeline = build_pipeline(X_train)

    # Последовательно запускаю четыре эксперимента
    for experiment_config in EXPERIMENTS:
        print("\n" + "=" * 60)
        print(f"Запускаю эксперимент: " f"{experiment_config['run_name']}")
        print("=" * 60)

        run_experiment(
            base_pipeline,
            experiment_config,
            X_train,
            X_test,
            y_train,
            y_test,
        )
