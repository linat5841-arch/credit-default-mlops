from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Определяю корневую директорию проекта
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Задаю путь к подготовленным данным
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "credit_default_processed.csv"
)

# Задаю директорию для сохранения графиков
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Задаю директорию для сохранения обученной модели
MODELS_DIR = PROJECT_ROOT / "models"

# Задаю путь к файлу модели
MODEL_PATH = MODELS_DIR / "credit_default_model.joblib"

# Задаю путь к локальной базе MLflow
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

# Задаю имя MLflow-эксперимента
MLFLOW_EXPERIMENT_NAME = "credit_default_pd_model"

# Задаю имя целевой переменной
TARGET_COLUMN = "default"

# Задаю параметры воспроизводимости
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Определяю категориальные признаки
CATEGORICAL_FEATURES = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE_GROUP",
]


def load_processed_data(
    path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Загружаю подготовленный датасет."""

    if not path.exists():
        raise FileNotFoundError(f"Файл с подготовленными данными не найден: {path}")

    df = pd.read_csv(path)

    print(
        f"Подготовленные данные загружены: "
        f"{df.shape[0]} строк, {df.shape[1]} столбцов"
    )

    return df


def split_data(
    df: pd.DataFrame,
):
    """Разделяю данные на обучающую и тестовую выборки."""

    # Отделяю признаки от целевой переменной
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Разделяю данные со стратификацией
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Обучающая выборка: {X_train.shape}")
    print(f"Тестовая выборка: {X_test.shape}")
    print(f"Доля дефолтов в train: {y_train.mean():.4f}")
    print(f"Доля дефолтов в test: {y_test.mean():.4f}")

    return X_train, X_test, y_train, y_test


def build_pipeline(
    X_train: pd.DataFrame,
) -> Pipeline:
    """Создаю пайплайн предобработки и обучения модели."""

    # Определяю числовые признаки
    numerical_features = [
        column for column in X_train.columns if column not in CATEGORICAL_FEATURES
    ]

    # Создаю пайплайн обработки числовых признаков
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    # Создаю пайплайн обработки категориальных признаков
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    # Объединяю обработку признаков
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "cat",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    # Создаю единый Pipeline
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return pipeline


def tune_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """Подбираю гиперпараметры модели на обучающей выборке."""

    # Задаю сетку гиперпараметров
    param_grid = {
        "model__C": [
            0.01,
            0.1,
            1.0,
            10.0,
        ],
        "model__class_weight": [
            None,
            "balanced",
        ],
    }

    # Создаю поиск по сетке
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        refit=True,
    )

    # Выполняю подбор только на обучающей выборке
    grid_search.fit(
        X_train,
        y_train,
    )

    print("\nПодбор гиперпараметров завершён.")
    print(f"Лучший ROC-AUC на CV: " f"{grid_search.best_score_:.4f}")
    print(f"Лучшие параметры: " f"{grid_search.best_params_}")

    return grid_search


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Оцениваю лучшую модель на тестовой выборке."""

    # Получаю прогноз класса
    y_pred = model.predict(X_test)

    # Получаю вероятность дефолта
    y_proba = model.predict_proba(X_test)[:, 1]

    # Рассчитываю метрики качества
    metrics = {
        "accuracy": accuracy_score(
            y_test,
            y_pred,
        ),
        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            y_proba,
        ),
    }

    print("\nМетрики лучшей модели на тестовой выборке:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    return metrics


def save_roc_curve(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Path:
    """Строю и сохраняю ROC-кривую модели."""

    # Создаю директорию для графиков
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Получаю вероятности дефолта
    y_proba = model.predict_proba(X_test)[:, 1]

    # Строю ROC-кривую
    RocCurveDisplay.from_predictions(
        y_test,
        y_proba,
        name="Logistic Regression",
    )

    # Оформляю график
    plt.title("ROC Curve — Logistic Regression")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Задаю путь к графику
    output_path = FIGURES_DIR / "roc_curve.png"

    # Сохраняю график
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    # Закрываю график
    plt.close()

    print(f"ROC-кривая сохранена: {output_path}")

    return output_path


def save_model(
    model,
    path: Path = MODEL_PATH,
) -> None:
    """Сохраняю обученную модель."""

    # Создаю директорию для модели
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Сохраняю весь обученный Pipeline
    joblib.dump(
        model,
        path,
    )

    print(f"Модель сохранена: {path}")


def configure_mlflow() -> None:
    """Настраиваю локальное хранилище MLflow."""

    # Формирую URI локальной SQLite-базы
    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

    # Настраиваю MLflow Tracking
    mlflow.set_tracking_uri(tracking_uri)

    # Выбираю эксперимент проекта
    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME,
    )


def log_mlflow_run(
    model,
    grid_search: GridSearchCV,
    metrics: dict,
    roc_curve_path: Path,
    X_train: pd.DataFrame,
) -> None:
    """Логирую параметры, метрики и артефакты в MLflow."""

    # Начинаю MLflow run
    with mlflow.start_run(run_name="logistic_regression_tuned"):
        # Логирую параметры эксперимента
        mlflow.log_param(
            "model_type",
            "LogisticRegression",
        )
        mlflow.log_param(
            "C",
            grid_search.best_params_["model__C"],
        )
        mlflow.log_param(
            "class_weight",
            str(grid_search.best_params_["model__class_weight"]),
        )
        mlflow.log_param(
            "cv_folds",
            5,
        )
        mlflow.log_param(
            "test_size",
            TEST_SIZE,
        )
        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
        )

        # Логирую качество на кросс-валидации
        mlflow.log_metric(
            "cv_roc_auc",
            grid_search.best_score_,
        )

        # Логирую метрики на тестовой выборке
        mlflow.log_metrics(metrics)

        # Логирую ROC-кривую как артефакт
        mlflow.log_artifact(
            str(roc_curve_path),
            artifact_path="figures",
        )

        # Создаю пример входных данных для сигнатуры
        input_example = X_train.head(5)

        # Логирую весь sklearn Pipeline в MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example,
            serialization_format="cloudpickle",
        )

        # Получаю идентификатор текущего run
        run_id = mlflow.active_run().info.run_id

        print("\nMLflow run успешно сохранён.")
        print(f"MLflow Run ID: {run_id}")


if __name__ == "__main__":
    # Настраиваю MLflow
    configure_mlflow()

    # Загружаю подготовленные данные
    df = load_processed_data()

    # Разделяю данные на train и test
    X_train, X_test, y_train, y_test = split_data(df)

    # Создаю Pipeline
    pipeline = build_pipeline(X_train)

    # Подбираю гиперпараметры только на train
    grid_search = tune_model(
        pipeline,
        X_train,
        y_train,
    )

    # Получаю лучшую модель
    best_model = grid_search.best_estimator_

    # Оцениваю лучшую модель на test
    metrics = evaluate_model(
        best_model,
        X_test,
        y_test,
    )

    # Строю и сохраняю ROC-кривую
    roc_curve_path = save_roc_curve(
        best_model,
        X_test,
        y_test,
    )

    # Сохраняю лучшую обученную модель
    save_model(
        best_model,
    )

    # Логирую эксперимент в MLflow
    log_mlflow_run(
        best_model,
        grid_search,
        metrics,
        roc_curve_path,
        X_train,
    )
