from pathlib import Path

import pandas as pd

# Определяю корневую директорию проекта
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Задаю пути к исходным и обработанным данным
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "UCI_Credit_Card.csv"
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "credit_default_processed.csv"
)

# Задаю имя целевой переменной
TARGET_COLUMN = "default.payment.next.month"

# Задаю группы признаков истории платежей
PAY_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Загружаю исходный датасет из CSV-файла."""
    if not path.exists():
        raise FileNotFoundError(f"Файл с исходными данными не найден: {path}")

    df = pd.read_csv(path)

    print(f"Исходные данные загружены: {df.shape[0]} строк, {df.shape[1]} столбцов")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Выполняю первичную очистку исходных данных."""
    df = df.copy()

    # Удаляю идентификатор клиента, так как он не является предиктором

    df = df.drop(columns=["ID"], errors="ignore")

    # Объединяю неопределённые и редкие категории образования в категорию Other = 4
    df["EDUCATION"] = df["EDUCATION"].replace(
        {
            0: 4,
            5: 4,
            6: 4,
        }
    )

    # Объединяю неопределённую категорию семейного положения с Other = 3
    df["MARRIAGE"] = df["MARRIAGE"].replace(
        {
            0: 3,
        }
    )

    # Переименовываю целевую переменную для удобства дальнейшей работы
    df = df.rename(columns={TARGET_COLUMN: "default"})

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создаю дополнительные признаки на основе истории клиента."""
    df = df.copy()

    # Создаю средний размер выставленного счёта за 6 месяцев
    df["AVG_BILL_AMT"] = df[BILL_COLUMNS].mean(axis=1)

    # Создаю средний размер фактического платежа за 6 месяцев
    df["AVG_PAY_AMT"] = df[PAYMENT_COLUMNS].mean(axis=1)

    # Считаю количество месяцев, в которых наблюдалась просрочка
    df["DELAY_MONTHS"] = (df[PAY_STATUS_COLUMNS] > 0).sum(axis=1)

    # Определяю максимальный статус просрочки за наблюдаемый период
    df["MAX_DELAY"] = df[PAY_STATUS_COLUMNS].max(axis=1)

    # Рассчитываю отношение среднего счёта к кредитному лимиту
    df["CREDIT_UTILIZATION"] = df["AVG_BILL_AMT"] / df["LIMIT_BAL"]

    # Выполняю биннинг возраста
    df["AGE_GROUP"] = pd.cut(
        df["AGE"],
        bins=[20, 30, 40, 50, 60, float("inf")],
        labels=["21-30", "31-40", "41-50", "51-60", "61+"],
        include_lowest=True,
    )

    return df


def save_data(
    df: pd.DataFrame,
    path: Path = PROCESSED_DATA_PATH,
) -> None:
    """Сохраняю подготовленный датасет в CSV-файл."""
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

    print(f"Обработанные данные сохранены: {path}")


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = create_features(df)
    save_data(df)

    print(
        f"Подготовка данных завершена: " f"{df.shape[0]} строк, {df.shape[1]} столбцов"
    )
