"""Проверка качества данных Credit Default с Great Expectations."""

from pathlib import Path

import great_expectations as gx

# Корневая директория проекта
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Исходный датасет
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "UCI_Credit_Card.csv"

# Ожидаемая структура данных
EXPECTED_COLUMNS = [
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    "default.payment.next.month",
]

# Строгие ожидаемые типы столбцов
TYPE_RULES = {
    "ID": "int64",
    "LIMIT_BAL": "float64",
    "SEX": "int64",
    "EDUCATION": "int64",
    "MARRIAGE": "int64",
    "AGE": "int64",
    "PAY_0": "int64",
    "PAY_2": "int64",
    "PAY_3": "int64",
    "PAY_4": "int64",
    "PAY_5": "int64",
    "PAY_6": "int64",
    "BILL_AMT1": "float64",
    "BILL_AMT2": "float64",
    "BILL_AMT3": "float64",
    "BILL_AMT4": "float64",
    "BILL_AMT5": "float64",
    "BILL_AMT6": "float64",
    "PAY_AMT1": "float64",
    "PAY_AMT2": "float64",
    "PAY_AMT3": "float64",
    "PAY_AMT4": "float64",
    "PAY_AMT5": "float64",
    "PAY_AMT6": "float64",
    "default.payment.next.month": "int64",
}

# Категориальные ограничения
CATEGORY_RULES = {
    "SEX": [1, 2],
    "EDUCATION": [0, 1, 2, 3, 4, 5, 6],
    "MARRIAGE": [0, 1, 2, 3],
}

# Допустимые числовые диапазоны
RANGE_RULES = {
    "AGE": (18, 100),
    "LIMIT_BAL": (1, 2_000_000),
    "PAY_0": (-2, 8),
    "PAY_2": (-2, 8),
    "PAY_3": (-2, 8),
    "PAY_4": (-2, 8),
    "PAY_5": (-2, 8),
    "PAY_6": (-2, 8),
}


def create_expectation_suite():
    """Создаёт единый набор правил Great Expectations."""

    # Инициализация Data Context для GX 1.24.0
    gx.get_context()

    suite = gx.ExpectationSuite(name="credit_default_validation")

    # 1. Структура и порядок столбцов
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchOrderedList(
            column_list=EXPECTED_COLUMNS
        )
    )

    # 2. Отсутствие пропусков
    for column in EXPECTED_COLUMNS:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=column)
        )

    # 3. Уникальность идентификаторов
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="ID"))

    # 4. Бинарная целевая переменная
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="default.payment.next.month",
            value_set=[0, 1],
        )
    )

    # 5. Категориальные ограничения
    for column, allowed_values in CATEGORY_RULES.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column=column,
                value_set=allowed_values,
            )
        )

    # 6. Числовые диапазоны
    for column, (min_value, max_value) in RANGE_RULES.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column,
                min_value=min_value,
                max_value=max_value,
            )
        )

    # 7. Неотрицательные суммы платежей
    for i in range(1, 7):
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=f"PAY_AMT{i}",
                min_value=0,
            )
        )

    # 8. Строгие типы данных
    for column, expected_type in TYPE_RULES.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeOfType(
                column=column,
                type_=expected_type,
            )
        )

    return suite


def _get_batch(path: Path):
    """Загружает CSV как batch Great Expectations."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Файл датасета не найден: {path}")

    context = gx.get_context()

    return context.data_sources.pandas_default.read_csv(path)


def _validate_expectations(batch, expectations):
    """Выполняет набор правил и сообщает о нарушениях."""

    for expectation in expectations:
        result = batch.validate(expectation)

        if not result.success:
            expectation_name = type(expectation).__name__

            column = getattr(
                expectation,
                "column",
                None,
            )

            raise ValueError(
                "Валидация не пройдена: " f"{expectation_name}, " f"столбец: {column}."
            )


def validate_numeric_ranges(
    path: Path = RAW_DATA_PATH,
) -> None:
    """Проверяет числовые диапазоны через единый suite."""

    batch = _get_batch(path)
    suite = create_expectation_suite()

    expectations = [
        expectation
        for expectation in suite.expectations
        if (
            isinstance(
                expectation,
                gx.expectations.ExpectColumnValuesToBeBetween,
            )
            and expectation.column in RANGE_RULES
        )
    ]

    _validate_expectations(batch, expectations)

    print("Проверка числовых диапазонов: True")


def validate_data(path: Path = RAW_DATA_PATH) -> None:
    """Выполняет полную валидацию входного датасета."""

    batch = _get_batch(path)
    suite = create_expectation_suite()

    _validate_expectations(
        batch,
        suite.expectations,
    )

    print("Great Expectations: " f"{len(suite.expectations)} проверок пройдено.")
    print("Все проверки данных успешно пройдены.")


if __name__ == "__main__":
    validate_data()
