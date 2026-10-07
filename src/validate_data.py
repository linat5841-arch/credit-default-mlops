from pathlib import Path

import great_expectations as gx

# Определяю корневую директорию проекта
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Задаю путь к исходным данным
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "UCI_Credit_Card.csv"

# Задаю ожидаемую структуру исходного датасета
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


def validate_data_types(path: Path = RAW_DATA_PATH) -> None:
    """Проверяю типы данных основных столбцов исходного датасета."""
    context = gx.get_context()

    batch = context.data_sources.pandas_default.read_csv(path)

    type_rules = {
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

    for column, expected_type in type_rules.items():
        expectation = gx.expectations.ExpectColumnValuesToBeOfType(
            column=column,
            type_=expected_type,
        )

        result = batch.validate(expectation)

        if not result.success:
            raise ValueError(
                f"Валидация не пройдена: столбец {column} "
                f"должен иметь тип {expected_type}."
            )

    print("Проверка типов данных: True")


def validate_columns(path: Path = RAW_DATA_PATH) -> None:
    """Проверяю структуру и порядок столбцов исходного датасета."""
    context = gx.get_context()

    batch = context.data_sources.pandas_default.read_csv(path)

    expectation = gx.expectations.ExpectTableColumnsToMatchOrderedList(
        column_list=EXPECTED_COLUMNS
    )

    result = batch.validate(expectation)

    print(f"Проверка структуры столбцов: {result.success}")

    if not result.success:
        raise ValueError(
            "Валидация не пройдена: структура входного датасета "
            "не соответствует ожидаемой."
        )


def validate_target(path: Path = RAW_DATA_PATH) -> None:
    """Проверяю допустимые значения целевой переменной."""
    context = gx.get_context()

    batch = context.data_sources.pandas_default.read_csv(path)

    expectation = gx.expectations.ExpectColumnValuesToBeInSet(
        column="default.payment.next.month",
        value_set=[0, 1],
    )

    result = batch.validate(expectation)

    print(f"Проверка целевой переменной: {result.success}")

    if not result.success:
        raise ValueError(
            "Валидация не пройдена: целевая переменная содержит "
            "значения, отличные от 0 и 1."
        )


def validate_categories(path: Path = RAW_DATA_PATH) -> None:
    """Проверяю допустимые значения категориальных признаков."""
    context = gx.get_context()

    batch = context.data_sources.pandas_default.read_csv(path)

    category_rules = {
        "SEX": [1, 2],
        "EDUCATION": [0, 1, 2, 3, 4, 5, 6],
        "MARRIAGE": [0, 1, 2, 3],
    }

    for column, allowed_values in category_rules.items():
        expectation = gx.expectations.ExpectColumnValuesToBeInSet(
            column=column,
            value_set=allowed_values,
        )

        result = batch.validate(expectation)

        if not result.success:
            raise ValueError(
                f"Валидация не пройдена: столбец {column} "
                "содержит недопустимые значения."
            )

    print("Проверка категориальных признаков: True")


def validate_missing_values(path: Path = RAW_DATA_PATH) -> None:
    """Проверяю отсутствие пропущенных значений во всех столбцах."""
    context = gx.get_context()

    batch = context.data_sources.pandas_default.read_csv(path)

    for column in EXPECTED_COLUMNS:
        expectation = gx.expectations.ExpectColumnValuesToNotBeNull(column=column)

        result = batch.validate(expectation)

        if not result.success:
            raise ValueError(
                f"Валидация не пройдена: в столбце {column} "
                "обнаружены пропущенные значения."
            )

    print("Проверка пропущенных значений: True")


def validate_numeric_ranges(
    path: Path = RAW_DATA_PATH,
) -> None:
    """Проверяю допустимые диапазоны числовых признаков."""
    context = gx.get_context()

    batch = context.data_sources.pandas_default.read_csv(path)

    range_rules = {
        "AGE": (18, 100),
        "LIMIT_BAL": (1, 2_000_000),
        "PAY_0": (-2, 8),
        "PAY_2": (-2, 8),
        "PAY_3": (-2, 8),
        "PAY_4": (-2, 8),
        "PAY_5": (-2, 8),
        "PAY_6": (-2, 8),
    }

    for column, (min_value, max_value) in range_rules.items():
        expectation = gx.expectations.ExpectColumnValuesToBeBetween(
            column=column,
            min_value=min_value,
            max_value=max_value,
        )

        result = batch.validate(expectation)

        if not result.success:
            raise ValueError(
                f"Валидация не пройдена: столбец {column} "
                f"содержит значения вне диапазона "
                f"[{min_value}, {max_value}]."
            )

    print("Проверка числовых диапазонов: True")


def validate_payment_amounts(
    path: Path = RAW_DATA_PATH,
) -> None:
    """Проверяю, что суммы платежей не являются отрицательными."""
    context = gx.get_context()

    batch = context.data_sources.pandas_default.read_csv(path)

    payment_columns = [
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]

    for column in payment_columns:
        expectation = gx.expectations.ExpectColumnValuesToBeBetween(
            column=column,
            min_value=0,
        )

        result = batch.validate(expectation)

        if not result.success:
            raise ValueError(
                f"Валидация не пройдена: столбец {column} "
                "содержит отрицательные суммы платежей."
            )

    print("Проверка сумм платежей: True")


def validate_data(path: Path = RAW_DATA_PATH) -> None:
    """Проверяю входной датасет перед его дальнейшей обработкой."""
    validate_columns(path)
    validate_data_types(path)
    validate_target(path)
    validate_categories(path)
    validate_missing_values(path)
    validate_numeric_ranges(path)
    validate_payment_amounts(path)

    print("Все проверки данных успешно пройдены.")


if __name__ == "__main__":
    validate_data()
