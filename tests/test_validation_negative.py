"""Отрицательные тесты валидации Great Expectations."""

from pathlib import Path

import pandas as pd
import pytest

from src.validate_data import validate_data

# Тестовый датасет, включённый в Git-репозиторий
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "valid_credit_data.csv"


@pytest.mark.parametrize(
    "column, invalid_value",
    [
        ("AGE", 150),
        ("LIMIT_BAL", -1000),
        ("SEX", 99),
        ("default.payment.next.month", 5),
        ("PAY_AMT1", -500),
        ("AGE", None),
    ],
)
def test_invalid_data_is_rejected(
    tmp_path: Path,
    column: str,
    invalid_value,
) -> None:
    """Проверяет отклонение данных с аномальными значениями."""

    # Загружаем корректный тестовый датасет
    df = pd.read_csv(FIXTURE_PATH)

    # Создаём копию, чтобы не изменять исходную фикстуру
    invalid_df = df.copy()

    # Добавляем некорректное значение
    invalid_df.loc[0, column] = invalid_value

    # Сохраняем аномальные данные во временный CSV
    invalid_file = tmp_path / "invalid_data.csv"
    invalid_df.to_csv(invalid_file, index=False)

    # Great Expectations должен отклонить аномальные данные
    with pytest.raises(
        ValueError,
        match="Валидация не пройдена",
    ):
        validate_data(invalid_file)
