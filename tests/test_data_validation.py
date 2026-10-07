from pathlib import Path

import pandas as pd
import pytest

from src.validate_data import validate_data, validate_numeric_ranges

# Определяю путь к тестовым данным
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "valid_credit_data.csv"


def test_valid_data_passes_validation() -> None:
    """Проверяю, что корректные данные проходят полную валидацию."""

    # Проверяю корректный тестовый датасет через Great Expectations
    validate_data(FIXTURE_PATH)


def test_anomalous_age_is_detected(tmp_path: Path) -> None:
    """Проверяю, что аномальный возраст не проходит валидацию."""

    # Загружаю корректные тестовые данные
    df = pd.read_csv(FIXTURE_PATH)

    # Создаю копию тестовых данных
    anomalous_df = df.copy()

    # Добавляю заведомо некорректный возраст
    anomalous_df.loc[0, "AGE"] = -5

    # Сохраняю испорченные данные во временный файл
    test_file = tmp_path / "anomalous_data.csv"
    anomalous_df.to_csv(test_file, index=False)

    # Проверяю, что Great Expectations обнаруживает аномалию
    with pytest.raises(ValueError):
        validate_numeric_ranges(test_file)
