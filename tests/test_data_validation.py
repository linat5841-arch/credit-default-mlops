from pathlib import Path

import pandas as pd
import pytest

from src.validate_data import validate_numeric_ranges

# Определяю корневую директорию проекта
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Задаю путь к исходным данным
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "UCI_Credit_Card.csv"


def test_anomalous_age_is_detected(tmp_path: Path) -> None:
    """Проверяю, что аномальный возраст не проходит валидацию."""

    # Загружаю корректные исходные данные
    df = pd.read_csv(RAW_DATA_PATH)

    # Создаю небольшую выборку для теста
    anomalous_df = df.head(100).copy()

    # Добавляю заведомо некорректный возраст
    anomalous_df.loc[0, "AGE"] = -5

    # Сохраняю испорченные данные во временный файл
    test_file = tmp_path / "anomalous_data.csv"
    anomalous_df.to_csv(test_file, index=False)

    # Проверяю, что Great Expectations обнаруживает аномалию
    with pytest.raises(ValueError):
        validate_numeric_ranges(test_file)
