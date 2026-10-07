import pandas as pd

from src.prepare import clean_data, create_features


def test_clean_data() -> None:
    """Проверяю корректность очистки исходных данных."""

    # Создаю небольшой тестовый DataFrame
    df = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "EDUCATION": [0, 5, 2],
            "MARRIAGE": [0, 1, 2],
            "default.payment.next.month": [0, 1, 0],
        }
    )

    # Выполняю очистку данных
    cleaned_df = clean_data(df)

    # Проверяю удаление идентификатора
    assert "ID" not in cleaned_df.columns

    # Проверяю объединение редких категорий образования
    assert cleaned_df["EDUCATION"].tolist() == [4, 4, 2]

    # Проверяю обработку неизвестной категории семейного положения
    assert cleaned_df["MARRIAGE"].tolist() == [3, 1, 2]

    # Проверяю переименование целевой переменной
    assert "default" in cleaned_df.columns
    assert "default.payment.next.month" not in cleaned_df.columns

    # Проверяю сохранение значений целевой переменной
    assert cleaned_df["default"].tolist() == [0, 1, 0]


def test_create_features() -> None:
    """Проверяю корректность создания новых признаков."""

    # Создаю небольшой тестовый DataFrame
    df = pd.DataFrame(
        {
            "LIMIT_BAL": [100000.0, 200000.0],
            "AGE": [25, 45],
            "PAY_0": [1, 0],
            "PAY_2": [2, 0],
            "PAY_3": [0, 0],
            "PAY_4": [0, 0],
            "PAY_5": [0, 0],
            "PAY_6": [0, 0],
            "BILL_AMT1": [50000.0, 100000.0],
            "BILL_AMT2": [50000.0, 100000.0],
            "BILL_AMT3": [50000.0, 100000.0],
            "BILL_AMT4": [50000.0, 100000.0],
            "BILL_AMT5": [50000.0, 100000.0],
            "BILL_AMT6": [50000.0, 100000.0],
            "PAY_AMT1": [10000.0, 20000.0],
            "PAY_AMT2": [10000.0, 20000.0],
            "PAY_AMT3": [10000.0, 20000.0],
            "PAY_AMT4": [10000.0, 20000.0],
            "PAY_AMT5": [10000.0, 20000.0],
            "PAY_AMT6": [10000.0, 20000.0],
        }
    )

    # Создаю новые признаки
    featured_df = create_features(df)

    # Проверяю среднюю сумму задолженности
    assert featured_df["AVG_BILL_AMT"].tolist() == [
        50000.0,
        100000.0,
    ]

    # Проверяю среднюю сумму платежей
    assert featured_df["AVG_PAY_AMT"].tolist() == [
        10000.0,
        20000.0,
    ]

    # Проверяю количество месяцев с просрочкой
    assert featured_df["DELAY_MONTHS"].tolist() == [2, 0]

    # Проверяю максимальную глубину просрочки
    assert featured_df["MAX_DELAY"].tolist() == [2, 0]

    # Проверяю коэффициент использования кредитного лимита
    assert featured_df["CREDIT_UTILIZATION"].tolist() == [0.5, 0.5]

    # Проверяю возрастные группы
    assert featured_df["AGE_GROUP"].astype(str).tolist() == [
        "21-30",
        "41-50",
    ]