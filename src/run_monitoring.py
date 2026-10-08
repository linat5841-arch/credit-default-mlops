"""Проверяю распределительный сдвиг признаков кредитного датасета."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.monitor import calculate_psi


def main():
    """Сравниваю эталонную и контрольную выборки с помощью PSI."""

    # Загружаю подготовленный датасет
    df = pd.read_csv("data/processed/credit_default_processed.csv")

    # Разделяю данные на две непересекающиеся выборки
    reference, current = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df["default"],
    )

    # Выбираю числовые признаки для мониторинга
    features = [
        "LIMIT_BAL",
        "AGE",
        "AVG_BILL_AMT",
        "AVG_PAY_AMT",
        "DELAY_MONTHS",
        "MAX_DELAY",
        "CREDIT_UTILIZATION",
    ]

    results = []

    # Рассчитываю PSI для каждого признака
    for feature in features:
        psi = calculate_psi(reference[feature], current[feature])

        if psi < 0.1:
            status = "Незначительный сдвиг"
        elif psi < 0.25:
            status = "Умеренный сдвиг"
        else:
            status = "Существенный сдвиг"

        results.append(
            {
                "feature": feature,
                "psi": round(psi, 4),
                "status": status,
            }
        )

    # Формирую таблицу результатов мониторинга
    report = pd.DataFrame(results).sort_values(
        by="psi",
        ascending=False,
    )

    print("\nРезультаты мониторинга Data Drift:")
    print(report.to_string(index=False))

    # Создаю папку для отчётов мониторинга
    from pathlib import Path

    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняю результаты мониторинга в CSV
    output_path = output_dir / "data_drift_report.csv"
    report.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nОтчёт сохранён: {output_path}")


if __name__ == "__main__":
    main()
