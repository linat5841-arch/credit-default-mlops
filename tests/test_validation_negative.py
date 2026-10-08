from pathlib import Path

import pandas as pd
import pytest

from src.validate_data import RAW_DATA_PATH, validate_data


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
    """Great Expectations must reject anomalous data."""

    df = pd.read_csv(RAW_DATA_PATH)

    df.loc[0, column] = invalid_value

    invalid_file = tmp_path / "invalid_data.csv"
    df.to_csv(invalid_file, index=False)

    with pytest.raises(ValueError, match="Валидация не пройдена"):
        validate_data(invalid_file)
