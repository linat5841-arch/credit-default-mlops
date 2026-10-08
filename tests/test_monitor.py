"""Проверяю корректность расчёта Population Stability Index."""

import numpy as np
import pandas as pd
import pytest

from src.monitor import calculate_psi


def test_psi_identical_distributions():
    """Проверяю, что одинаковые распределения дают PSI, равный нулю."""
    reference = pd.Series(np.arange(100))
    current = reference.copy()

    psi = calculate_psi(reference, current)

    assert psi == pytest.approx(0.0)


def test_psi_detects_distribution_shift():
    """Проверяю, что PSI обнаруживает заметный сдвиг распределения."""
    reference = pd.Series(np.arange(1000))
    current = pd.Series(np.arange(500, 1500))

    psi = calculate_psi(reference, current)

    assert psi > 0.25


def test_psi_rejects_empty_samples():
    """Проверяю обработку пустой выборки."""
    reference = pd.Series(dtype=float)
    current = pd.Series([1, 2, 3])

    with pytest.raises(ValueError, match="не должны быть пустыми"):
        calculate_psi(reference, current)


def test_psi_rejects_invalid_bins():
    """Проверяю ограничение на количество интервалов."""
    reference = pd.Series([1, 2, 3])
    current = pd.Series([2, 3, 4])

    with pytest.raises(ValueError, match="не меньше 2"):
        calculate_psi(reference, current, bins=1)
