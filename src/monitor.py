"""Оцениваю изменение распределения числовых признаков с помощью PSI."""

import numpy as np
import pandas as pd


def calculate_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """Рассчитываю PSI между эталонной и текущей выборками."""

    if bins < 2:
        raise ValueError("Количество интервалов должно быть не меньше 2.")

    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()

    if reference.empty or current.empty:
        raise ValueError("Выборки не должны быть пустыми.")

    if reference.nunique() == 1:
        value = reference.iloc[0]
        edges = np.array([-np.inf, value, np.inf])
    else:
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(reference, quantiles))

        if len(edges) < 2:
            raise ValueError("Не удалось построить интервалы.")

        edges[0] = -np.inf
        edges[-1] = np.inf

    reference_counts = np.histogram(reference, bins=edges)[0]
    current_counts = np.histogram(current, bins=edges)[0]

    reference_share = reference_counts / reference_counts.sum()
    current_share = current_counts / current_counts.sum()

    epsilon = 1e-6

    reference_share = np.clip(reference_share, epsilon, None)
    current_share = np.clip(current_share, epsilon, None)

    psi = np.sum(
        (current_share - reference_share) * np.log(current_share / reference_share)
    )

    return float(psi)
