from typing import Iterable

import numpy as np

from .common import DEFAULT_ZERO_TOLERANCE, NumericalMethodError, as_vector


def sweep_method(
    lower_values: Iterable[float],
    main_values: Iterable[float],
    upper_values: Iterable[float],
    rhs_values: Iterable[float],
    zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    main = np.asarray(main_values, dtype=float)
    if main.ndim != 1 or main.size == 0:
        raise NumericalMethodError("Главная диагональ должна быть непустым вектором")

    n = main.size
    main = as_vector(main, n, "главная диагональ")
    lower = as_vector(lower_values, n - 1, "нижняя диагональ")
    upper = as_vector(upper_values, n - 1, "верхняя диагональ")
    rhs = as_vector(rhs_values, n, "правая часть")

    P = np.zeros(n, dtype=float)
    Q = np.zeros(n, dtype=float)

    if abs(main[0]) <= zero_tolerance:
        raise NumericalMethodError("Нулевой знаменатель на шаге прогонки 1")
    P[0] = -upper[0] / main[0] if n > 1 else 0.0
    Q[0] = rhs[0] / main[0]

    for i in range(1, n):
        denominator = main[i] + lower[i - 1] * P[i - 1]
        if abs(denominator) <= zero_tolerance:
            raise NumericalMethodError(f"Нулевой знаменатель на шаге прогонки {i + 1}")
        P[i] = -upper[i] / denominator if i < n - 1 else 0.0
        Q[i] = (rhs[i] - lower[i - 1] * Q[i - 1]) / denominator

    solution = np.zeros(n, dtype=float)
    solution[-1] = Q[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = P[i] * solution[i + 1] + Q[i]

    return P, Q, solution
