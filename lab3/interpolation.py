import numpy as np
from .common import validate_nodes


def lagrange_value(x, y, point) -> float:
    result = 0.0
    for i in range(len(x)):
        term = y[i]
        for j in range(len(x)):
            if j != i:
                term *= (point - x[j]) / (x[i] - x[j])
        result += term
    return float(result)


def divided_differences(x, y) -> np.ndarray:
    x, y = validate_nodes(x, y)
    n = len(x)
    table = np.zeros((n, n))
    table[:, 0] = y
    for order in range(1, n):
        for i in range(n - order):
            table[i, order] = (
                (table[i + 1, order - 1] - table[i, order - 1])
                / (x[i + order] - x[i])
            )
    return table


def newton_value(x, table, point) -> float:
    result = table[0, len(x) - 1]
    for i in range(len(x) - 2, -1, -1):
        result = table[0, i] + (point - x[i]) * result
    return float(result)
