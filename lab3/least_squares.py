import numpy as np

from chm.common import NumericalMethodError
from chm.lu import lu_decomposition, solve_with_lu
from .common import polynomial_value, validate_nodes


def least_squares(x, y, degree):
    if degree not in (1, 2, 3):
        raise NumericalMethodError("Степень МНК должна быть 1, 2 или 3")
    x, y = validate_nodes(x, y, min_count=degree + 1, ordered=False)
    if len(set(x)) < degree + 1:
        raise NumericalMethodError("Для МНК нужно не менее degree+1 различных значений x")
    sums = [sum(float(xi) ** power for xi in x) for power in range(2 * degree + 1)]
    matrix = np.array([[sums[i + j] for j in range(degree + 1)]
                       for i in range(degree + 1)])
    rhs = np.array([sum(float(yi) * float(xi) ** i for xi, yi in zip(x, y))
                    for i in range(degree + 1)])
    coefficients = solve_with_lu(lu_decomposition(matrix), rhs)
    mse = sum((polynomial_value(coefficients, xi) - yi) ** 2
              for xi, yi in zip(x, y)) / len(x)
    return coefficients, float(mse)
