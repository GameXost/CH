import math
import numpy as np

from chm.common import NumericalMethodError
from chm.tridiagonal import sweep_method
from .common import polynomial_value, validate_nodes


def natural_spline(x, y) -> list[tuple[float, float, float, float]]:
    x, y = validate_nodes(x, y)
    n = len(x)
    h = np.array([x[i + 1] - x[i] for i in range(n - 1)])
    slopes = np.array([(y[i + 1] - y[i]) / h[i] for i in range(n - 1)])
    # c_i = S''(x_i) / 2; на концах c = 0.
    c = np.zeros(n)
    if n > 2:
        main = [2 * (h[i - 1] + h[i]) for i in range(1, n - 1)]
        rhs = [3 * (slopes[i] - slopes[i - 1]) for i in range(1, n - 1)]
        _, _, c[1:-1] = sweep_method(h[1:-1], main, h[1:-1], rhs)
    result = []
    for i in range(n - 1):
        a = y[i]
        b = slopes[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
        d = (c[i + 1] - c[i]) / (3 * h[i])
        result.append((float(a), float(b), float(c[i]), float(d)))
    return result


def spline_interval(x, point) -> int:
    if not math.isfinite(point) or not x[0] <= point <= x[-1]:
        raise NumericalMethodError("Точка должна принадлежать отрезку построения сплайна")
    for i in range(len(x) - 2):
        if point < x[i + 1]:
            return i
    return len(x) - 2


def spline_value(x, coefficients, point) -> float:
    i = spline_interval(x, point)
    return float(polynomial_value(coefficients[i], point - x[i]))


def spline_derivative(coefficients, i, offset, order=1) -> float:
    if order not in (1, 2):
        raise NumericalMethodError("Поддерживаются первая и вторая производные сплайна")
    _, b, c, d = coefficients[i]
    if order == 1:
        return float(b + 2 * c * offset + 3 * d * offset * offset)
    return float(2 * c + 6 * d * offset)
