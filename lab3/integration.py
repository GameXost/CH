import math

from chm.common import NumericalMethodError
from .common import finite_number


METHODS = (("left", "Левые прямоугольники", 1),
           ("right", "Правые прямоугольники", 1),
           ("midpoint", "Средние прямоугольники", 2),
           ("trapezoid", "Трапеции", 2),
           ("simpson", "Симпсон", 4))


def interval_count(a, b, h):
    a, b, h = (finite_number(value, name) for value, name in ((a, "a"), (b, "b"), (h, "h")))
    if b <= a or h <= 0:
        raise NumericalMethodError("Требуются a < b и h > 0")
    count = (b - a) / h
    if not math.isfinite(count):
        raise NumericalMethodError("Слишком маленький шаг интегрирования")
    n = round(count)
    if n < 1 or not math.isclose(count, n, rel_tol=1e-10, abs_tol=1e-10):
        raise NumericalMethodError("Шаг должен делить длину отрезка на целое число частей")
    return n


def integrate(function, a, b, h, method):
    n = interval_count(a, b, h)
    if method == "simpson" and n % 2:
        raise NumericalMethodError("Для Симпсона нужно чётное число отрезков")
    if method == "left":
        return h * sum(function(a + i * h) for i in range(n))
    if method == "right":
        return h * sum(function(a + i * h) for i in range(1, n + 1))
    if method == "midpoint":
        return h * sum(function(a + (i + 0.5) * h) for i in range(n))
    if method == "trapezoid":
        return h * ((function(a) + function(b)) / 2
                    + sum(function(a + i * h) for i in range(1, n)))
    if method == "simpson":
        return h / 3 * (function(a) + function(b)
                        + 4 * sum(function(a + i * h) for i in range(1, n, 2))
                        + 2 * sum(function(a + i * h) for i in range(2, n, 2)))
    raise NumericalMethodError(f"Неизвестный метод: {method}")


def runge_romberg(coarse, fine, h1, h2, order):
    h1, h2 = finite_number(h1, "h1"), finite_number(h2, "h2")
    if not h1 > h2 > 0 or order <= 0:
        raise NumericalMethodError("Для Рунге–Ромберга нужны h1 > h2 > 0 и p > 0")
    correction = (fine - coarse) / ((h1 / h2) ** order - 1)
    return fine + correction, abs(correction)
