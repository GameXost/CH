from chm.common import NumericalMethodError
from .common import finite_number, validate_nodes


def differentiate(x, y, point):
    x, y = validate_nodes(x, y, min_count=3)
    point = finite_number(point, "x*")
    if point < x[0] or point > x[-1]:
        raise NumericalMethodError("x* должна находиться внутри таблицы")
    # В табличном узле используем центральную тройку.
    left = 0
    while left < len(x) - 2 and point > x[left + 1]:
        left += 1
    left = min(left, len(x) - 3)
    x0, x1, x2 = x[left:left + 3]
    y0, y1, y2 = y[left:left + 3]
    first = (y1 - y0) / (x1 - x0)
    second = ((y2 - y1) / (x2 - x1) - first) / (x2 - x0)
    derivative = first + second * (2 * point - x0 - x1)
    return float(derivative), float(2 * second), (left, left + 1, left + 2)
