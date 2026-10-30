from typing import Iterable
import numpy as np


DEFAULT_ZERO_TOLERANCE = 1e-12
DEFAULT_MAX_ITERATIONS = 10_000


class NumericalMethodError(ValueError):
    pass


class SingularMatrixError(NumericalMethodError):
    pass


class ConvergenceError(NumericalMethodError):
    pass


def as_square_matrix(values: Iterable[Iterable[float]], name: str = "A") -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise NumericalMethodError(f"{name} должна быть квадратной матрицей")
    if matrix.shape[0] == 0:
        raise NumericalMethodError(f"{name} не должна быть пустой")
    if not np.all(np.isfinite(matrix)):
        raise NumericalMethodError(f"{name} содержит некорректные значения")
    return matrix.copy()


def as_vector(values: Iterable[float], size: int, name: str = "b") -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size != size:
        raise NumericalMethodError(f"{name} должна быть вектором длины {size}")
    if not np.all(np.isfinite(vector)):
        raise NumericalMethodError(f"{name} содержит некорректные значения")
    return vector.copy()


def vector_infinity_norm(vector: np.ndarray) -> float:
    if vector.size == 0:
        return 0.0
    return max(abs(value) for value in vector)


def matrix_row_sum_norm(matrix: np.ndarray) -> float:
    return max(sum(abs(float(value)) for value in row) for row in matrix)


def max_abs_entry(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    return max(abs(float(value)) for value in matrix.flat)


def format_number(value: complex | float, precision: int = 6) -> str:
    value = complex(value)
    threshold = 10 ** (-precision)
    real = 0.0 if abs(value.real) < threshold else value.real
    imaginary = 0.0 if abs(value.imag) < threshold else value.imag
    if imaginary == 0.0:
        return f"{real:.{precision}g}"
    sign = "+" if imaginary > 0 else "-"
    return f"{real:.{precision}g}{sign}{abs(imaginary):.{precision}g}i"


def format_vector(vector: np.ndarray, precision: int = 6) -> str:
    return "[" + ", ".join(format_number(value, precision) for value in vector) + "]"


def format_matrix(matrix: np.ndarray, precision: int = 6) -> str:
    rows = [[format_number(value, precision) for value in row] for row in matrix]
    widths = [max(len(rows[i][j]) for i in range(len(rows))) for j in range(len(rows[0]))]
    return "\n".join(
        "[ " + "  ".join(value.rjust(widths[j]) for j, value in enumerate(row)) + " ]"
        for row in rows
    )


def print_named_matrix(name: str, matrix: np.ndarray, precision: int = 6) -> None:
    print(f"{name} =")
    print(format_matrix(matrix, precision))


def print_named_vector(name: str, vector: np.ndarray, precision: int = 6) -> None:
    print(f"{name} = {format_vector(vector, precision)}")
