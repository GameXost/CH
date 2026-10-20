import cmath
import math
from dataclasses import dataclass
import numpy as np
from .common import ConvergenceError, NumericalMethodError, as_square_matrix


@dataclass(frozen=True)
class QRDecomposition:
    Q: np.ndarray
    R: np.ndarray


@dataclass(frozen=True)
class QREigenResult:
    eigenvalues: np.ndarray
    schur_form: np.ndarray
    iterations: int


def qr_decomposition(matrix: np.ndarray) -> QRDecomposition:
    R = as_square_matrix(matrix)
    n = R.shape[0]
    Q = np.eye(n)

    for column in range(n - 1):
        v = R[column:, column].copy()
        scale = max(abs(v))
        if scale == 0:
            continue

        v /= scale
        column_norm = math.hypot(*v)
        v[0] += math.copysign(column_norm, v[0])
        reflection_norm = math.hypot(*v)
        v /= reflection_norm

        H = np.eye(n)
        H[column:, column:] -= 2.0 * np.outer(v, v)
        R = H @ R
        Q = Q @ H
        R[column + 1:, column] = 0.0

    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(Q)):
        raise ConvergenceError("QR-разложение: некорректные числа при вычислениях")
    return QRDecomposition(Q=Q, R=R)


def _block_eigenvalues(block: np.ndarray) -> tuple[complex, complex]:
    scale = float(np.max(np.abs(block)))
    if scale == 0:
        return 0j, 0j
    a, b = block[0] / scale
    c, d = block[1] / scale
    half_trace = (a + d) / 2.0
    radicand = ((a - d) / 2.0) ** 2 + b * c
    root = cmath.sqrt(complex(radicand))
    return scale * (half_trace + root), scale * (half_trace - root)


def _converged_eigenvalues(
    matrix: np.ndarray,
    previous: np.ndarray | None,
    epsilon: float,
) -> np.ndarray | None:
    values = []
    n = matrix.shape[0]
    column = 0

    while column < n:
        below_diagonal = matrix[column + 1:, column]
        if math.hypot(*below_diagonal) <= epsilon:
            values.append(complex(matrix[column, column]))
            column += 1
            continue

        below_block = matrix[column + 2:, column:column + 2]
        if math.hypot(*below_block.flat) > epsilon:
            return None

        block = matrix[column:column + 2, column:column + 2]
        pair = _block_eigenvalues(block)
        if pair[0].imag == 0 or previous is None:
            return None

        previous_block = previous[column:column + 2, column:column + 2]
        previous_pair = _block_eigenvalues(previous_block)
        eigenvalue_change = max(
            abs(value - old) for value, old in zip(pair, previous_pair)
        )
        if eigenvalue_change > epsilon:
            return None
        values.extend(pair)
        column += 2

    return np.asarray(values, dtype=complex)


def qr_eigenvalues(
    matrix: np.ndarray,
    epsilon: float,
    max_iterations: int = 10_000,
) -> QREigenResult:
    A_current = as_square_matrix(matrix)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise NumericalMethodError("epsilon должна быть положительным конечным числом")
    if max_iterations <= 0:
        raise NumericalMethodError("max_iterations должно быть положительным")

    previous = None
    for iteration in range(max_iterations + 1):
        if not np.all(np.isfinite(A_current)):
            raise ConvergenceError("QR-алгоритм: некорректные числа при вычислениях")
        eigenvalues = _converged_eigenvalues(A_current, previous, epsilon)
        if eigenvalues is not None:
            if not np.all(np.isfinite(eigenvalues)):
                raise ConvergenceError("QR-алгоритм: некорректные собственные значения")
            return QREigenResult(
                eigenvalues=eigenvalues,
                schur_form=A_current,
                iterations=iteration,
            )
        if iteration == max_iterations:
            break

        decomposition = qr_decomposition(A_current)
        previous = A_current
        A_current = decomposition.R @ decomposition.Q

    raise ConvergenceError(
        f"QR-алгоритм не достиг epsilon={epsilon:g} за {max_iterations} итераций"
    )
