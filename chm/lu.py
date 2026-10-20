from dataclasses import dataclass
import numpy as np
from .common import (
    DEFAULT_ZERO_TOLERANCE,
    SingularMatrixError,
    as_square_matrix,
    as_vector,
)


@dataclass(frozen=True)
class LUDecomposition:
    L: np.ndarray
    U: np.ndarray
    P: np.ndarray
    permutation_count: int


def lu_decomposition(
    matrix: np.ndarray,
    zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
) -> LUDecomposition:
    U = as_square_matrix(matrix)
    n = U.shape[0]
    L = np.zeros((n, n), dtype=float)
    P = np.eye(n, dtype=float)
    permutation_count = 0

    for k in range(n):
        pivot_row = max(range(k, n), key=lambda row: abs(float(U[row, k])))
        if abs(U[pivot_row, k]) <= zero_tolerance:
            raise SingularMatrixError(
                f"На шаге {k + 1} в столбце {k + 1} отсутствует ненулевой главный элемент"
            )

        if pivot_row != k:
            U[[k, pivot_row], :] = U[[pivot_row, k], :]
            P[[k, pivot_row], :] = P[[pivot_row, k], :]
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
            permutation_count += 1

        L[k, k] = 1.0
        for row in range(k + 1, n):
            multiplier = U[row, k] / U[k, k]
            L[row, k] = multiplier
            U[row, k:] = U[row, k:] - multiplier * U[k, k:]
            U[row, k] = 0.0

    return LUDecomposition(L=L, U=U, P=P, permutation_count=permutation_count)

def forward_substitution(
    lower: np.ndarray,
    rhs: np.ndarray,
    zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
) -> np.ndarray:
    n = lower.shape[0]
    rhs = as_vector(rhs, n, "правая часть")
    result = np.zeros(n, dtype=float)
    for i in range(n):
        diagonal = lower[i, i]
        if abs(diagonal) <= zero_tolerance:
            raise SingularMatrixError(f"Нулевой диагональный элемент L в строке {i + 1}")
        known_sum = sum(lower[i, j] * result[j] for j in range(i))
        result[i] = (rhs[i] - known_sum) / diagonal
    return result

def backward_substitution(
    upper: np.ndarray,
    rhs: np.ndarray,
    zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
) -> np.ndarray:
    n = upper.shape[0]
    rhs = as_vector(rhs, n, "правая часть")
    result = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        diagonal = upper[i, i]
        if abs(diagonal) <= zero_tolerance:
            raise SingularMatrixError(f"Нулевой диагональный элемент U в строке {i + 1}")
        known_sum = sum(upper[i, j] * result[j] for j in range(i + 1, n))
        result[i] = (rhs[i] - known_sum) / diagonal
    return result


def solve_with_lu(decomposition: LUDecomposition, rhs: np.ndarray) -> np.ndarray:
    n = decomposition.L.shape[0]
    rhs = as_vector(rhs, n, "b")
    z = forward_substitution(decomposition.L, decomposition.P @ rhs)
    return backward_substitution(decomposition.U, z)


def inverse_with_lu(decomposition: LUDecomposition) -> np.ndarray:
    n = decomposition.L.shape[0]
    inverse = np.zeros((n, n), dtype=float)
    identity = np.eye(n, dtype=float)
    for column in range(n):
        inverse[:, column] = solve_with_lu(decomposition, identity[:, column])
    return inverse


def determinant_with_lu(decomposition: LUDecomposition) -> float:
    product = 1.0
    for i in range(decomposition.U.shape[0]):
        product *= decomposition.U[i, i]
    return (-1.0 if decomposition.permutation_count % 2 else 1.0) * product
