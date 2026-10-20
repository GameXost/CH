import math
from dataclasses import dataclass
import numpy as np
from .common import ConvergenceError, NumericalMethodError, as_square_matrix


@dataclass(frozen=True)
class JacobiRotationResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    diagonal_matrix: np.ndarray
    iterations: int
    off_diagonal_norm: float


def _off_diagonal_norm(matrix: np.ndarray) -> float:
    total = 0.0
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[0]):
            total += matrix[i, j] ** 2
    return math.sqrt(total)


def _largest_off_diagonal(matrix: np.ndarray) -> tuple[int, int]:
    n = matrix.shape[0]
    largest_row, largest_column = 0, 1
    largest_value = abs(matrix[largest_row, largest_column])
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i, j]) > largest_value:
                largest_value = abs(matrix[i, j])
                largest_row, largest_column = i, j
    return largest_row, largest_column


def jacobi_rotation(
    matrix: np.ndarray,
    epsilon: float,
    max_iterations: int = 100_000,
) -> JacobiRotationResult:
    A_current = as_square_matrix(matrix)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise NumericalMethodError("epsilon должна быть положительным конечным числом")
    if max_iterations <= 0:
        raise NumericalMethodError("max_iterations должно быть положительным")

    n = A_current.shape[0]
    eigenvectors = np.eye(n, dtype=float)
    current_norm = _off_diagonal_norm(A_current)

    for iteration in range(max_iterations + 1):
        if not np.isfinite(current_norm) or not np.all(np.isfinite(A_current)):
            raise ConvergenceError("Метод вращений Якоби: некорректные числа при вычислениях")
        if current_norm < epsilon:
            return JacobiRotationResult(
                eigenvalues=np.diag(A_current).copy(),
                eigenvectors=eigenvectors,
                diagonal_matrix=A_current,
                iterations=iteration,
                off_diagonal_norm=current_norm,
            )
        if iteration == max_iterations:
            break

        i, j = _largest_off_diagonal(A_current)
        if A_current[i, i] == A_current[j, j]:
            phi = math.pi / 4.0
        else:
            phi = 0.5 * math.atan(
                2.0 * A_current[i, j] / (A_current[i, i] - A_current[j, j])
            )

        cosinus = math.cos(phi)
        sinus = math.sin(phi)
        U = np.eye(n, dtype=float)
        U[i, i] = cosinus
        U[j, j] = cosinus
        U[i, j] = -sinus
        U[j, i] = sinus

        A_current = U.T @ A_current @ U
        A_current = 0.5 * (A_current + A_current.T)
        eigenvectors = eigenvectors @ U
        current_norm = _off_diagonal_norm(A_current)

    raise ConvergenceError(
        f"Метод вращений Якоби не достиг epsilon={epsilon:g} за {max_iterations} вращений"
    )
