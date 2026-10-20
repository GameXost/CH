from dataclasses import dataclass
import numpy as np

from .common import (
    DEFAULT_ZERO_TOLERANCE,
    ConvergenceError,
    NumericalMethodError,
    as_square_matrix,
    as_vector,
    matrix_row_sum_norm,
    vector_infinity_norm,
)


@dataclass(frozen=True)
class EquivalentSystem:
    alpha: np.ndarray
    beta: np.ndarray
    alpha_norm: float


@dataclass(frozen=True)
class IterativeResult:
    solution: np.ndarray
    iterations: int
    error_estimate: float
    guaranteed_estimate: bool


def build_equivalent_system(matrix: np.ndarray, rhs: np.ndarray) -> EquivalentSystem:
    A = as_square_matrix(matrix)
    b = as_vector(rhs, A.shape[0])

    n = A.shape[0]
    alpha = np.zeros((n, n), dtype=float)
    beta = np.zeros(n, dtype=float)
    for i in range(n):
        if abs(A[i, i]) <= DEFAULT_ZERO_TOLERANCE:
            raise NumericalMethodError(f"Нулевой диагональный элемент в строке {i + 1}")
        beta[i] = b[i] / A[i, i]
        for j in range(n):
            if i != j:
                alpha[i, j] = -A[i, j] / A[i, i]

    alpha_norm = matrix_row_sum_norm(alpha)
    if not np.isfinite(alpha_norm):
        raise NumericalMethodError("Норма alpha слишком велика для вычислений")
    return EquivalentSystem(alpha, beta, alpha_norm)


def _iterate(
    system: EquivalentSystem,
    epsilon: float,
    max_iterations: int,
    seidel: bool,
) -> IterativeResult:
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise NumericalMethodError("epsilon должна быть положительным конечным числом")
    if max_iterations <= 0:
        raise NumericalMethodError("max_iterations должно быть положительным")

    alpha, beta = system.alpha, system.beta
    q = system.alpha_norm
    guaranteed = q < 1.0

    numerator = matrix_row_sum_norm(np.triu(alpha)) if seidel else q
    factor = numerator / (1.0 - q) if guaranteed else 1.0
    x_previous = beta.copy()
    method = "Метод Зейделя" if seidel else "Метод простых итераций"

    for iteration in range(1, max_iterations + 1):
        if seidel:
            x_current = x_previous.copy()
            for i in range(beta.size):
                lower_sum = sum(alpha[i, j] * x_current[j] for j in range(i))
                upper_sum = sum(alpha[i, j] * x_previous[j] for j in range(i + 1, beta.size))
                x_current[i] = beta[i] + lower_sum + upper_sum
        else:
            x_current = alpha @ x_previous + beta

        difference = vector_infinity_norm(x_current - x_previous)
        error_estimate = factor * difference

        if not np.all(np.isfinite(x_current)) or not np.isfinite(error_estimate):
            raise ConvergenceError(f"{method}: некорректные числа на итерации {iteration}")
        if error_estimate <= epsilon:
            return IterativeResult(x_current, iteration, float(error_estimate), guaranteed)
        x_previous = x_current

    raise ConvergenceError(f"{method} не достиг epsilon={epsilon:g} за {max_iterations} итераций")


def simple_iterations(
    system: EquivalentSystem,
    epsilon: float,
    max_iterations: int = 10000,
) -> IterativeResult:
    return _iterate(system, epsilon, max_iterations, seidel=False)


def seidel_iterations(
    system: EquivalentSystem,
    epsilon: float,
    max_iterations: int = 10000,
) -> IterativeResult:
    return _iterate(system, epsilon, max_iterations, seidel=True)
