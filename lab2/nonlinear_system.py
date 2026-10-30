from math import exp, log, sqrt
import numpy as np
from chm.common import DEFAULT_MAX_ITERATIONS
from chm.lu import lu_decomposition, solve_with_lu
from .nonlinear_common import iterate


SYSTEM_Q = 0.75


def system_function(values: np.ndarray) -> np.ndarray:
    x, y = values
    return np.array([x * x + y * y - 9, x - exp(y) + 3])


def system_jacobian(values: np.ndarray) -> np.ndarray:
    x, y = values
    return np.array([[2 * x, 2 * y], [1, -exp(y)]])


def system_phi(values: np.ndarray) -> np.ndarray:
    x, y = values
    return np.array([sqrt(9 - y * y), log(x + 3)])


def system_simple_iterations(
    epsilon: float, max_iterations: int = DEFAULT_MAX_ITERATIONS
) -> tuple:
    return iterate(
        system_phi, np.array([2.5, 1.7]), epsilon, max_iterations,
        factor=SYSTEM_Q / (1 - SYSTEM_Q),
    )


def _newton_step(values: np.ndarray) -> np.ndarray:
    decomposition = lu_decomposition(system_jacobian(values))
    correction = solve_with_lu(decomposition, -system_function(values))
    return values + correction


def system_newton(epsilon: float, max_iterations: int = DEFAULT_MAX_ITERATIONS) -> tuple:
    return iterate(_newton_step, np.array([2.5, 1.7]), epsilon, max_iterations)
