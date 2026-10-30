import math
from chm.common import DEFAULT_MAX_ITERATIONS
from .nonlinear_common import iterate


def equation(x: float) -> float:
    return math.log1p(x) - 2 * x * x + 1


def equation_derivative(x: float) -> float:
    return 1 / (1 + x) - 4 * x


def equation_nn(x: float) -> float:
    return math.sqrt((math.log1p(x) + 1) / 2)


EQUATION_Q = 1 / (4 * 1.9 * equation_nn(0.9))


def equation_simple_iterations(
    epsilon: float, max_iterations: int = DEFAULT_MAX_ITERATIONS
) -> tuple:
    return iterate(
        equation_nn, 1.0, epsilon, max_iterations, EQUATION_Q / (1 - EQUATION_Q)
    )


def equation_newton_step(x: float) -> float:
    return x - equation(x) / equation_derivative(x)


def equation_newton(epsilon: float, max_iterations: int = DEFAULT_MAX_ITERATIONS) -> tuple:
    return iterate(equation_newton_step, 1.0, epsilon, max_iterations)
