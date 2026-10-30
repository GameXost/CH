from numbers import Integral, Real
from sys import float_info
from typing import Callable
import numpy as np
from chm.common import ConvergenceError, NumericalMethodError


def validate_iteration_parameters(epsilon: float, max_iterations: int) -> None:
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise NumericalMethodError("epsilon должна быть числом")
    if not 0 < epsilon <= float_info.max:
        raise NumericalMethodError("epsilon должна быть положительным конечным числом")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral):
        raise NumericalMethodError("max_iterations должно быть целым числом")
    if max_iterations <= 0:
        raise NumericalMethodError("max_iterations должно быть положительным")


def iterate(
    step: Callable,
    initial: float | np.ndarray,
    epsilon: float,
    max_iterations: int,
    factor: float = 1.0,
) -> tuple:
    validate_iteration_parameters(epsilon, max_iterations)
    history = [initial]
    estimates = []
    for _ in range(max_iterations):
        current = step(history[-1])
        difference = float(np.max(np.abs(current - history[-1])))
        estimate = factor * difference
        if not np.all(np.isfinite(current)) or not np.isfinite(estimate):
            raise ConvergenceError("Получены некорректные числа при вычислении приближения")
        history.append(current)
        estimates.append(estimate)
        if estimate < epsilon:
            return current, tuple(history), tuple(estimates)
    raise ConvergenceError(f"Не достигнута точность epsilon={epsilon:g} за {max_iterations} итераций")
