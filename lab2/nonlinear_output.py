from pathlib import Path
import numpy as np
from chm.common import DEFAULT_MAX_ITERATIONS, format_vector
from chm.io_utils import read_json, require
from .nonlinear_equation import (
    EQUATION_Q, equation, equation_newton, equation_simple_iterations,
)
from .nonlinear_system import (
    SYSTEM_Q, system_function, system_newton, system_simple_iterations,
)


def reference_errors(history, reference: float | np.ndarray) -> list[float]:
    return [float(np.max(np.abs(value - reference))) for value in history]


def print_iterations(name, result, reference, function, precision):
    solution, history, estimates = result
    errors = reference_errors(history, reference)
    print(f"\n{name}:")
    print(" k | x^(k) | шаг / оценка | ошибка к уточнённому корню")
    for k, value in enumerate(history):
        estimate = '-' if k == 0 else f'{estimates[k - 1]:.{precision}g}'
        print(f"{k:2} | {format_vector(np.atleast_1d(value), precision)} | {estimate} | {errors[k]:.{precision}g}")
    print(f"Итерации: {len(estimates)}")
    print(f"Невязка: {float(np.max(np.abs(function(solution)))):.{precision}g}")


def plot_initial_approximation(axis, lab):
    if lab == '2.1':
        x = np.linspace(0.5, 1.2, 400)
        axis.plot(x, np.log1p(x) - 2 * x * x + 1, label='f(x)')
        axis.axhline(0, color='black', linewidth=0.8)
        axis.plot(1, equation(1), 'o', label='x⁽⁰⁾ = 1')
        axis.set(xlabel='x', ylabel='f(x)')
    else:
        x = np.linspace(2.0, 2.8, 400)
        axis.plot(x, np.sqrt(9 - x * x), label='x₁² + x₂² = 9')
        axis.plot(x, np.log(x + 3), label='x₂ = ln(x₁ + 3)')
        axis.plot(2.5, 1.7, 'o', label='x⁽⁰⁾ = (2.5, 1.7)')
        axis.set(xlabel='x₁', ylabel='x₂')
    axis.set_title('Выбор начального приближения')
    axis.legend(fontsize=9)


def save_plot(lab, results, reference):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    plot_initial_approximation(axes[0], lab)
    for name, result in results:
        _, history, estimates = result
        errors = reference_errors(history, reference)
        # Нулевую ошибку нельзя изобразить на логарифмической шкале.
        axes[1].semilogy(range(len(errors)), [e if e > 0 else np.nan for e in errors], 'o-', label=name)
        axes[1].semilogy(range(1, len(history)), estimates, '--', label=f'{name}: шаг/оценка')
    axes[1].set(title='Погрешность по итерациям', xlabel='k', ylabel='Погрешность (норма ∞)')
    axes[1].minorticks_off()
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle(f'Лабораторная {lab}, вариант 8')
    output = Path('lab2') / 'output' / f'lab{lab.replace(".", "_")}.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def run_nonlinear(path: str | Path, lab: str, precision: int = 6) -> None:
    data = read_json(path)
    epsilon = require(data, 'epsilon')
    max_iterations = data.get('max_iterations', DEFAULT_MAX_ITERATIONS)
    if lab == '2.1':
        simple, newton = equation_simple_iterations, equation_newton
        function, q = equation, EQUATION_Q
        description = 'ln(x+1) - 2x² + 1 = 0; x⁽⁰⁾ = 1; [0.9, 1]'
    else:
        simple, newton = system_simple_iterations, system_newton
        function, q = system_function, SYSTEM_Q
        description = 'x₁²+x₂²-9 = 0; x₁-exp(x₂)+3 = 0; x⁽⁰⁾ = (2.5, 1.7)'
    results = (
        ('Простые итерации', simple(epsilon, max_iterations)),
        ('Ньютон', newton(epsilon, max_iterations)),
    )
    reference, _, _ = newton(min(epsilon, 1e-14))
    print(f'Лабораторная {lab}, вариант 8\n{description}')
    print(f'epsilon = {epsilon:g}; q = {q:.{precision}g}')
    print('Простые итерации: оценка = q/(1-q) · ||x^(k)-x^(k-1)||∞.')
    print('Ньютон: шаг = ||x^(k)-x^(k-1)||∞; это критерий остановки, не гарантированная оценка.')
    print('Ошибка в таблице — сравнение с уточнённым решением Ньютона (epsilon ≤ 1e-14).')
    for name, result in results:
        print_iterations(name, result, reference, function, precision)
    print(f'\nГрафики: {save_plot(lab, results, reference).resolve()}')
