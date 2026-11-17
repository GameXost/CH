from pathlib import Path
import math
import numpy as np
from chm.common import NumericalMethodError
from chm.io_utils import read_json, require
from .common import finite_number, make_function, polynomial_value, validate_nodes
from .interpolation import lagrange_value, divided_differences, newton_value
from .spline import natural_spline, spline_interval, spline_value, spline_derivative
from .least_squares import least_squares
from .differentiation import differentiate
from .integration import METHODS, integrate, interval_count, runge_romberg


def print_nodes(x, y, precision):
    print(' i | x_i | y_i')
    for i, (xi, yi) in enumerate(zip(x, y)):
        print(f'{i:2} | {xi:.{precision}g} | {yi:.{precision}g}')


def polynomial_text(coefficients, precision, factors=None):
    if factors is None:
        factors = [''] + ['x' if i == 1 else f'x^{i}' for i in range(1, len(coefficients))]
    result = ''
    for coefficient, factor in zip(coefficients, factors):
        if coefficient == 0:
            continue
        sign = '-' if coefficient < 0 else '+'
        term = f'{abs(coefficient):.{precision}g}' + (f'*{factor}' if factor else '')
        result += (sign if not result else f' {sign} ') + term
    return result.lstrip('+') or '0'


def factor_text(x, precision):
    if x == 0:
        return 'x'
    return f'(x {"+" if x < 0 else "-"} {abs(x):.{precision}g})'


def save_plot(name, x, y, curves, point=None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 6), layout='constrained')
    for label, grid, values, style in curves:
        axis.plot(grid, values, style, label=label)
    axis.plot(x, y, 'ko', label='Исходные узлы')
    if point is not None:
        axis.plot(*point, 'r*', markersize=12, label=f'x* = {point[0]:g}')
    axis.set(title=name, xlabel='x', ylabel='y')
    axis.grid(True, alpha=0.3)
    axis.legend()
    output = Path(__file__).resolve().parent / 'output' / f'{name}.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f'График: {output}')


def run_interpolation(data, precision):
    function = make_function(require(data, 'function'))
    point = finite_number(require(data, 'x_star'), 'x*')
    exact = function(point)
    print(f'f(x) = {data["function"]}; x* = {point:g}')
    node_sets = require(data, 'node_sets')
    if not isinstance(node_sets, list) or not node_sets:
        raise NumericalMethodError('node_sets должен содержать наборы узлов')
    for number, nodes in enumerate(node_sets, 1):
        x, _ = validate_nodes(require(nodes, 'x'), nodes['x'])
        y = [function(t) for t in x]
        table = divided_differences(x, y)
        print(f'\nНабор {nodes.get("name", number)}:')
        print_nodes(x, y, precision)

        coefficients, factors = [], []
        for i in range(len(x)):
            coefficients.append(y[i] / math.prod(x[i] - x[j] for j in range(len(x)) if j != i))
            factors.append('*'.join(factor_text(x[j], precision) for j in range(len(x)) if j != i))
        print('L(x) = ' + polynomial_text(coefficients, precision, factors))
        print('\nРазделённые разности: x_i | f[x_i] | порядки 1, 2, ...')
        for i in range(len(x)):
            row = [f'{table[i, k]:.{precision}g}' if i + k < len(x) else '-' for k in range(len(x))]
            print(f'{x[i]:.{precision}g} | ' + ' | '.join(row))
        factors = ['*'.join(factor_text(x[j], precision) for j in range(k)) for k in range(len(x))]
        print('N(x) = ' + polynomial_text(table[0], precision, factors))

        left = lagrange_value(x, y, point)
        right = newton_value(x, table, point)
        print(f'L(x*) = {left:.{precision}g}; N(x*) = {right:.{precision}g}; f(x*) = {exact:.{precision}g}')
        print(f'|L-f| = {abs(left-exact):.{precision}g}; |N-f| = {abs(right-exact):.{precision}g}')
        print(f'L-N = {left-right:.{precision}g}; |L-N| = {abs(left-right):.{precision}g}')
        grid = np.linspace(min(x[0], point), max(x[-1], point), 400)
        curves = [('f(x)', grid, [function(t) for t in grid], 'k-'),
                  ('Лагранж', grid, [lagrange_value(x, y, t) for t in grid], 'b-'),
                  ('Ньютон', grid, [newton_value(x, table, t) for t in grid], 'r--')]
        save_plot(f'lab3_1_{number}', x, y, curves, (point, left))


def run_spline(data, precision):
    x, y = validate_nodes(require(data, 'x'), require(data, 'y'))
    point = finite_number(require(data, 'x_star'), 'x*')
    coefficients = natural_spline(x, y)
    interval = spline_interval(x, point)
    print_nodes(x, y, precision)
    print('\nS_i(x) = a_i + b_i*t + c_i*t² + d_i*t³, t = x - левая граница участка.')
    print(' i | интервал | a_i | b_i | c_i | d_i')
    curves = []
    for i, section in enumerate(coefficients):
        print(f'{i+1:2} | [{x[i]:g}; {x[i+1]:g}] | ' + ' | '.join(f'{c:.{precision}g}' for c in section))
        grid = np.linspace(x[i], x[i+1], 100)
        curves.append((f'Участок {i+1}', grid, [polynomial_value(section, t-x[i]) for t in grid], '-'))
    value = spline_value(x, coefficients, point)
    print(f'\nx* = {point:g}; интервал № {interval+1}: [{x[interval]:g}; {x[interval+1]:g}]')
    print(f'S(x*) = {value:.{precision}g}')
    print('\nПроверка узлов: x_i | y_i | S(x_i) | |S(x_i)-y_i|')
    for xi, yi in zip(x, y):
        actual = spline_value(x, coefficients, xi)
        print(f'{xi:g} | {yi:.{precision}g} | {actual:.{precision}g} | {abs(actual-yi):.{precision}g}')
    print(f"S''({x[0]:g}) = {spline_derivative(coefficients, 0, 0, 2):.{precision}g}")
    print(f"S''({x[-1]:g}) = {spline_derivative(coefficients, len(x)-2, x[-1]-x[-2], 2):.{precision}g}")
    save_plot('lab3_2', x, y, curves, (point, value))


def run_least_squares(data, precision):
    x, y = validate_nodes(require(data, 'x'), require(data, 'y'), min_count=4, ordered=False)
    print_nodes(x, y, precision)
    print('\nКоэффициенты от a_0 до a_m; MSE = sum((P(x_i)-y_i)^2)/N.')
    grid = np.linspace(min(x), max(x), 400)
    errors, curves = [], []
    for degree in (1, 2, 3):
        coefficients, mse = least_squares(x, y, degree)
        errors.append(mse)
        print(f'\nСтепень {degree}: ' + ', '.join(f'a_{i} = {a:.{precision}g}' for i, a in enumerate(coefficients)))
        print(f'P{degree}(x) = {polynomial_text(coefficients, precision)}')
        print(f'MSE{degree} = {mse:.{precision}g}')
        curves.append((f'P{degree}(x)', grid, [polynomial_value(coefficients, t) for t in grid], ('-', '--', '-.')[degree-1]))
    best = min(errors)
    winners = [f'P{i+1}' for i, error in enumerate(errors) if math.isclose(error, best, rel_tol=1e-10, abs_tol=1e-24)]
    print('\nЛучшая MSE по исходным точкам: ' + ', '.join(winners))
    save_plot('lab3_3', x, y, curves)


def run_differentiation(data, precision):
    x, y = validate_nodes(require(data, 'x'), require(data, 'y'), min_count=3)
    point = finite_number(require(data, 'x_star'), 'x*')
    first, second, indices = differentiate(x, y, point)
    print_nodes(x, y, precision)
    print(f'\nx* = {point:g}; использованные индексы: {indices}')
    print('Использованные узлы: ' + ', '.join(f'({x[i]:g}; {y[i]:.{precision}g})' for i in indices))
    print(f"f'(x*) ≈ {first:.{precision}g}")
    print(f"f''(x*) ≈ {second:.{precision}g}")


def run_integration(data, precision):
    a, b, h1, h2 = [finite_number(require(data, key), key) for key in ('a', 'b', 'h1', 'h2')]
    if not h1 > h2 > 0:
        raise NumericalMethodError('Требуются h1 > h2 > 0')
    if any(interval_count(a, b, h) % 2 for h in (h1, h2)):
        raise NumericalMethodError('Для Симпсона обе сетки должны иметь чётное число отрезков')
    function = make_function(require(data, 'function'))
    exact = None
    print(f'f(x) = {data["function"]}; [{a:g}; {b:g}]; h1 = {h1:g}; h2 = {h2:g}')
    if 'antiderivative' in data:
        primitive = make_function(data['antiderivative'])
        exact = primitive(b) - primitive(a)
        print(f'F(x) = {data["antiderivative"]}; точный интеграл = {exact:.{precision}g}')
    print('I_RR = I(h2) + (I(h2)-I(h1))/((h1/h2)^p-1); оценка E(h2) = |I_RR-I(h2)|.')
    header = 'Метод | I(h1) | I(h2) | p | I_RR | оценка E(h2)'
    print(header + (' | ошибка h1 | ошибка h2 | ошибка RR' if exact is not None else ''))
    errors = []
    for method, name, order in METHODS:
        coarse = integrate(function, a, b, h1, method)
        fine = integrate(function, a, b, h2, method)
        refined, estimate = runge_romberg(coarse, fine, h1, h2, order)
        row = [coarse, fine, order, refined, estimate]
        if exact is not None:
            row += [abs(coarse-exact), abs(fine-exact), abs(refined-exact)]
            errors.append((name, abs(fine-exact), abs(refined-exact)))
        print(f'{name:23} | ' + ' | '.join(f'{value:.{precision}g}' for value in row))
    if errors:
        for column, label in ((1, 'при h2'), (2, 'после Рунге–Ромберга')):
            best = min(row[column] for row in errors)
            winners = [row[0] for row in errors if math.isclose(row[column], best, rel_tol=1e-10, abs_tol=1e-15)]
            print(f'Наиболее точный {label}: ' + ', '.join(winners))


def run_lab3(path: str | Path, lab: str, precision: int = 6) -> None:
    data = read_json(path)
    print(f'Лабораторная {lab}' + (f', вариант {data["variant"]}' if 'variant' in data else ''))
    if lab == '3.1':
        run_interpolation(data, precision)
    elif lab == '3.2':
        run_spline(data, precision)
    elif lab == '3.3':
        run_least_squares(data, precision)
    elif lab == '3.4':
        run_differentiation(data, precision)
    else:
        run_integration(data, precision)
