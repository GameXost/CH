import ast
import math
import numpy as np

from chm.common import NumericalMethodError, as_vector


def finite_number(value, name="значение"):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise NumericalMethodError(f"{name}: требуется конечное число") from error
    if not math.isfinite(value):
        raise NumericalMethodError(f"{name}: требуется конечное число")
    return value


def validate_nodes(x, y, min_count=2, ordered=True):
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as error:
        raise NumericalMethodError("Узлы и значения должны быть числами") from error
    if x.ndim != 1 or x.size < min_count:
        raise NumericalMethodError(f"Нужно не менее {min_count} узлов")
    x, y = as_vector(x, x.size, "x"), as_vector(y, x.size, "y")
    if ordered and any(x[i] >= x[i + 1] for i in range(len(x) - 1)):
        raise NumericalMethodError("Узлы x должны строго возрастать")
    return x, y


def polynomial_value(coefficients, point):
    value = 0.0
    for coefficient in reversed(coefficients):
        value = value * point + coefficient
    return value


def make_function(expression):
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 1000:
        raise NumericalMethodError("Нужна непустая формула длиной не более 1000 символов")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError) as error:
        raise NumericalMethodError("Некорректная формула функции") from error
    functions = {name: getattr(math, name) for name in
                 ("sin", "cos", "tan", "asin", "acos", "atan", "exp", "log", "sqrt")}
    functions["abs"] = abs
    names = {**functions, "pi": math.pi, "e": math.e}
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
               ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise NumericalMethodError("В формуле разрешены только числа, x и математические операции")
        if isinstance(node, ast.Name) and node.id not in names and node.id != "x":
            raise NumericalMethodError(f"Неизвестное имя в формуле: {node.id}")
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise NumericalMethodError("В формуле разрешены только действительные числа")
            node.value = finite_number(node.value, "Число в формуле")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name)
                or node.func.id not in functions or len(node.args) != 1 or node.keywords):
            raise NumericalMethodError("Математическая функция должна иметь один аргумент")
    code = compile(tree, "<формула>", "eval")

    def function(x):
        try:
            value = eval(code, {"__builtins__": {}, **names}, {"x": finite_number(x, "x")})
            return finite_number(value, "f(x)")
        except (ArithmeticError, ValueError, TypeError) as error:
            raise NumericalMethodError(f"Нельзя вычислить {expression} при x={x}") from error

    return function
