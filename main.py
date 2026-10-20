import argparse
import sys
from pathlib import Path
import numpy as np
from chm.common import (
    NumericalMethodError,
    as_square_matrix,
    as_vector,
    format_number,
    max_abs_entry,
    print_named_matrix,
    print_named_vector,
    vector_infinity_norm,
)
from chm.io_utils import read_json, require
from chm.iterative import build_equivalent_system, seidel_iterations, simple_iterations
from chm.jacobi_rotation import jacobi_rotation
from chm.lu import (
    determinant_with_lu,
    inverse_with_lu,
    lu_decomposition,
    solve_with_lu,
)
from chm.qr import qr_decomposition, qr_eigenvalues
from chm.tridiagonal import sweep_method


def run_lu(path: str | Path, precision: int = 6) -> None:
    data = read_json(path)
    A = as_square_matrix(require(data, "matrix"))
    b = as_vector(require(data, "rhs"), A.shape[0])

    decomposition = lu_decomposition(A)
    solution = solve_with_lu(decomposition, b)
    inverse = inverse_with_lu(decomposition)
    determinant = determinant_with_lu(decomposition)

    LU = decomposition.L @ decomposition.U
    PA = decomposition.P @ A
    identity_check = A @ inverse

    print_named_matrix("A", A, precision)
    print_named_vector("b", b, precision)
    print_named_matrix("P", decomposition.P, precision)
    print_named_matrix("L", decomposition.L, precision)
    print_named_matrix("U", decomposition.U, precision)
    print_named_matrix("L · U", LU, precision)
    print_named_matrix("P · A", PA, precision)
    print_named_vector("x", solution, precision)
    print_named_matrix("A^(-1)", inverse, precision)
    print(f"det(A) = {format_number(determinant, precision)}")
    print_named_matrix("A · A^(-1)", identity_check, precision)

    print(f"Перестановки строк: {decomposition.permutation_count}")
    print(f"||L·U - P·A||max = {max_abs_entry(LU - PA):.{precision}g}")
    print(
        f"||A·A^(-1) - E||max = "
        f"{max_abs_entry(identity_check - np.eye(A.shape[0])):.{precision}g}"
    )
    print(f"||A·x - b||∞ = {vector_infinity_norm(A @ solution - b):.{precision}g}")


def run_tridiagonal(path: str | Path, precision: int = 6) -> None:
    data = read_json(path)
    P, Q, solution = sweep_method(
        require(data, "lower"),
        require(data, "main"),
        require(data, "upper"),
        require(data, "rhs"),
    )

    for i, (p_i, q_i) in enumerate(zip(P, Q), start=1):
        print(f"P_{i} = {format_number(p_i, precision)}, Q_{i} = {format_number(q_i, precision)}")
    print_named_vector("x", solution, precision)


def run_iterative(path: str | Path, precision: int = 6) -> None:
    data = read_json(path)
    A = as_square_matrix(require(data, "matrix"))
    b = as_vector(require(data, "rhs"), A.shape[0])
    epsilon = require(data, "epsilon")
    max_iterations = data.get("max_iterations", 10_000)

    equivalent = build_equivalent_system(A, b)
    simple = simple_iterations(equivalent, epsilon, max_iterations)
    seidel = seidel_iterations(equivalent, epsilon, max_iterations)
    reference = solve_with_lu(lu_decomposition(A), b)

    print(f"epsilon = {epsilon:g}")
    print_named_matrix("A", A, precision)
    print_named_vector("b", b, precision)
    print_named_matrix("alpha", equivalent.alpha, precision)
    print_named_vector("beta = x^(0)", equivalent.beta, precision)
    print(f"||alpha||c = {equivalent.alpha_norm:.{precision}g}")
    if not simple.guaranteed_estimate:
        print("||alpha||c >= 1: погрешность не гарантируется")

    print_named_vector("x_эталон (LU)", reference, precision)
    for name, result in (("Простые итерации", simple), ("Зейдель", seidel)):
        print(f"\n{name}:")
        print_named_vector("x", result.solution, precision)
        print(f"Итерации: {result.iterations}")
        estimate_name = "Оценка погрешности" if result.guaranteed_estimate else "Разность приближений"
        print(f"{estimate_name}: {result.error_estimate:.{precision}g}")
        print(f"||Ax-b||∞ = {vector_infinity_norm(A @ result.solution - b):.{precision}g}")
        print(f"||x-x_эталон||∞ = {vector_infinity_norm(result.solution - reference):.{precision}g}")

    if simple.iterations < seidel.iterations:
        print("Меньше итераций: простые итерации")
    elif seidel.iterations < simple.iterations:
        print("Меньше итераций: Зейдель")
    else:
        print("Итераций поровну")


def run_jacobi(path: str | Path, precision: int = 6) -> None:
    data = read_json(path)
    A = as_square_matrix(require(data, "matrix"))
    epsilon = require(data, "epsilon")
    max_iterations = data.get("max_iterations", 100_000)
    result = jacobi_rotation(A, epsilon, max_iterations)
    V = result.eigenvectors
    Lambda = np.diag(result.eigenvalues)
    left = A @ V
    right = V @ Lambda

    print(f"epsilon = {epsilon:g}")
    print_named_matrix("A", A, precision)
    print(f"Вращения: {result.iterations}")
    print(f"t(A^(k)) = {result.off_diagonal_norm:.{precision}g}")
    print_named_matrix("A^(k)", result.diagonal_matrix, precision)
    print_named_vector("lambda", result.eigenvalues, precision)
    for i in range(V.shape[1]):
        print_named_vector(f"v_{i + 1}", V[:, i], precision)
    print_named_matrix("V", V, precision)
    print_named_matrix("Lambda", Lambda, precision)
    print_named_matrix("A · V", left, precision)
    print_named_matrix("V · Lambda", right, precision)
    print(f"||A·V - V·Lambda||max = {max_abs_entry(left - right):.{precision}g}")
    print(f"||V^T·V - E||max = {max_abs_entry(V.T @ V - np.eye(A.shape[0])):.{precision}g}")


def run_qr(path: str | Path, precision: int = 6) -> None:
    data = read_json(path)
    A = as_square_matrix(require(data, "matrix"))
    epsilon = require(data, "epsilon")
    max_iterations = data.get("max_iterations", 10_000)
    decomposition = qr_decomposition(A)
    result = qr_eigenvalues(A, epsilon, max_iterations)
    Q, R = decomposition.Q, decomposition.R
    QR = Q @ R
    orthogonality_error = max_abs_entry(Q.T @ Q - np.eye(A.shape[0]))

    print(f"epsilon = {epsilon:g}")
    print_named_matrix("A", A, precision)
    print_named_matrix("Q", Q, precision)
    print_named_matrix("R", R, precision)
    print_named_matrix("Q · R", QR, precision)
    print(f"||Q·R - A||max = {max_abs_entry(QR - A):.{precision}g}")
    print(f"||Q^T·Q - E||max = {orthogonality_error:.{precision}g}")
    print(f"Итерации: {result.iterations}")
    print_named_vector("lambda", result.eigenvalues, precision)
    print_named_matrix("A^(k)", result.schur_form, precision)

    if data.get("numpy_check", False):
        print_named_vector("lambda (NumPy)", np.linalg.eigvals(A), precision)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Лабораторная работа 1: LU, прогонка, итерации, Зейдель, вращения Якоби и QR"
    )
    parser.add_argument("input_file", help="путь к JSON-файлу с входными данными")
    parser.add_argument(
        "--lab", choices=("1.1", "1.2", "1.3", "1.4", "1.5"), default="1.1", help="номер задания"
    )
    parser.add_argument("--precision", type=int, default=6, help="число значащих цифр")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        if arguments.lab == "1.2":
            run_tridiagonal(arguments.input_file, arguments.precision)
        elif arguments.lab == "1.3":
            run_iterative(arguments.input_file, arguments.precision)
        elif arguments.lab == "1.4":
            run_jacobi(arguments.input_file, arguments.precision)
        elif arguments.lab == "1.5":
            run_qr(arguments.input_file, arguments.precision)
        else:
            run_lu(arguments.input_file, arguments.precision)
    except (NumericalMethodError, OSError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
