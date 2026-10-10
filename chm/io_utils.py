import json
from pathlib import Path
from typing import Any

from .common import NumericalMethodError


def read_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise NumericalMethodError(f"Файл входных данных не найден: {file_path}")
    try:
        with file_path.open("r", encoding="utf-8") as source:
            data = json.load(source)
    except json.JSONDecodeError as error:
        raise NumericalMethodError(
            f"Некорректный JSON в {file_path}: строка {error.lineno}, столбец {error.colno}"
        ) from error
    if not isinstance(data, dict):
        raise NumericalMethodError("Корневой элемент JSON должен быть объектом")
    return data


def require(data: dict[str, Any], field: str) -> Any:
    if field not in data:
        raise NumericalMethodError(f"Во входном файле отсутствует поле '{field}'")
    return data[field]

