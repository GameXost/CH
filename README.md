# Лаба 1

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Пусковая кнопка:)

```powershell
.\.venv\Scripts\python.exe main.py data\lu.json --lab 1.1
.\.venv\Scripts\python.exe main.py data\tridiagonal.json --lab 1.2
.\.venv\Scripts\python.exe main.py data\iterative.json --lab 1.3
.\.venv\Scripts\python.exe main.py data\jacobi_rotation.json --lab 1.4
.\.venv\Scripts\python.exe main.py data\qr.json --lab 1.5
.\.venv\Scripts\python.exe main.py data\qr_complex_example.json --lab 1.5
```
Входные данные через json 