LRC Extandator V6 DROP-IN PATCH
===============================

1. Сделай копию своего репозитория LRC_Extandator.
2. Распакуй содержимое этого архива в КОРЕНЬ репозитория с заменой файлов.
3. Запусти install.bat.
4. После успешной установки запускай launcher.bat.

Это НЕ regex-patcher и не builder. Все изменённые файлы уже лежат готовыми в архиве.
Старые templates/index.html и static/script.js остаются на месте; V6 добавляет к ним свой UI enhancer.

Быстрая проверка без моделей:
    .venv\Scripts\python.exe -m unittest discover -s tests -v

Benchmark:
    .venv\Scripts\python.exe benchmark.py run benchmarks\manifest.json --out benchmark-results.json
