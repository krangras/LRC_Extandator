#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    printf 'Python 3 не найден. Установи Python 3.10–3.12 и повтори запуск.\n' >&2
    exit 1
fi

if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)'; then
    printf 'Нужен Python 3.10, 3.11 или 3.12. Текущая версия: '
    "$PYTHON" --version
    exit 1
fi

VENV_DIR="${VENV_DIR:-.venv}"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -r requirements.txt

export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
[[ -s "$NVM_DIR/nvm.sh" ]] && . "$NVM_DIR/nvm.sh"
if command -v npm >/dev/null 2>&1; then
    npm install
    "$VENV_DIR/bin/python" -c 'import yt_dlp' 2>/dev/null || "$VENV_DIR/bin/python" -m pip install yt-dlp
    node scripts/patch_trackdl.js
else
    printf '\nПредупреждение: Node.js/npm не найдены. Скачивание треков работать не будет.\n' >&2
    printf 'Установи Node.js (например, через nvm: https://github.com/nvm-sh/nvm) и выполни: npm install\n' >&2
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    printf '\nПредупреждение: ffmpeg не найден в PATH. Он нужен для обработки аудио.\n' >&2
fi

printf '\nПроверка окружения py-roller...\n'
"$VENV_DIR/bin/py-roller" doctor

printf '\nУстановка завершена. Запуск: ./run.sh\n'
