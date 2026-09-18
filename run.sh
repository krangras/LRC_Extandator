#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Подключаем nvm, если node установлен через него (нужен для track-dl)
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
[[ -s "$NVM_DIR/nvm.sh" ]] && . "$NVM_DIR/nvm.sh"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="$VENV_DIR/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    printf 'Окружение не найдено. Сначала выполни ./install.sh\n' >&2
    exit 1
fi

# yt-dlp из venv должен быть виден дочерним процессам (track-dl)
export PATH="$ROOT/$VENV_DIR/bin:$PATH"

# CUDA-рунтайм, поставляемый pip-пакетами nvidia-*, должен быть виден ctranslate2/faster-whisper
for libdir in "$ROOT/$VENV_DIR/lib/python"*/site-packages/nvidia/{cublas,cuda_runtime}/lib; do
    if [ -d "$libdir" ]; then
        export LD_LIBRARY_PATH="${libdir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done

exec "$PYTHON" app.py
