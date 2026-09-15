@echo off
chcp 65001 >nul
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Не найден Python Launcher. Установи Python 3.11 x64.
    pause
    exit /b 1
)

if not exist "lrc_env\Scripts\python.exe" py -3.11 -m venv lrc_env
if errorlevel 1 (
    echo Не удалось создать окружение на Python 3.11.
    pause
    exit /b 1
)

call "lrc_env\Scripts\activate.bat"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
if errorlevel 1 goto :failed

where npm >nul 2>nul
if not errorlevel 1 npm install

where ffmpeg >nul 2>nul
if errorlevel 1 echo ВНИМАНИЕ: ffmpeg не найден в PATH. Установи ffmpeg или положи его в ffmpeg\bin.

echo.
echo Установка завершена. Запускай run.bat
pause
exit /b 0

:failed
echo.
echo Установка Python-зависимостей завершилась с ошибкой.
pause
exit /b 1
