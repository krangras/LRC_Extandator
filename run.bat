@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist "lrc_env\Scripts\python.exe" (
    echo Виртуальное окружение не найдено. Сначала запусти setup.bat
    pause
    exit /b 1
)
echo Запуск LRC Studio 5.0...
call "lrc_env\Scripts\activate.bat"
python app.py
pause
