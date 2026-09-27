@echo off
setlocal
cd /d "%~dp0"
python -m unittest discover -s tests -v
if errorlevel 1 (
  echo.
  echo TESTS FAILED
  pause
  exit /b 1
)
echo.
echo ALL TESTS PASSED
pause
