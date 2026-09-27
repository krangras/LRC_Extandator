@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" -Install
set "rc=%ERRORLEVEL%"
if not "%rc%"=="0" (
  echo.
  echo Installation failed with code %rc%.
  pause
  exit /b %rc%
)
echo.
echo LRC Extandator installed successfully.
pause
