@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "target=%~1"
if "%target%"=="" (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0APPLY_TO_REPO.ps1"
) else (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0APPLY_TO_REPO.ps1" -Target "%target%"
)
set "rc=%ERRORLEVEL%"
if not "%rc%"=="0" (
  echo.
  echo Patch failed with code %rc%.
  pause
  exit /b %rc%
)
echo.
pause
