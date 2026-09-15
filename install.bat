@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" -Install
set "rc=%ERRORLEVEL%"
if not "%rc%"=="0" (
    echo.
    echo ERROR: installation failed with code %rc%.
    pause
    exit /b %rc%
)
echo.
echo Installation finished. You can now run launcher.bat.
pause
exit /b 0
