@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_nsr2.ps1"
if errorlevel 1 (
  echo.
  echo NSR2 startup failed. Check .codex-run logs.
  pause
)
