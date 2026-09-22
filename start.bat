@echo off
cd /d "%~dp0"
title LitSync
echo.
echo  Starting LitSync...
echo  Keep this window open while you use the website.
echo  Open http://localhost:8000 in your browser.
echo.
set "LITSYNC_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "LITSYNC_PYTHON=.venv\Scripts\python.exe"
"%LITSYNC_PYTHON%" run_litsync.py
if errorlevel 1 (
  echo.
  echo LitSync could not start. Make sure Python and the project dependencies are installed.
  pause
)
