@echo off
cd /d "%~dp0"
title LitSync
echo.
echo  Starting LitSync...
echo  Keep this window open while you use the website.
echo  Open http://localhost:8000 in your browser.
echo.
python run_litsync.py
if errorlevel 1 (
  echo.
  echo LitSync could not start. Make sure Python and the project dependencies are installed.
  pause
)
