@echo off
cd /d "%~dp0"
title LitSync Screening Lab
echo.
echo  Starting LitSync Screening Lab...
echo  Keep this window open while you use the website.
echo.
python run_litsync.py
if errorlevel 1 (
  echo.
  echo LitSync could not start. Make sure Python and the project dependencies are installed.
  pause
)
