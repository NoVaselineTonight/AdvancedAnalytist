@echo off
rem Starts the AdvancedAnalytist desktop app (no console window).
rem /D: the app folder is the working directory, so "-m replay" runs THIS folder's code (the venv's editable
rem install may point elsewhere) even from a shortcut or another folder.
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
  echo AdvancedAnalytist: "%~dp0.venv\Scripts\pythonw.exe" was not found. Reinstall the virtual environment.
  pause
  exit /b 1
)
start "" /D "%~dp0." "%~dp0.venv\Scripts\pythonw.exe" -m replay gui
