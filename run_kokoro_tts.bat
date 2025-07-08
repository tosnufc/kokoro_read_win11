@echo off
REM Kokoro TTS Quick Launch Script (Minimized)
REM Activates virtual environment and runs optimized TTS

REM Check if running minimized, if not restart minimized
if not "%1"=="minimized" (
    start /min cmd /c "%~dpnx0" minimized
    exit /b
)

echo 🚀 Starting Kokoro TTS (Minimized)...
echo 📍 Location: %~dp0

REM Change to script directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found!
    echo Please make sure .venv folder exists in this directory.
    pause
    exit /b 1
)

REM Activate virtual environment
echo ⚡ Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if Python script exists
if not exist "kokoro_readaloud_win11.py" (
    echo ❌ kokoro_readaloud_win11.py not found!
    echo Please make sure the Python script is in this directory.
    pause
    exit /b 1
)

REM Run the TTS script
echo 🎯 Running optimized TTS...
echo.
python kokoro_readaloud_win11.py

REM Keep window open briefly to see results
echo.
echo ✨ TTS completed!
timeout /t 2 /nobreak >nul

REM Deactivate virtual environment
deactivate 