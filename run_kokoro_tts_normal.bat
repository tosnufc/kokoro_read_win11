@echo off
REM Kokoro TTS Quick Launch Script (Normal Window)
REM Activates virtual environment and runs optimized TTS

echo 🚀 Starting Kokoro TTS...
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

REM Keep window open to see results
echo.
echo ✨ TTS completed!
echo Press any key to close...
pause >nul

REM Deactivate virtual environment
deactivate 