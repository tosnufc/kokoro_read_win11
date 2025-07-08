# Kokoro TTS Quick Launch Script (PowerShell)
# Activates virtual environment and runs optimized TTS

Write-Host "🚀 Starting Kokoro TTS..." -ForegroundColor Green
Write-Host "📍 Location: $PSScriptRoot" -ForegroundColor Cyan

# Change to script directory
Set-Location $PSScriptRoot

# Check if virtual environment exists
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please make sure .venv folder exists in this directory." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
Write-Host "⚡ Activating virtual environment..." -ForegroundColor Yellow
& ".venv\Scripts\Activate.ps1"

# Check if Python script exists
if (-not (Test-Path "kokoro_readaloud_win11.py")) {
    Write-Host "❌ kokoro_readaloud_win11.py not found!" -ForegroundColor Red
    Write-Host "Please make sure the Python script is in this directory." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Run the TTS script
Write-Host "🎯 Running optimized TTS..." -ForegroundColor Green
Write-Host ""
python kokoro_readaloud_win11.py

# Keep window open for a moment to see results
Write-Host ""
Write-Host "✨ TTS completed!" -ForegroundColor Green
Start-Sleep -Seconds 3

# Deactivate virtual environment
deactivate 