@echo off
REM Agent-NEE FinAI — Start Script
REM Usage: start.bat

cd /d "%~dp0"

REM 1. Install dependencies
pip install -r requirements.txt 2>nul

REM 2. Check Ollama
where ollama >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Ollama not installed. Install from https://ollama.com
    pause
    exit /b 1
)

REM 3. Create runtime directories
if not exist logs mkdir logs
if not exist data\daily mkdir data\daily
if not exist data\replay_buffer mkdir data\replay_buffer
if not exist models\adapters mkdir models\adapters
if not exist charts mkdir charts

REM 4. Run
echo.
echo ============================================
echo   Agent-NEE FinAI — Starting
echo   Dashboard: http://localhost:8080/web/index.html
echo ============================================
echo.
python main.py
