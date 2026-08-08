@echo off
REM Agent-NEE FinAI — Setup Script (Windows)
echo ============================================
echo   Agent-NEE FinAI — Setup
echo ============================================

echo [1/4] Installing Python dependencies...
pip install -r requirements.txt

echo [2/4] Checking Ollama...
where ollama >nul 2>nul
if %errorlevel% neq 0 (
    echo Ollama not found. Please install from https://ollama.com
    echo Then run: ollama pull phi3:mini
    pause
    exit /b 1
)

echo [3/4] Pulling phi3:mini model...
ollama pull phi3:mini

echo [4/4] Creating runtime directories...
if not exist logs mkdir logs
if not exist data\daily mkdir data\daily
if not exist data\replay_buffer mkdir data\replay_buffer
if not exist models\adapters mkdir models\adapters
if not exist charts mkdir charts

echo.
echo ============================================
echo   Setup complete! Run: python main.py
echo ============================================
pause
