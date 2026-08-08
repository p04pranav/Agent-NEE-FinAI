#!/bin/bash
# Agent-NEE FinAI — Start Script
# Usage: ./start.sh
set -e

cd "$(dirname "$0")"

# 1. Install dependencies if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# 2. Check Ollama
if ! command -v ollama &>/dev/null; then
    echo "ERROR: Ollama not installed. Install from https://ollama.com"
    exit 1
fi

if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 3
fi

if ! ollama list 2>/dev/null | grep -q "phi3:mini"; then
    echo "Pulling phi3:mini model (~2.2 GB)..."
    ollama pull phi3:mini
fi

# 3. Create runtime directories
mkdir -p logs data/daily data/replay_buffer models/adapters charts

# 4. Run
echo ""
echo "============================================"
echo "  Agent-NEE FinAI — Starting"
echo "  Dashboard: http://localhost:8080/web/index.html"
echo "============================================"
echo ""
python3 main.py
