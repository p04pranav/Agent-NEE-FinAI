#!/bin/bash
# Agent-NEE FinAI — Setup Script (Linux/macOS)
set -e

echo "============================================"
echo "  Agent-NEE FinAI — Setup"
echo "============================================"

# 1. Install Python dependencies
echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt

# 2. Check Ollama installation
echo "[2/4] Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo "  Ollama found: $(ollama --version)"
else
    echo "  Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# 3. Pull model
echo "[3/4] Pulling phi3:mini model (~2.2 GB)..."
ollama pull phi3:mini

# 4. Create runtime directories
echo "[4/4] Creating runtime directories..."
mkdir -p logs data/daily data/replay_buffer models/adapters charts

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Run: python main.py"
echo "============================================"
