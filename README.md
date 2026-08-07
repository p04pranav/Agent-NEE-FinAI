<![CDATA[<div align="center">

# Agent-NEE FinAI

### Agentic Supervised Fine-Tuning — Neural Execution Engine for Financial Analytics

---

**A local, AI-powered stock price prediction system for the Indian equity market (NSE) using multi-agent LLM inference, two-phase Parquet ledger, LoRA SFT post-market training, and a real-time terminal-themed web dashboard.**

---

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge&logo=ollama&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

---

</div>

## What is Agent-NEE?

Agent-NEE is a **fully local** AI stock prediction system built for the **Indian Stock Market (NSE)**. It runs entirely on your machine — no cloud APIs, no data leaks, no subscription fees.

The system uses a **multi-agent approach** where three specialist AI agents (Technical Analyst, Volatility Analyst, Volume Analyst) independently analyze market data, then a Synthesizer merges their outputs into a single prediction:

```
Direction: UP / DOWN / SIDEWAYS
Target Return: +1.2%
Confidence: HIGH / MED / LOW
```

---

## Key Highlights

| Feature | Description |
|---------|-------------|
| **Multi-Agent Prediction Squad** | 3 specialist agents + synthesizer running on local Ollama LLM |
| **Two-Phase Parquet Ledger** | UUID-based prediction tracking with 5-minute resolution |
| **LoRA SFT Post-Market Training** | Automatic fine-tuning on correct predictions after market close |
| **Real-Time Web Dashboard** | Terminal-themed (green-on-black) with live candlestick charts |
| **Hardware Auto-Detection** | Runs on CPU (3 tickers) or GPU (10 tickers) with zero config |
| **Built-in Mock Data Fallback** | Works without Dhan API credentials for development and testing |
| **100% Local Execution** | Data never leaves your machine. No cloud, no API keys, no billing |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     AGENT-NEE ORCHESTRATOR                           │
│                   main.py (event loop + SIGINT)                      │
│                   Mode: training_enabled = auto_detect()             │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
         ┌──────────────────────┴──────────────────────┐
         ▼                                             ▼
┌─────────────────────┐                   ┌─────────────────────────┐
│  MARKET HOURS GATE  │                   │  POST-MARKET TRIGGER    │
│  09:25-15:15 IST    │                   │  15:30 IST → learn.py   │
│  Mon-Fri only       │                   │  (if CUDA 12.1+ & 12GB) │
└─────────┬───────────┘                   └────────────┬────────────┘
          ▼                                            │
┌─────────────────────┐                                ▼
│  DHAN API / MOCK    │                   ┌─────────────────────────┐
│  dhan_client.py     │                   │  LoRA SFT TRAINING      │
│  10 NSE tickers     │                   │  transformers + peft    │
└─────────┬───────────┘                   │  Mixed Replay Buffer    │
          ▼                               │  Validation Rollback    │
┌─────────────────────┐                   └─────────────────────────┘
│  INDICATORS (pandas-ta)
│  VWAP, RSI, MACD, BB, ATR
└─────────┬───────────┘
          ▼
┌─────────────────────────────────────┐
│  PREDICTION SQUAD                   │
│  ┌───────────────────────────────┐  │
│  │  Technical Analyst            │  │
│  │  Volatility Analyst           │  │
│  │  Volume Analyst               │  │
│  │  Synthesizer → JSON output    │  │
│  └───────────────────────────────┘  │
│  LocalBandSDK room coordination    │
└─────────┬───────────────────────────┘
          ▼
┌─────────────────────┐    ┌─────────────────────────┐
│  TWO-PHASE LEDGER   │    │  WEB DASHBOARD          │
│  Parquet + UUID     │    │  FastAPI + WebSocket    │
│  Phase 1 → Phase 2  │    │  Terminal Dark Theme    │
└─────────────────────┘    └─────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.10+ |
| Local LLM | Ollama (phi3:mini / llama3.2:8b / llama3.3:8b) | latest |
| Inference | Ollama REST API + GBNF JSON enforcement | built-in |
| Data Source | Dhan API (dhanhq SDK) + mock fallback | 1.1.0 |
| Indicators | pandas-ta | 0.3.14b0 |
| Storage | Parquet (pyarrow) | 14.0.0+ |
| Web Server | FastAPI + Uvicorn | 0.115.0+ / 0.32.0+ |
| Browser Charts | Chart.js + Lightweight Charts (TradingView) | CDN |
| Training | transformers + peft LoRA SFT | 4.44.0+ / 0.12.0+ |
| Timezone | zoneinfo + holidays | 0.40+ |

---

## Device Compatibility

| Device | Inference | Dashboard | Training |
|--------|-----------|-----------|----------|
| CPU only | Yes | Yes | No |
| NVIDIA T4 16GB | Yes | Yes | Yes |
| NVIDIA RTX 3060 12GB | Yes | Yes | Yes |
| NVIDIA RTX 4090 24GB | Yes | Yes | Yes |
| NVIDIA RTX 2060 6GB | Yes | Yes | No |
| AMD RX 7900 XTX | Yes | Yes | No |
| Apple M1/M2/M3 | Yes | Yes | No |
| Intel Arc A770 | Yes | Yes | No |
| WSL2 + T4/3060 | Yes | Yes | Yes |

---

## Target Tickers

All 10 are high-liquidity NSE large caps:

| # | Ticker | Sector | Base Price (Mock) |
|---|--------|--------|-------------------|
| 1 | NSE:RELIANCE | Energy / Conglomerate | 2,845 |
| 2 | NSE:TCS | IT Services | 3,920 |
| 3 | NSE:HDFCBANK | Banking | 1,650 |
| 4 | NSE:INFY | IT Services | 1,480 |
| 5 | NSE:ICICIBANK | Banking | 1,120 |
| 6 | NSE:SBIN | Banking (Public) | 780 |
| 7 | NSE:BHARTIARTL | Telecom | 1,250 |
| 8 | NSE:ITC | FMCG / Conglomerate | 480 |
| 9 | NSE:WIPRO | IT Services | 510 |
| 10 | NSE:AXISBANK | Banking | 1,080 |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Ollama installed and running

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/p04pranav/Agent-NEE-FinAI.git
cd Agent-NEE-FinAI

# 2. Install Ollama (if not installed)
curl -fsSL https://ollama.com/install.sh | sh

# 3. Pull the model
ollama pull phi3:mini

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. (Optional) Set Dhan API credentials
cp .env.example .env
# Edit .env with your Dhan credentials
# If skipped, the system uses mock data automatically

# 6. Run the system
python main.py
```

The web dashboard auto-opens at `http://localhost:8080/web/index.html`

---

## Project Structure

```
Agent-NEE/
├── main.py                  # Entry point: hardware probe, event loop
├── config.py                # All configuration constants
├── agents.py                # LocalBandSDK + agent roles + synthesis prompt
├── dhan_client.py           # Dhan API wrapper + mock data fallback
├── market.py                # IST market hours gate
├── indicators.py            # pandas-ta technical indicators
├── predict.py               # PredictionSquad: multi-agent orchestration
├── ledger.py                # Two-phase Parquet ledger + replay buffer
├── learn.py                 # LoRA SFT training (subprocess)
├── charts.py                # matplotlib saved reports
├── dashboard.py             # DashboardState thread-safe data container
├── web_server.py            # FastAPI + WebSocket broadcast
├── utils.py                 # Logging, exceptions, retry decorators
├── web/                     # Frontend (terminal theme)
│   ├── index.html
│   ├── css/styles.css
│   └── js/
│       ├── app.js
│       ├── charts.js
│       └── candlestick.js
├── requirements.txt
├── setup.sh / setup.bat
├── .env.example
├── .gitignore
│
│── PRD.md                   # Product Requirements Document
│── TRD.md                   # Technical Requirements Document
│── UI_UX_Design.md          # UI/UX Design Specifications
│── Appflow.md               # Application Flow
│── Backend_Schema.md        # Backend & Database Schema
└── Implementation_Plan.md   # Step-by-Step Execution Roadmap
```

---

## Performance Targets

| Operation | Target | Critical |
|-----------|--------|----------|
| Full cycle (GPU, 10 tickers) | < 20s | < 60s |
| Full cycle (CPU, 3 tickers) | < 15s | < 45s |
| Single agent inference (GPU) | < 500ms | < 2s |
| Single agent inference (CPU) | < 3s | < 10s |
| Dashboard update | < 1ms | < 5ms |
| WebSocket broadcast | < 10ms | < 50ms |
| Prediction accuracy | 62%+ | > 50% |

---

## How It Works

### During Market Hours (09:25 - 15:15 IST)

1. **Every 5 seconds**, the system checks if a new prediction cycle is due (every 5 minutes)
2. **Polls market data** from Dhan API (or generates mock data)
3. **Computes technical indicators**: VWAP, RSI(14), MACD, Bollinger Bands, ATR(14)
4. **Runs 3 specialist agents** sequentially on local Ollama LLM:
   - Technical Analyst: price action, VWAP, RSI, MACD, BB, ATR
   - Volatility Analyst: ATR, BB width, volatility regime
   - Volume Analyst: volume trends, VWAP, conviction
5. **Synthesizer merges** all 3 analyses into a single JSON prediction
6. **Writes to Parquet ledger** (Phase 1) with UUID tracking
7. **5 minutes later**: Phase 2 resolves prediction against actual market outcome
8. **Broadcasts to web dashboard** via WebSocket at 1 FPS

### After Market Close (15:30 IST)

1. Generates matplotlib charts (accuracy trend, pred vs actual, latency)
2. If GPU available (CUDA 12.1+ & 12 GB+ VRAM):
   - Loads replay buffer (30-day rolling + 30% static baseline)
   - Balances UP/DOWN/SIDEWAYS classes
   - Fine-tunes model with LoRA SFT on correct predictions
   - Validates and rolls back if loss increases
   - Saves adapter checkpoint

---

## Dashboard Preview

```
┌───────────────────────────────────────────────────────────────────────────┐
│  [MODE: INFERENCE ONLY] [UPTIME: 2h 34m] [ACC: 62.3%] [TKRS: 3/10]     │
├─────────────────────────────────┬────────────────────────────────────────┤
│                                 │  PREDICTIONS                           │
│  RELIANCE                       │  RELIANCE  [UP]   +1.24%  ████████    │
│  ▲ 2,855.00  +12.50 (0.44%)    │  TCS       [DN]   -0.52%  ████        │
│  ┌───────────────────────────┐ │  HDFCBANK  [UP]   +0.81%  ██████      │
│  │    ╱╲   ╱╲   ╱╲          │ │  INFY      [--]    0.00%  ██          │
│  │   ╱  ╲ ╱  ╲ ╱  ╲         │ │  ICICIBANK [UP]   +1.13%  ███████     │
│  │  ╱    ╲╱    ╲╱    ╲        │ ├────────────────────────────────────────┤
│  │  ██  ██  ██  ██  ██  ██   │ │  AGENT ACTIVITY — RELIANCE             │
│  │       VOLUME HISTOGRAM     │ │  [TECH] Tech Analyst  0.8s             │
│  └───────────────────────────┘ │  [VOL]  Vol Analyst   0.7s             │
│  ◄ RELIANCE TCS HDFCBANK ►    │  [SYN]  Synthesizer   0.5s             │
├─────────────────────────────────┴────────────────────────────────────────┤
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐         │
│  │ Accuracy Trend   │ │ Pred vs Actual   │ │ Latency Trend    │         │
│  │ ~~~~~~~~~~~~     │ │  *  *    *       │ │ ~~~~~~~~~~~~     │         │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘         │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| [PRD.md](PRD.md) | Product Requirements — vision, features, functional requirements, personas |
| [TRD.md](TRD.md) | Technical Requirements — tech stack, config, memory, error handling, performance |
| [UI_UX_Design.md](UI_UX_Design.md) | UI/UX Design — terminal theme, components, WebSocket format, Chart.js specs |
| [Appflow.md](Appflow.md) | Application Flow — startup, event loop, market gate, state management, edge cases |
| [Backend_Schema.md](Backend_Schema.md) | Backend Schema — Parquet schema, agents, API contracts, training data formation |
| [Implementation_Plan.md](Implementation_Plan.md) | Implementation Plan — 5-phase roadmap, testing, latency benchmarks |

---

## Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Inference Engine | Ollama REST API | Zero compilation, any OS, trivial install |
| Base Model | phi3:mini (configurable) | 2.2 GB, runs on CPU. Swap to LLaMA 3.2/3.3 8B for GPU |
| JSON Enforcement | Ollama GBNF (built-in) | Token-level constraint, zero dependencies |
| Training Approach | LoRA SFT (not GRPO) | Always converges, battle-tested, ~90% first-attempt success |
| Agent System | 3 + synthesizer | Technical + Volatility + Volume. Cross-ticker removed for latency |
| Storage | Parquet with UUID | Type-safe, efficient, exact Phase 1→Phase 2 matching |
| Dashboard | FastAPI + WebSocket | Browser-accessible, real-time, no terminal dependency |

---

## License

MIT License — See [LICENSE](LICENSE) for details.

---

## Author

**Pranav S** — [GitHub](https://github.com/p04pranav)

---

<div align="center">

**Built with local AI for the Indian stock market**

![Made with Python](https://img.shields.io/badge/Made_with-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Powered by Ollama](https://img.shields.io/badge/Powered_by-Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)
![NSE India](https://img.shields.io/badge/Market-NSE_India-FF6600?style=for-the-badge)

</div>
]]>