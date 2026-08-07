# Agent-NEE — Product Requirements Document (PRD)

**Full Name**: Agentic Supervised Fine-Tuning — Neural Execution Engine for Financial Analytics
**Version**: 1.0
**Date**: 2024-10-24
**Author**: Pranav S
**Market**: Indian Stock Market (NSE — National Stock Exchange)

---

## 1. Project Vision

Agent-NEE is a local, AI-powered stock price prediction system designed specifically for the **Indian equity market (NSE)**. It leverages a local Large Language Model (LLM) via **Ollama** to generate short-term price predictions using a multi-agent analytical approach. The system operates entirely on the user's machine — no cloud APIs, no data leaks, no subscription fees.

**Core Philosophy**: Bring institutional-grade AI analytics to individual Indian stock traders through a fully local, privacy-first, hardware-adaptive system that runs on anything from a laptop CPU to a high-end GPU workstation.

---

## 2. Problem Statement

### The Gap
Individual NSE traders lack access to AI-driven analytics that are:
- **Affordable**: Cloud AI APIs (OpenAI, Anthropic) incur per-token costs that scale with usage
- **Private**: Sending portfolio data to third-party servers raises confidentiality concerns
- **Localized**: Most AI tools are built for US markets; Indian market nuances (IST timing, NSE tickers, holiday calendars) are ignored
- **Accessible**: Existing algo-trading tools require programming expertise or expensive terminals

### The Solution
Agent-NEE addresses all four gaps by providing:
1. **Zero-cost inference** via local Ollama (no API keys, no per-token billing)
2. **100% local execution** — data never leaves the user's machine
3. **NSE-native design** — Indian holidays, IST time zones, NSE ticker format built-in
4. **Web dashboard** — browser-based UI, no terminal or coding skills required

---

## 3. Target Audience

### Persona 1: The Day Trader
- **Profile**: Active NSE trader, 5-20 trades per day, monitors 5-10 large-cap stocks
- **Need**: Quick directional signals (UP/DOWN/SIDEWAYS) with confidence levels
- **Value**: Real-time predictions every 5 minutes during market hours, visual dashboard
- **Technical Level**: Non-programmer, comfortable with browser-based tools

### Persona 2: The Quant Hobbyist
- **Profile**: Finance enthusiast with Python knowledge, experiments with algo-trading
- **Need**: Customizable prediction pipeline, training on historical data, performance metrics
- **Value**: Full source code, LoRA fine-tuning on their own prediction accuracy, replay buffer
- **Technical Level**: Intermediate programmer, can run Python scripts and modify configs

### Persona 3: The System Builder
- **Profile**: Developer building trading tools or integrating AI into existing workflows
- **Need**: Clean API, modular architecture, extensible agent system
- **Value**: REST/WebSocket APIs, well-documented modules, pluggable agent roles
- **Technical Level**: Advanced programmer, comfortable with system architecture

---

## 4. Core Features

### Feature 1: Multi-Agent Prediction Squad
- **What**: Three specialist AI agents (Technical Analyst, Volatility Analyst, Volume Analyst) each analyze market data independently, then a Synthesizer merges their outputs into a single prediction
- **Why**: Diverse analytical perspectives reduce single-model bias; sequential execution on shared Ollama instance keeps resource usage minimal
- **Output**: `{direction: "UP|DOWN|SIDEWAYS", target_return_pct: float, confidence: "LOW|MED|HIGH"}`

### Feature 2: Two-Phase Ledger (Parquet-based State Management)
- **What**: Phase 1 logs predictions in real-time with UUID tracking. Phase 2 resolves predictions against actual market outcomes ~5 minutes later
- **Why**: Enables accuracy tracking, training data generation, and post-hoc analysis without blocking the prediction loop
- **Storage**: Per-ticker Parquet files with 20+ columns per row

### Feature 3: Web Dashboard with Live Charts
- **What**: Browser-based terminal-themed dashboard served via FastAPI + WebSocket at `localhost:8080`
- **Why**: Accessible from any browser, real-time updates at 1 FPS, no terminal dependency
- **Panels**: Live candlestick chart, prediction cards, agent activity, accuracy/latency/scatter charts

### Feature 4: LoRA SFT Post-Market Training
- **What**: After market close, the system automatically fine-tunes the model using correct predictions from the replay buffer (supervised fine-tuning, not reinforcement learning)
- **Why**: Model improves daily on its own correct predictions; SFT is more stable and reliable than GRPO
- **Hardware**: Auto-detected — only runs on CUDA 12.1+ with 12 GB+ VRAM

### Feature 5: Built-in Mock Data Fallback
- **What**: If Dhan API credentials are missing or the API is unreachable, the system generates realistic OHLCV mock data for all 10 tickers
- **Why**: Enables development, testing, and demos without live market data or API keys
- **Behavior**: Transparent fallback with logging; dashboard shows "DATA: MOCK" indicator

### Feature 6: Hardware Auto-Detection & Adaptive Scaling
- **What**: At startup, the system detects GPU availability, VRAM capacity, and CUDA version. It automatically enables/disables training and scales the number of active tickers
- **Why**: Zero configuration — the same codebase runs on a MacBook, a cloud VM, or a GPU workstation
- **Scaling**: CPU → 3 tickers, GPU → 10 tickers; training disabled below 12 GB VRAM

---

## 5. Functional Requirements

| ID | Requirement | Source Rule |
|----|-------------|-------------|
| FR-01 | Training dependencies (torch, transformers, peft) must NEVER be imported in `main.py`. They are imported only inside `learn.py` which runs as a subprocess | Rule 1 |
| FR-02 | All structured LLM output must use Ollama's built-in GBNF enforcement (`format: "json"` or explicit `grammar` parameter). No external JSON enforcement libraries | Rule 2 |
| FR-03 | All 3 agents and the synthesizer must run sequentially on the same Ollama instance. No parallel model loading | Rule 3 |
| FR-04 | Before the prediction loop starts, the system must verify Ollama is running and the model is available | Rule 4 |
| FR-05 | If CUDA OOM occurs during training, the system must auto-reduce batch size and retry | Rule 5 |
| FR-06 | Every prediction row must have a UUID `row_id`. Phase 2 resolution must match on UUID, never sequential IDs or timestamps alone | Rule 6 |
| FR-07 | Ticker symbols must be defined only in `config.py` and imported everywhere else. No hardcoded tickers in other modules | Rule 7 |
| FR-08 | Predictions must only run during Indian market hours: 09:25-15:15 IST (Mon-Thu), 09:25-15:00 IST (Fri), no weekends or Indian public holidays | Rule 8 |
| FR-09 | Dashboard updates must be non-blocking — fast dict writes with a Lock. WebSocket broadcast runs in a separate async task. Prediction loop never blocks | Rule 9 |
| FR-10 | Every Phase 1 row must include `agent_contributions` (JSON blob) and `band_room_id` columns for traceability | Rule 10 |
| FR-11 | The system must work without Dhan API credentials by falling back to `generate_mock_data()` | Rule 11 |
| FR-12 | All logs go to `logs/bifas_nexus.log` with daily rotation. Every prediction cycle logs ticker, direction, return%, confidence, latency_ms | Rule 12 |
| FR-13 | Every agent call must be completely stateless — no `context` array passed between sequential Ollama calls. No `keep_alive: 0` | Rule 13 |
| FR-14 | On startup, the system must detect hardware mode and cap active tickers: CPU → 3, GPU → 10 | Rule 14 |
| FR-15 | WebSocket dashboard must never block the prediction loop. All WebSocket I/O runs in asyncio on a separate daemon thread | Rule 15 |

---

## 6. Two-Mode Architecture

### Mode 1: Universal (Any Device)
- Runs on any hardware with Python 3.10+
- Full inference, ledger, dashboard, charts
- Ollama handles CPU/GPU automatically
- 0 GB VRAM required (CPU mode) to 4 GB (GPU offload)

### Mode 2: Training (CUDA 12.1+ & 12 GB+ VRAM)
- Everything in Mode 1, plus:
- Automatic LoRA fine-tuning after market close
- LoRA adapter save/load with versioning
- Validation rollback if loss increases
- Requires `torch`, `transformers`, `peft`, `datasets`, `accelerate`

### Mode Detection
Auto-detected at startup via `auto_detect_training()`. Zero manual configuration. The user just runs `main.py`.

---

## 7. Device Compatibility Matrix

| Device | Inference | Dashboard | Training |
|--------|-----------|-----------|----------|
| CPU only (no GPU) | Yes (Ollama CPU) | Yes | No |
| NVIDIA T4 16 GB | Yes (Ollama GPU) | Yes | Yes |
| NVIDIA RTX 3060 12 GB | Yes | Yes | Yes |
| NVIDIA RTX 4090 24 GB | Yes | Yes | Yes |
| NVIDIA RTX 2060 6 GB | Yes | Yes | No (VRAM < 12 GB) |
| AMD RX 7900 XTX | Yes (Ollama ROCm) | Yes | No (ROCm unsupported) |
| Apple M1/M2/M3 | Yes (Ollama Metal) | Yes | No (unsupported platform) |
| Intel Arc A770 | Yes (Ollama) | Yes | No (no CUDA) |
| WSL2 Windows + T4 | Yes | Yes | Yes |
| WSL2 Windows + 3060 | Yes | Yes | Yes |

---

## 8. Target Tickers

All 10 are high-liquidity NSE large caps:

| # | Ticker | Sector |
|---|--------|--------|
| 1 | NSE:RELIANCE | Energy / Conglomerate |
| 2 | NSE:TCS | IT Services |
| 3 | NSE:HDFCBANK | Banking |
| 4 | NSE:INFY | IT Services |
| 5 | NSE:ICICIBANK | Banking |
| 6 | NSE:SBIN | Banking (Public) |
| 7 | NSE:BHARTIARTL | Telecom |
| 8 | NSE:ITC | FMCG / Conglomerate |
| 9 | NSE:WIPRO | IT Services |
| 10 | NSE:AXISBANK | Banking |

---

## 9. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Inference Engine** | Ollama REST API | Zero compilation, works on any OS, trivial install (`curl`) |
| **Base Model** | Configurable (default: `phi3:mini`) | Phi-3-mini: 2.2 GB, runs anywhere. Swap to LLaMA 3.2/3.3 8B for higher quality |
| **JSON Enforcement** | Ollama GBNF (built-in) | Token-level constraint, zero dependencies |
| **Data Source** | Dhan API + mock fallback | Broker-grade data when available, offline testing when not |
| **Training Approach** | LoRA SFT (not GRPO) | Always converges, battle-tested APIs, ~90% first-attempt success vs ~40% for GRPO |
| **Agent System** | 3 agents + synthesizer | Technical + Volatility + Volume perspectives. Cross-ticker agent removed for latency |
| **Coordination** | LocalBandSDK (in-memory) | Per-agent tracking, dashboard integration, ledger traceability |
| **Dashboard** | Web (FastAPI + WebSocket) | Browser-accessible, real-time, no terminal dependency |
| **Storage** | Parquet with UUID matching | Type-safe, efficient, exact Phase 1→Phase 2 resolution |

---

## 10. Success Metrics

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| Prediction accuracy (direction) | 62%+ | > 50% (better than random) |
| Full cycle latency (GPU, 10 tickers) | < 20s | < 60s (must fit in 5-min window) |
| Full cycle latency (CPU, 3 tickers) | < 15s | < 45s |
| Single agent inference (GPU) | < 500ms | < 2s |
| Single agent inference (CPU) | < 3s | < 10s |
| DashboardState update | < 1ms | < 5ms |
| WebSocket broadcast | < 10ms | < 50ms |
| Dhan API poll (10 tickers) | < 500ms | < 2s |
| Training convergence | 90%+ first attempt | 70%+ |
| System uptime | Full trading day (6.5h) | No crashes |

---

## 11. Scope & Boundaries

### In Scope
- NSE large-cap equities (10 tickers)
- 5-minute prediction intervals during market hours
- Directional prediction (UP/DOWN/SIDEWAYS) with return % and confidence
- Post-market LoRA fine-tuning on correct predictions
- Web-based dashboard with real-time charts
- Mock data fallback for offline development

### Out of Scope
- Options, futures, or derivatives trading
- Cryptocurrency or forex markets
- Real-time tick-by-tick prediction
- Automated order execution (no broker integration for trades)
- Multi-exchange support (BSE not included)
- Portfolio optimization or risk management
- Cloud deployment or SaaS model

---

## 12. Constraints & Assumptions

### Constraints
- Python 3.10+ required
- Ollama must be installed and running (`ollama serve`)
- Model must be pre-pulled (`ollama pull phi3:mini`)
- Dhan API requires valid credentials (optional — mock fallback available)
- Training requires NVIDIA GPU with CUDA 12.1+ and 12 GB+ VRAM

### Assumptions
- User has basic command-line ability to run `python main.py`
- Market hours follow NSE schedule (09:15-15:30 IST, Mon-Fri)
- 5-minute prediction window is sufficient for the target audience
- Local hardware can run Ollama with at least Phi-3-mini (4 GB RAM minimum)
