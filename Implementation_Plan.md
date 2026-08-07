# Agent-NEE — Implementation Plan

**Full Name**: Agentic Supervised Fine-Tuning — Neural Execution Engine for Financial Analytics
**Version**: 1.0
**Estimated Duration**: 14 days (5 phases)
**Execution Order**: Sequential — each phase builds on the previous

---

## Phase 0 — Infrastructure Validation (Day 1)

Validate core infrastructure before writing any code.

### Tasks

| Step | Task | Verification |
|------|------|-------------|
| 0.1 | Install Ollama: `curl -fsSL https://ollama.com/install.sh \| sh` | `ollama --version` returns version |
| 0.2 | Pull Phi-3-mini model: `ollama pull phi3:mini` | ~2.2 GB download completes |
| 0.3 | Test Ollama text API: `curl http://localhost:11434/api/generate -d '{"model":"phi3:mini","prompt":"Hello"}'` | Returns text response |
| 0.4 | Test Ollama JSON mode: `curl ... -d '{"model":"phi3:mini","prompt":"Return JSON","format":"json"}'` | Returns valid JSON |
| 0.5 | Test mock data deps: `python -c "import numpy, pandas; print('OK')"` | Prints "OK" |

### Acceptance Criteria
- [ ] Ollama installed and running
- [ ] phi3:mini model pulled and available
- [ ] Text generation works
- [ ] JSON mode returns valid JSON
- [ ] Python dependencies importable

---

## Phase 1 — Foundation & Data Pipeline (Days 2-4)

### Step 1.1: Project Scaffolding

| File | Deliverable |
|------|-------------|
| `requirements.txt` | Core deps only (11 packages). Training deps commented out |
| `setup.sh` | pip install core deps, check Ollama, pull model, create runtime dirs |
| `setup.bat` | Windows delegate to setup.sh |
| `.env.example` | `DHAN_CLIENT_ID=your_client_id`, `DHAN_ACCESS_TOKEN=your_access_token` |
| `.gitignore` | `__pycache__/`, `*.pyc`, `.env`, `models/adapters/*/`, `data/`, `charts/`, `logs/` |

### Step 1.2: Configuration & Utilities

| File | Deliverable |
|------|-------------|
| `config.py` | ALL constants: Ollama URL, model name, 10 ticker symbols, time windows, LoRA hyperparams, mock data config, paths. Key values: `N_CTX=4096`, `AGENT_MAX_TOKENS=512`, `SYNTH_MAX_TOKENS=64`, `CPU_TICKER_LIMIT=3`, `GPU_TICKER_LIMIT=10` |
| `utils.py` | `setup_logging()` (file + console, daily rotation), `retry_with_backoff()` decorator, 7 custom exceptions: `BIFASError`, `ConfigError`, `DhanAPIError`, `OllamaError`, `LedgerError`, `TrainingError`, `AgentError` |

### Step 1.3: Dhan API Client (with Mock Fallback)

| File | Deliverable |
|------|-------------|
| `dhan_client.py` | `fetch_security_map()` — resolve `NSE:RELIANCE` → Security ID. `generate_mock_data()` — realistic OHLCV for 10 tickers with base price map. `safe_poll_tickers()` — poll Dhan API OR fall back to mock. STALE sentinel on per-ticker failure. Exponential backoff on rate limit |

### Step 1.4: Market Gate

| File | Deliverable |
|------|-------------|
| `market.py` | `is_trading_day(dt)` — weekends + `holidays.INDIA`. `is_within_trading_hours(dt)` — 09:25-15:15 Mon-Thu, 09:25-15:00 Fri. Pre-market warmup at 09:15 |

### Step 1.5: Indicators Engine

| File | Deliverable |
|------|-------------|
| `indicators.py` | `compute_indicators(candles_df)` — VWAP (daily reset), RSI(14), MACD(12,26,9), BB(20,2), ATR(14), SMA(50/200). 5-day warmup enforcement. `format_market_data_for_agent(ticker, row)` — formatted string for LLM |

### Step 1.6: Ledger

| File | Deliverable |
|------|-------------|
| `ledger.py` | `phase1_write(row_data)` — Parquet with UUID + all 26 schema columns. `resolve_phase2(timestamp)` — UUID matching, actual return calc, direction classification. `update_replay_buffer()` — append + trim to 30 days. `get_replay_buffer()`, `get_completed_rows()`, `cleanup_orphans()`. Per-ticker files in `data/daily/` |

### Acceptance Criteria
- [ ] `requirements.txt` installs without errors
- [ ] `config.py` loads all constants correctly
- [ ] `utils.py` logging creates `logs/bifas_nexus.log`
- [ ] `dhan_client.py` generates mock data for 10 tickers
- [ ] `market.py` correctly identifies trading hours
- [ ] `indicators.py` computes all 7 indicators
- [ ] `ledger.py` writes and reads Parquet with UUID

---

## Phase 2 — Multi-Agent AI Inference (Days 5-8)

### Step 2.1: Ollama API Wrapper

| File | Deliverable |
|------|-------------|
| `predict.py` | `ollama_generate()` — POST to `/api/generate`, support `format: "json"`, `n_ctx` as `options.num_ctx`. **No context array between calls** (stateless). **No keep_alive: 0** (model warm). Timeout handling, retry on connection errors |

### Step 2.2: Ollama Startup Verification

| File | Deliverable |
|------|-------------|
| `main.py` | `verify_ollama()` — check `GET /api/tags`, verify phi3:mini available, clear error if not running |

### Step 2.3: Multi-Agent System

| File | Deliverable |
|------|-------------|
| `agents.py` | `LocalBandSDK` class (create_room, send_message, get_room_history, get_history_formatted). `band` singleton. `AGENT_ROLES` dict with 3 agents (exact prompts from Backend_Schema.md). `SYNTHESIS_PROMPT` with `{agent_analyses}` placeholder |

### Step 2.4: PredictionSquad

| File | Deliverable |
|------|-------------|
| `predict.py` | `PredictionSquad` class. Constructor takes `band`. `run(ticker_data, room_id)` — sequential agent execution + synthesizer via Ollama API. Synthesizer uses temperature 0.1. Returns `{direction, target_return_pct, confidence}` |

### Step 2.5: End-to-End Integration

| File | Deliverable |
|------|-------------|
| `main.py` | Connect fetch → indicators → PredictionSquad.run() → band room → phase1_write. `trim_tickers_for_hardware()` — caps to 3 on CPU, 10 on GPU. Agent calls use `AGENT_MAX_TOKENS=512`, synthesizer uses `SYNTH_MAX_TOKENS=64`. Dashboard shows active ticker count |

### Acceptance Criteria
- [ ] `ollama_generate()` returns text from Ollama
- [ ] JSON mode returns parseable JSON
- [ ] `LocalBandSDK` creates rooms and tracks messages
- [ ] All 3 agents produce distinct analytical outputs
- [ ] Synthesizer merges into `{direction, target_return_pct, confidence}`
- [ ] Full pipeline works with mock data (no Dhan API needed)
- [ ] CPU mode limits to 3 tickers, GPU mode runs all 10

---

## Phase 3 — Web Dashboard & Interactive Charts (Days 8-9)

### Step 3.1: Web Server

| File | Deliverable |
|------|-------------|
| `web_server.py` | FastAPI app with `@app.websocket("/ws")`, `ConnectionManager` class, `broadcast_loop()` at 1 FPS, REST endpoints `/api/status` and `/api/config`, static file mount `/web` |

### Step 3.2: Dashboard State

| File | Deliverable |
|------|-------------|
| `dashboard.py` | Refactored `DashboardState` dataclass: 5 attributes (market_data, predictions, agent_activity, latency, status), thread-safe Lock, `update(**kwargs)`, `get_snapshot()` deep copy |

### Step 3.3: Frontend — Terminal Theme & Layout

| File | Deliverable |
|------|-------------|
| `web/index.html` | Terminal layout: candlestick (left 60%), prediction + agent panels (right 40%), AI charts (full width), status bar (top). Script tags for Chart.js CDN, Lightweight Charts CDN |
| `web/css/styles.css` | Terminal green-on-black: `#0a0a0a` bg, `#00ff41` primary, JetBrains Mono font, no border-radius, 1px solid green borders, green glow on headings, red/green price colors, amber for warnings, CSS Grid layout |

### Step 3.4: Frontend — JavaScript

| File | Deliverable |
|------|-------------|
| `web/js/app.js` | WebSocket client with auto-reconnect, `renderPredictionPanel()`, `renderAgentPanel()`, `renderStatusBar()`, ticker click handler, routes data to candlestick.js and charts.js |
| `web/js/candlestick.js` | `initCandlestick()`, `createCandlestickSeries()` (green-up/red-down), `createVolumeSeries()`, `updateCandles()` (100-candle buffer), `switchTicker()`, `updatePriceHeader()`. Terminal colors |
| `web/js/charts.js` | Chart.js global defaults (terminal theme), `initAccuracyChart()` (line), `initScatterChart()` (scatter), `initLatencyChart()` (multi-line), `updateCharts()` |

### Step 3.5: Integration in main.py

| File | Deliverable |
|------|-------------|
| `main.py` | Launch Uvicorn in daemon thread, auto-open browser, wire prediction loop to DashboardState |

### Step 3.6: Charts (matplotlib — saved reports)

| File | Deliverable |
|------|-------------|
| `charts.py` | Keep matplotlib for post-market static PNG reports in `charts/`. Separate from live Chart.js |

### Acceptance Criteria
- [ ] WebSocket connects and receives JSON broadcasts
- [ ] Candlestick chart renders and updates with OHLCV data
- [ ] Ticker switching updates candlestick view
- [ ] Prediction panel shows [UP]/[DN]/[--] badges
- [ ] Agent panel shows collapsible cards with [TECH]/[VOL]/[SYN]
- [ ] Chart.js and Lightweight Charts coexist without conflicts
- [ ] REST API endpoints return correct data
- [ ] Browser auto-opens on startup
- [ ] Multiple browser tabs receive same broadcast

---

## Phase 4 — LoRA SFT Training (Days 10-12)

### Step 4.1: Mode Detection

| File | Deliverable |
|------|-------------|
| `main.py` | `auto_detect_training()` — try import torch, check CUDA, VRAM >= 12 GB, CUDA >= 12.1. Return boolean, log result |

### Step 4.2: Training Module

| File | Deliverable |
|------|-------------|
| `learn.py` | Standalone script (subprocess). Load replay buffer → balance classes → format prompt-completion pairs → train/val split → load `microsoft/Phi-3-mini-4k-instruct` → apply LoRA (r=16, alpha=16, target q_proj/v_proj) → tokenize → configure Trainer (lr=2e-5, epochs=3, batch=2, grad_accum=4) → train with OOM recovery → validation rollback → save adapter → mark rows as trained → cleanup old adapters (max 10) |

#### Training Pipeline Detail

```
1. Load replay buffer (30-day rolling + 30% static baseline)
2. Filter rows where used_in_training == False
3. Check len(rows) >= MIN_TRAINING_ROWS (20)
4. Balance UP/DOWN/SIDEWAYS classes (sample min_count from each)
5. Format as prompt-completion pairs (only correct predictions)
6. Split: 90% train, 10% validation
7. Load microsoft/Phi-3-mini-4k-instruct in BF16
8. Apply LoRA: r=16, alpha=16, dropout=0.0, target=[q_proj, v_proj]
9. Tokenize: max_length=512, padding="max_length"
10. Configure Trainer: lr=2e-5, epochs=3, batch=2, grad_accum=4, early_stopping=3
11. Train with OOM recovery:
    → If CUDA OOM: halve batch_size (floor=1), double grad_accum, retry
    → If still OOM at batch=1: skip training, log warning
12. Validation rollback check:
    → If val_loss > previous_val_loss: restore previous adapter
    → Else: save new adapter
13. Mark rows as used_in_training = True
14. Cleanup old adapters (keep max 10)
```

### Step 4.3: Training Integration

| File | Deliverable |
|------|-------------|
| `main.py` | Post-market handler at 15:30 IST: check `training_enabled AND get_completed_rows() >= 20`, then `subprocess.run(["python", "learn.py", "--date", today])`. No model unload/reload needed. Handle subprocess failure gracefully |

### Acceptance Criteria
- [ ] `auto_detect_training()` correctly detects GPU/CPU
- [ ] `learn.py` runs as subprocess without crashing main.py
- [ ] Training data is balanced across UP/DOWN/SIDEWAYS
- [ ] LoRA adapter saves to `models/adapters/v1/`
- [ ] Validation rollback works when loss increases
- [ ] OOM recovery reduces batch size and retries
- [ ] Old adapters cleaned up (max 10 retained)
- [ ] Rows marked as `used_in_training` after training

---

## Phase 5 — Hardening (Days 13-14)

### Step 5.1: CPU-Only Testing

| Test | Expected Result |
|------|----------------|
| Run full system on CPU-only machine | Inference + dashboard + charts work |
| No Dhan API credentials | Auto-fallback to mock data |
| Measure end-to-end cycle latency | < 20s per cycle (3 tickers) |
| Dashboard updates | Correctly reflects predictions and status |

### Step 5.2: GPU Training Testing

| Test | Expected Result |
|------|----------------|
| Run on T4 16 GB (or equivalent) | Inference + training + adapter save work |
| Monitor VRAM during training | Stays within 12-14 GB |
| LoRA adapter saves and loads | Correctly loaded on next inference |
| Validation rollback | Rolls back when loss increases |

### Step 5.3: Low-VRAM GPU Testing

| Test | Expected Result |
|------|----------------|
| Run on 6 GB GPU (RTX 2060) | Inference works within Ollama GPU limits |
| Training detection | Gracefully skips (VRAM < 12 GB) |
| Log output | "Training disabled: 6 GB VRAM < 12 GB required" |

### Step 5.4: Edge Cases

| Scenario | Expected Behavior |
|----------|------------------|
| Ollama not running | Clear error message on startup, SystemExit(1) |
| Phi-3 model not pulled | Clear warning with `ollama pull` instructions |
| Dhan API credentials missing | Auto-fallback to mock data, log INFO |
| Dhan API down mid-session | Auto-fallback to mock data, log WARNING |
| Orphaned ledger rows | `cleanup_orphans()` resolves on startup |
| Holiday (e.g., Diwali) | No predictions made, log INFO |
| Friday early close | 15:00 IST cutoff, not 15:15 |
| KeyboardInterrupt during training | Graceful skip, log, exit |
| WebSocket disconnect | Auto-reconnect in browser JS |
| Multiple browser tabs | All receive same WebSocket broadcast |
| Uvicorn port in use | CRITICAL log, exit |

---

## Per-Module Testing Requirements

### config.py
- Assert `len(TICKER_SYMBOLS) == 10`
- Assert `OLLAMA_BASE_URL == "http://localhost:11434"`
- Assert `PREDICTION_START < PREDICTION_END`

### utils.py
- Logger creates log file
- Retry decorator retries correct number of times

### dhan_client.py
- `fetch_security_map()` caches 10 Security IDs
- `generate_mock_data()` returns 10 DataFrames with correct columns
- `safe_poll_tickers()` falls back to mock without credentials

### market.py
- Monday 10:00 AM IST → trading
- Monday 08:00 AM IST → not trading
- Saturday → not trading
- Holiday → not trading

### indicators.py
- `compute_indicators()` returns all indicator columns
- RSI values between 0 and 100
- Insufficient data → NaN indicators

### agents.py
- `LocalBandSDK` creates rooms, sends messages, returns history
- `AGENT_ROLES` has exactly 3 agents

### predict.py
- Ollama connection test passes
- JSON generation returns valid JSON with correct keys
- `PredictionSquad.run()` returns `{direction, target_return_pct, confidence}`
- Band room contains 4 messages (3 agents + synthesizer)

### ledger.py
- `phase1_write()` returns valid UUID
- `resolve_phase2()` resolves pending rows
- `cleanup_orphans()` cleans stale rows
- `update_replay_buffer()` maintains 30-day window

### dashboard.py
- `DashboardState` initializes with empty dicts
- `update()` is non-blocking (< 50ms)
- `get_snapshot()` returns thread-safe deep copy

### charts.py
- `generate_all()` creates 3 PNG files in `charts/`

### web_server.py
- WebSocket accepts connections and sends JSON
- REST `/api/status` returns 200 with mode field
- Static files served at `/web/index.html`

### learn.py
- `format_training_pair()` returns prompt + completion
- `balance_classes()` equalizes class distribution
- Training runs without errors (on capable hardware)

---

## Latency Benchmarks

| Operation | Target | Critical Threshold |
|-----------|--------|-------------------|
| Dhan API poll (10 tickers) | < 500ms | < 2s |
| Mock data generation (10 tickers) | < 50ms | < 200ms |
| Indicator computation | < 50ms | < 200ms |
| Single agent inference (GPU) | < 500ms | < 2s |
| Single agent inference (CPU) | < 3s | < 10s |
| Full PredictionSquad (GPU) | < 2s | < 6s |
| Full PredictionSquad (CPU) | < 15s | < 45s |
| Parquet write | < 50ms | < 200ms |
| DashboardState update | < 1ms | < 5ms |
| WebSocket broadcast | < 10ms | < 50ms |
| Chart generation | < 5s | < 30s |
| Full cycle (GPU, all tickers) | < 20s | < 60s |

---

## File Creation Order

Build files in this exact sequence to satisfy dependencies:

```
Day 1:  requirements.txt, setup.sh, setup.bat, .env.example, .gitignore
Day 2:  config.py, utils.py
Day 3:  dhan_client.py, market.py, indicators.py
Day 4:  ledger.py
Day 5:  agents.py (LocalBandSDK + AGENT_ROLES)
Day 6:  predict.py (ollama_generate + PredictionSquad)
Day 7:  main.py (startup + event loop skeleton)
Day 8:  dashboard.py, web_server.py
Day 9:  web/index.html, web/css/styles.css, web/js/app.js, web/js/candlestick.js, web/js/charts.js
Day 10: main.py (auto_detect_training + post-market handler)
Day 11: learn.py
Day 12: charts.py, integration testing
Day 13: CPU-only testing, GPU training testing
Day 14: Edge case testing, documentation
```
