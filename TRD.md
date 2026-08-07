# Agent-NEE — Technical Requirements Document (TRD)

**Full Name**: Agentic Supervised Fine-Tuning — Neural Execution Engine for Financial Analytics
**Version**: 1.0
**Date**: 2024-10-24
**Python**: 3.10+
**OS**: Any (Linux, macOS, Windows) with Ollama support

---

## 1. Tech Stack

| Layer | Technology | Version | Mode |
|-------|-----------|---------|------|
| **Language** | Python | 3.10+ | Both |
| **Local LLM** | Configurable (default: `phi3:mini`, alt: `llama3.2:8b`, `llama3.3:8b`) | - | Both |
| **Inference Engine** | Ollama REST API | latest | Both |
| **JSON Enforcement** | Ollama `format: "json"` or explicit GBNF `grammar` parameter | built-in | Both |
| **Ollama Client** | `requests` to `http://localhost:11434` | `>=2.31.0` | Both |
| **Data Source** | Dhan API (dhanhq SDK) + built-in mock fallback | `==1.1.0` | Both |
| **Technical Indicators** | pandas-ta | `==0.3.14b0` | Both |
| **State Storage** | Parquet (pyarrow) | `>=14.0.0` | Both |
| **Web Server** | FastAPI + Uvicorn | `>=0.115.0` / `>=0.32.0` | Both |
| **Browser Charts** | Chart.js (CDN) | latest | Both |
| **Candlestick Charts** | Lightweight Charts by TradingView (CDN) | `4.2.1` | Both |
| **Saved Reports** | matplotlib | `>=3.8.0` | Both |
| **Timezone** | zoneinfo + holidays | `>=0.40` | Both |
| **Agent Coordination** | LocalBandSDK (in-memory rooms) | built-in | Both |
| **Training Framework** | transformers + peft LoRA SFT | `>=4.44.0` / `>=0.12.0` | Training only |
| **Efficient Training** | torch (CUDA) + accelerate | `>=2.1.0` / `>=0.34.0` | Training only |
| **Dataset Handling** | datasets (HuggingFace) | `>=2.21.0` | Training only |

---

## 2. Directory Structure

```
Agent-NEE/
│
├── main.py                       # Entry point: hardware probe, event loop, mode dispatch
├── config.py                     # All configuration: Dhan creds, tickers, Ollama config, paths
├── agents.py                     # LocalBandSDK + predefined agent roles + synthesis prompt
├── dhan_client.py                # Dhan API wrapper: Auth, Security ID map, Token refresh, mock fallback
├── market.py                     # IST gate (holidays) + data orchestration
├── indicators.py                 # pandas-ta math engine (5d warmup, VWAP daily reset)
├── predict.py                    # PredictionSquad: multi-agent orchestration + synthesizer + Ollama API calls
├── ledger.py                     # Two-phase Parquet + UUID matching + replay buffer + agent_contributions
├── learn.py                      # Standalone LoRA SFT training (transformers + peft, validation rollback)
├── charts.py                     # matplotlib: accuracy trend, pred vs actual, latency
├── dashboard.py                  # Data provider (DashboardState thread-safe singleton)
├── web_server.py                 # FastAPI app: WebSocket broadcast + REST endpoints + static file mount
├── utils.py                      # Logging, custom exceptions, retry decorators
│
├── web/                          # Frontend static files
│   ├── index.html                # Single-page dashboard layout (terminal theme)
│   ├── css/
│   │   └── styles.css            # Terminal green-on-black CSS
│   └── js/
│       ├── app.js                # WebSocket client + DOM panel updates
│       ├── charts.js             # Chart.js (accuracy, scatter, latency) — terminal themed
│       └── candlestick.js        # Lightweight Charts (TradingView) — live candlestick chart
│
├── requirements.txt              # Core deps only (training deps commented out)
├── setup.sh                      # Linux/macOS: pip install core deps + ollama pull phi3:mini
├── setup.bat                     # Windows: delegate to setup.sh or direct install
├── .env.example
├── .gitignore
└── README.md

# Runtime (auto-created, gitignored)
logs/
data/
├── daily/                        # Per-ticker daily Parquet ledgers
└── replay_buffer/                # Parquet files, last 30 trading days
    └── static_baseline.parquet   # 6-month historical baseline for mixed replay
models/
└── adapters/                     # LoRA checkpoints (only if training ran)
    ├── latest.txt
    ├── current_version.txt
    ├── previous_version.txt
    └── v1/ ... v10/              # Versioned adapter checkpoints
charts/
```

---

## 3. Requirements Split

### Core Dependencies (install on every device)

```
requests>=2.31.0
rich>=13.7.0
pandas>=2.0.0
pandas-ta==0.3.14b0
dhanhq==1.1.0
matplotlib>=3.8.0
holidays>=0.40
pyarrow>=14.0.0
python-dotenv>=1.0.0
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
```

### Training Dependencies (CUDA 12.1+ & 12 GB VRAM only — commented out in requirements.txt)

```
# torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu121
# transformers>=4.44.0
# peft>=0.12.0
# datasets>=2.21.0
# accelerate>=0.34.0
```

Training deps are imported only inside `learn.py` (standalone subprocess). They never cause import errors in `main.py` — if absent, the system simply skips training and logs the reason.

---

## 4. System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        AGENT-NEE ORCHESTRATOR                            │
│                      main.py (event loop + SIGINT)                       │
│                      Mode: training_enabled = auto_detect()              │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
          ┌──────────────────────┴──────────────────────┐
          ▼                                             ▼
┌─────────────────────────┐                 ┌─────────────────────────┐
│    MARKET HOURS GATE    │                 │   POST-MARKET TRIGGER   │
│    market.py            │                 │   Only if               │
│  09:25-15:15 M-F only   │                 │   training_enabled      │
│  Fri 3PM+ & Holidays    │                 │   15:30 IST → spawn     │
└────────────┬────────────┘                 │   subprocess(learn.py)  │
             │                              └────────────┬────────────┘
             ▼                                           │
┌─────────────────────────┐                              │
│   DHAN API INGESTION    │                              ▼
│  dhan_client.py         │                 ┌─────────────────────────┐
│  Fetch via Security IDs │                 │   learn.py (subprocess) │
│  Dynamic ID resolution  │                 │   Only called if        │
│  Mock fallback built-in │                 │   training_enabled AND  │
└────────────┬────────────┘                 │   rows >= 20            │
             │                              │                         │
             ▼                              │  ┌───────────────────┐  │
┌─────────────────────────┐                 │  │ Standard SFT      │  │
│  DETERMINISTIC MATH     │                 │  │ (supervised)      │  │
│  indicators.py          │                 │  │ prompt → correct  │  │
│  pandas-ta:             │                 │  │ completion pairs  │  │
│  VWAP, RSI, MACD,       │                 │  └───────────────────┘  │
│  BB, ATR (5d warmup)    │                 │  ┌───────────────────┐  │
└────────────┬────────────┘                 │  │ Mixed Replay      │  │
             │                              │  │ Buffer (70/30)    │  │
             ▼                              │  └───────────────────┘  │
┌─────────────────────────────────┐        │  ┌───────────────────┐  │
│  PREDICTION SQUAD (agents.py)   │        │  │ peft LoRA         │  │
│  ┌───────────────────────────┐  │        │  │ transformers      │  │
│  │ Technical Analyst         │  │        │  │ Trainer           │  │
│  │ Volatility Analyst       │  │        │  │ (Rollback logic)  │  │
│  │ Volume Analyst           │  │        │  │ (Class balancing) │  │
│  │ Synthesizer              │  │        │  └───────────────────┘  │
│  │ (sequential, shared      │  │        └────────────┬────────────┘
│  │  Ollama API)             │  │                     │
│  │ → final direction +      │  │                     │
│  │   return% + confidence   │  │                     │
│  └────────────┬─────────────┘  │                     │
│               │ LocalBandSDK   │                     │
│               │ room logging   │                     │
└───────────────┼─────────────────┘                     │
                │                                       │
                ▼                                       ▼
┌─────────────────────────┐    ┌────────────────────────────────────────┐
│  THE LEDGER — PHASE 1   │    │   charts.py (matplotlib)               │
│  ledger.py              │    │   accuracy_trend.png                   │
│  Parquet: UUID + state  │    │   pred_vs_actual.png                  │
└────────────┬────────────┘    │   latency_trend.png                    │
             │                 └────────────────────────────────────────┘
             ▼
┌─────────────────────────┐
│  THE LEDGER — PHASE 2   │
│  At T+1 (~5min later):  │
│  resolve via UUID match │
└─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    WEB DASHBOARD                                      │
│                    web_server.py (FastAPI + Uvicorn)                  │
│                    WebSocket /ws → real-time JSON push                │
│                    Browser at http://localhost:8080/web/index.html    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. Configuration Reference

### 5.1 Environment Variables

```python
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID")     # Optional — mock fallback if missing
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN") # Optional — mock fallback if missing
```

### 5.2 Ticker Configuration

```python
TICKER_SYMBOLS = [
    "NSE:RELIANCE", "NSE:TCS", "NSE:HDFCBANK", "NSE:INFY", "NSE:ICICIBANK",
    "NSE:SBIN", "NSE:BHARTIARTL", "NSE:ITC", "NSE:WIPRO", "NSE:AXISBANK",
]
```

### 5.3 Ollama Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MODEL_NAME` | `"phi3:mini"` | Default model. Swap to `llama3.2:8b` or `llama3.3:8b` |
| `OLLAMA_BASE_URL` | `"http://localhost:11434"` | Ollama API base URL |
| `OLLAMA_API_GENERATE` | `"{BASE}/api/generate"` | Generation endpoint |
| `OLLAMA_API_TAGS` | `"{BASE}/api/tags"` | Model listing endpoint |

#### Supported Models

| Model | Size | RAM Required | Best For |
|-------|------|-------------|----------|
| `phi3:mini` (default) | 2.2 GB | ~4 GB CPU | Low-resource devices, CPU-only |
| `llama3.2:8b` | 4.9 GB | ~8 GB CPU / 4 GB VRAM | Better quality, needs decent GPU |
| `llama3.3:8b` | 4.9 GB | ~8 GB CPU / 4 GB VRAM | Best quality, needs decent GPU |

### 5.4 Inference Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `INFERENCE_TEMPERATURE` | `0.3` | Sampling temperature for agents |
| `INFERENCE_TOP_P` | `0.9` | Top-p sampling |
| `INFERENCE_MAX_TOKENS` | `64` | Maximum tokens per inference |
| `N_CTX` | `4096` | Context window (must fit synthesizer input ~1K+ tokens) |
| `AGENT_MAX_TOKENS` | `512` | Max tokens per agent prose output (~200 words) |
| `SYNTH_MAX_TOKENS` | `64` | Max tokens for synthesizer compact JSON |
| `SYNTHESIS_TEMPERATURE` | `0.1` | Near-deterministic merging |
| `OLLAMA_TIMEOUT` | `30` | HTTP request timeout in seconds |

### 5.5 Time Windows (IST)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `PREDICTION_START` | `"09:25"` | First prediction of the day |
| `PREDICTION_END` | `"15:15"` | Last prediction (Mon-Thu) |
| `FRIDAY_EARLY_CLOSE` | `"15:00"` | Last prediction on Friday |
| `PREMARKET_WARMUP_TIME` | `"09:15"` | Warmup data fetch starts |
| `INTERVAL_MINUTES` | `5` | Overlapping cycle interval |

#### Holiday Handling
- Trading days: Monday through Friday
- Weekend: Saturday and Sunday (no predictions)
- Holidays: All Indian public holidays via `holidays.INDIA`
- Early close: Friday at 15:00 IST instead of 15:15 IST

### 5.6 Indicator Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CANDLE_HISTORY_DAYS` | `5` | Days of historical data for indicator computation |
| `MIN_INDICATOR_ROWS` | `30` | Minimum rows needed for stable indicators |
| `SIDEWAYS_THRESHOLD_PCT` | `0.25` | Below this absolute return%, direction = SIDEWAYS |

#### Indicators Computed (via pandas-ta)

| Indicator | Parameters | Source |
|-----------|-----------|--------|
| VWAP | Daily reset, volume-weighted | pandas-ta |
| RSI | 14-period | pandas-ta |
| MACD | 12, 26, 9 | pandas-ta |
| Bollinger Bands | 20-period, 2 std | pandas-ta |
| ATR | 14-period | pandas-ta |
| SMA | 50 and 200 period (if enough data) | pandas-ta |

### 5.7 Training Configuration (LoRA SFT)

#### Hardware Thresholds

| Parameter | Value |
|-----------|-------|
| `TRAINING_VRAM_MIN_GB` | `12` |
| `TRAINING_CUDA_MIN_VERSION` | `"12.1"` |

#### Training Data

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MIN_TRAINING_ROWS` | `20` | Minimum rows before training starts |
| `REPLAY_BUFFER_DAYS` | `30` | Rolling window size |
| `VALIDATION_SPLIT` | `0.1` | 10% validation split |
| `ROLLBACK_IF_VAL_LOSS_INCREASES` | `True` | Roll back if validation loss increases |
| `MAX_ADAPTERS_TO_KEEP` | `10` | Max LoRA checkpoints to retain |

#### LoRA Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `LORA_R` | `16` | LoRA rank |
| `LORA_ALPHA` | `16` | LoRA alpha scaling |
| `LORA_DROPOUT` | `0.0` | LoRA dropout |
| `LORA_TARGET_MODULES` | `["q_proj", "v_proj"]` | Target modules for LoRA |

#### SFT Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SFT_LEARNING_RATE` | `2e-5` | Learning rate |
| `SFT_NUM_EPOCHS` | `3` | Number of training epochs |
| `SFT_PER_DEVICE_BATCH_SIZE` | `2` | Batch size per device |
| `SFT_GRADIENT_ACCUMULATION_STEPS` | `4` | Gradient accumulation steps |
| `SFT_EARLY_STOPPING_PATIENCE` | `3` | Early stopping patience |
| `SFT_MAX_SEQ_LENGTH` | `512` | Maximum sequence length |

#### Mixed Replay Buffer

| Parameter | Value | Description |
|-----------|-------|-------------|
| `STATIC_BASELINE_PATH` | `"./data/replay_buffer/static_baseline.parquet"` | 6-month historical baseline |
| `STATIC_BASELINE_SPLIT` | `0.3` | 30% static, 70% rolling |

### 5.8 Agent Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_AGENTS` | `3` | Number of predefined agents |
| `AGENT_ROLES_TO_RUN` | `["technical_analyst", "volatility_analyst", "volume_analyst"]` | Subset selection |
| `BAND_ENABLED` | `True` | Enable LocalBandSDK coordination |
| `CPU_TICKER_LIMIT` | `3` | Max tickers per cycle on CPU |
| `GPU_TICKER_LIMIT` | `10` | Max tickers on GPU |

### 5.9 Mock Data Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MOCK_DATA_ENABLED` | `True` | Auto-fallback when Dhan API unavailable |
| `MOCK_PRICE_BASE` | `2500.0` | Base price for mock data |
| `MOCK_VOLATILITY` | `0.02` | Daily volatility (2%) |
| `MOCK_VOLUME_BASE` | `1000000` | Base volume |

### 5.10 Web Server Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `WEB_HOST` | `"0.0.0.0"` | Bind address |
| `WEB_PORT` | `8080` | Port for web dashboard |
| `WEB_AUTO_OPEN_BROWSER` | `True` | Auto-open browser on startup |
| `WEB_REFRESH_INTERVAL` | `1.0` | WebSocket broadcast interval (seconds) |

### 5.11 Candlestick Chart Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CANDLESTICK_HISTORY_CANDLES` | `100` | Historical candles maintained client-side |
| `CANDLESTICK_UP_COLOR` | `"#00ff41"` | Green candle body for up moves |
| `CANDLESTICK_DOWN_COLOR` | `"#ff3355"` | Red candle body for down moves |
| `CANDLESTICK_VOLUME_HEIGHT` | `0.2` | Volume histogram height ratio |

### 5.12 Paths

| Path | Purpose |
|------|---------|
| `./models/adapters/` | LoRA adapter checkpoints |
| `./data/daily/` | Per-ticker daily Parquet ledgers |
| `./data/replay_buffer/` | 30-day replay buffer + static baseline |
| `./charts/` | Saved matplotlib PNG reports |
| `./logs/` | Daily rotating log files |

---

## 6. Memory Management

### Mode 1: Universal (Inference Only)

| Component | VRAM | CPU RAM |
|-----------|------|---------|
| Ollama (Phi-3-mini, CPU mode) | 0 GB | ~4 GB |
| Ollama (Phi-3-mini, GPU offload) | ~2-3 GB | ~1 GB |
| pandas + parquet + DashboardState + web_server + LocalBandSDK | 0 GB | ~1 GB |
| **Total (GPU)** | **~2-3 GB** | **~2 GB** |
| **Total (CPU)** | **0 GB** | **~5 GB** |

#### Multi-Agent Impact
Agents run **sequentially** on the same Ollama instance — no additional RAM beyond the single loaded model. 3 agents + synthesizer (~4 sequential inferences per ticker) adds:
- **GPU**: ~1-2 seconds of latency per ticker
- **CPU**: ~8-12 seconds of latency per ticker

### Mode 2: Training (Auto-Enabled)

Training runs as a separate subprocess (`learn.py`). Ollama continues running for dashboard.

| Component | VRAM | Notes |
|-----------|------|-------|
| Phi-3-mini (transformers, LoRA, BF16) | ~8-10 GB | Full model loaded in BF16 |
| Optimizer states (AdamW) | ~2-3 GB | Per-parameter momentum + variance |
| Gradient accumulation buffer | ~1 GB | |
| **Total** | **~12-14 GB** | May exceed 12 GB on T4; auto-reduce triggers |

### OOM Auto-Recovery Chain

```
Training OOM:
   1. Catch RuntimeError
   2. Halve per_device_train_batch_size (floor at 1)
   3. Double gradient_accumulation_steps
   4. Retry training
   5. If OOM at batch_size=1: skip training for today, log warning
```

### Memory Cleanup Rules

1. **LocalBandSDK rooms** — Auto-cleaned after Phase 2 resolution. ~100 rooms/day, ~200 KB total. Cleaned to prevent unbounded growth.
2. **Parquet write buffers** — Flush to disk immediately after each write. Never accumulate in-memory.
3. **DashboardState** — Single `Lock` for writes; `get_snapshot()` returns deep copy. No UI rendering objects held.
4. **Log files** — Daily rotation, max 30 days. `TimedRotatingFileHandler`.
5. **Adapter checkpoints** — Keep max `MAX_ADAPTERS_TO_KEEP = 10` LoRA adapters. Delete oldest when exceeded.

---

## 7. Error Handling Architecture

### 7.1 Custom Exception Hierarchy

```
BIFASError (base)
├── ConfigError          — Configuration or environment variable error
├── DhanAPIError         — Dhan API connection or data error
├── OllamaError          — Ollama API connection or inference error
├── LedgerError          — Parquet ledger read/write error
├── TrainingError        — LoRA SFT training error
└── AgentError           — Multi-agent squad execution error
```

### 7.2 Retry Decorator

`retry_with_backoff(max_retries=3, base_delay=1, backoff_factor=2, max_delay=60)` — Exponential backoff retry decorator applied to Dhan API calls, Parquet writes, and Ollama inference.

### 7.3 Recovery Flows

| Error | Detection | Recovery Action | Log Level |
|-------|-----------|----------------|-----------|
| Dhan API rate limit | HTTP 429 | Exponential backoff (max 60s), then mock fallback | WARNING |
| Dhan API connection failure | Timeout / ConnectionError | 3 retries with backoff, then mock fallback | WARNING |
| Dhan API token expired | Auth error | Log + fall back to mock data | WARNING |
| Dhan credentials missing | Empty env vars | Automatically use mock data | INFO |
| Ollama not running | ConnectionError | Log critical, prompt user to start Ollama | CRITICAL |
| Ollama request timeout | Timeout | 3 retries with backoff, then mark cycle FAILED | WARNING |
| Ollama returns gibberish | JSON parse error | Retry with lower temperature, then FAILED | WARNING |
| CUDA OOM (training) | RuntimeError | Reduce batch size, double gradient accumulation | WARNING |
| Parquet write conflict | PermissionError | 3 retries with 0.5s delay | WARNING |
| LoRA adapter load failure | Exception | Fall back to base model (no adapter) | WARNING |
| Agent inference failure | Timeout / Exception | Skip agent, log, synthesize with remaining | WARNING |
| WebSocket broadcast failure | Connection reset | Log debug, remove client from broadcast list | DEBUG |
| WebSocket client disconnect | WebSocketDisconnect | Remove client from connection manager | INFO |
| Uvicorn startup failure | Port in use / OSError | Log critical, fall back to console-only mode | CRITICAL |
| KeyboardInterrupt / SIGINT | Signal handler | Graceful shutdown: save state, exit | INFO |
| Orphaned ledger rows | Startup scan | Resolve or expire stale rows | INFO |

### 7.4 Logging Configuration

- **File**: `logs/bifas_nexus.log` with daily rotation (`TimedRotatingFileHandler`)
- **Retention**: 30 days max
- **Levels**: INFO during normal operation, DEBUG for troubleshooting, ERROR for failures
- **Format**: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- **Console**: `%(levelname)s: %(message)s` (INFO level)
- **Required logs per cycle**: ticker, direction, return%, confidence, latency_ms

---

## 8. Performance Targets

| Operation | Target | Critical Threshold | Measurement |
|-----------|--------|-------------------|-------------|
| Dhan API poll (10 tickers) | < 500ms | < 2s | `dhan_client.safe_poll_tickers()` |
| Mock data generation (10 tickers) | < 50ms | < 200ms | `dhan_client.generate_mock_data()` |
| Indicator computation | < 50ms | < 200ms | `indicators.compute_indicators()` |
| Single agent inference (GPU) | < 500ms | < 2s | `ollama_generate()` single call |
| Single agent inference (CPU) | < 3s | < 10s | `ollama_generate()` single call |
| Full PredictionSquad (GPU) | < 2s | < 6s | `squad.run()` 3 agents + synthesizer |
| Full PredictionSquad (CPU) | < 15s | < 45s | `squad.run()` 3 agents + synthesizer |
| Parquet write | < 50ms | < 200ms | `ledger.phase1_write()` |
| DashboardState update | < 1ms | < 5ms | `DashboardState.update()` |
| WebSocket broadcast | < 10ms | < 50ms | `broadcast_loop()` push to N clients |
| Chart generation | < 5s | < 30s | `charts.generate_all()` |
| Full cycle (GPU, all tickers) | < 20s | < 60s | Start to finish for 10 tickers |

---

## 9. Technical Rules & Constraints

### Rule 1: Dependency Isolation
Training dependencies (`torch`, `transformers`, `peft`, `datasets`, `accelerate`) must NEVER be imported in `main.py`. They are imported only inside `learn.py` (standalone subprocess). `auto_detect_training()` uses try/except to probe availability.

### Rule 2: GBNF/JSON Enforcement
All structured LLM output must use Ollama's built-in GBNF enforcement. Two options:
- **Option A**: `format: "json"` — simple, auto-generated grammar
- **Option B**: Explicit `grammar` parameter — tighter schema control with exact enum values

### Rule 3: Sequential Agents
All 3 agents and the synthesizer must run sequentially on the same Ollama instance. No parallel model loading. Ollama queues requests per model — parallel would serialize anyway.

### Rule 4: Ollama Health Check
Before the prediction loop starts, `main.py` must verify Ollama is running via `GET /api/tags` and confirm the model is available.

### Rule 5: OOM Auto-Reduce
If CUDA OOM during training: halve batch size (floor at 1), double gradient accumulation, retry. If still OOM at batch_size=1, skip training for the day.

### Rule 6: UUID-Based Ledger
Every prediction row must have a UUID `row_id` generated at Phase 1 write. Phase 2 resolves by matching UUID. Never use sequential IDs or timestamps alone.

### Rule 7: Single Ticker Source
Ticker symbols defined only in `config.py`. Never hardcoded in other modules.

### Rule 8: Market Hours Gate
Predictions only during 09:25-15:15 IST (Mon-Thu), 09:25-15:00 IST (Fri). No weekends or Indian holidays. Silently skipped outside windows.

### Rule 9: Non-Blocking Dashboard
`DashboardState.update()` must be non-blocking (fast dict writes with Lock). WebSocket broadcast in separate async task. Prediction loop never blocks.

### Rule 10: Agent Contributions in Ledger
Every Phase 1 row must include `agent_contributions` (JSON) and `band_room_id` columns.

### Rule 11: Mock Data Fallback
System must work without Dhan API credentials. `safe_poll_tickers()` falls back to `generate_mock_data()`.

### Rule 12: Logging Standards
All logs to `logs/bifas_nexus.log` with daily rotation. Every cycle logs: ticker, direction, return%, confidence, latency_ms.

### Rule 13: Stateless Agent Calls
No `context` array between sequential Ollama calls. No `keep_alive: 0` (keeps model warm). Fresh KV cache per agent.

### Rule 14: Dynamic Ticker Scaling
CPU: max 3 tickers per cycle. GPU: all 10 tickers. Dashboard reflects active count.

### Rule 15: Non-Blocking WebSocket
All WebSocket I/O in asyncio on daemon thread. Prediction loop writes to DashboardState and continues immediately.
