# Agent-NEE — Backend & Database Schema

**Full Name**: Agentic Supervised Fine-Tuning — Neural Execution Engine for Financial Analytics
**Version**: 1.0
**Storage**: Parquet (pyarrow) with UUID-based row matching

---

## 1. Module Dependency Graph

```
main.py (entry point)
├── config.py          (imported by all)
├── utils.py           (imported by all)
├── market.py          (imports: config, utils)
│   └── calls dhan_client.py, indicators.py
├── dhan_client.py     (imports: config, utils)
│   └── generate_mock_data() fallback
├── indicators.py      (imports: pandas-ta)
├── agents.py          (imports: config, utils)
│   └── LocalBandSDK singleton used by predict.py, dashboard.py, ledger.py
├── predict.py         (imports: config, utils, agents, requests)
│   └── PredictionSquad uses AGENT_ROLES + LocalBandSDK + Ollama API
├── ledger.py          (imports: config, utils, pyarrow)
├── dashboard.py       (imports: config, utils) — DashboardState data container
├── web_server.py      (imports: dashboard, fastapi, uvicorn)
│   └── reads DashboardState singleton, broadcasts via WebSocket
├── charts.py          (imports: config, ledger, matplotlib)
├── learn.py           (standalone subprocess, imports: torch, transformers, peft)
└── setup.sh / setup.bat  (one-time setup, not imported)
```

### Import Rules

| Rule | Description |
|------|-------------|
| `config.py` must be importable without any external deps | Only stdlib + python-dotenv |
| `agents.py` must be importable without `requests` | LocalBandSDK is pure Python |
| `learn.py` must never be imported by `main.py` | Called as subprocess only |
| `dashboard.py` must not block the prediction loop | Thread-safe Lock on DashboardState |
| All modules import `config` for constants | Single source of truth |

---

## 2. Parquet Schema — The Ledger

### Phase 1 Columns (written at prediction time)

| Column | Type | Description |
|--------|------|-------------|
| `row_id` | UUID (string) | Primary key, generated at Phase 1 write |
| `timestamp` | datetime64[ns] | IST timestamp of prediction |
| `ticker` | str | NSE ticker symbol (e.g., "NSE:RELIANCE") |
| `cycle_id` | str | Prediction cycle identifier |
| `open` | float64 | Opening price |
| `high` | float64 | High price |
| `low` | float64 | Low price |
| `close` | float64 | Closing price |
| `volume` | float64 | Trading volume |
| `vwap` | float64 | Volume Weighted Average Price |
| `rsi_14` | float64 | 14-period RSI |
| `macd` | float64 | MACD line |
| `macd_signal` | float64 | MACD signal line |
| `bb_upper` | float64 | Bollinger Band upper |
| `bb_lower` | float64 | Bollinger Band lower |
| `bb_middle` | float64 | Bollinger Band middle |
| `atr_14` | float64 | 14-period ATR |
| `prediction_direction` | str | "UP" / "DOWN" / "SIDEWAYS" |
| `prediction_return_pct` | float64 | Target return percentage |
| `prediction_confidence` | str | "LOW" / "MED" / "HIGH" |
| `agent_contributions` | str | JSON array of {agent, text, latency_ms} |
| `band_room_id` | str | LocalBandSDK room ID for traceability |
| `model_version` | str | Ollama model identifier (e.g., "phi3:mini") |
| `inference_latency_ms` | float64 | Total prediction squad latency |
| `mode` | str | "inference_only" / "training_enabled" |
| `data_source` | str | "dhan_api" / "mock" |

### Phase 2 Columns (added at resolution time, T+1 cycle)

| Column | Type | Description |
|--------|------|-------------|
| `actual_direction` | str | "UP" / "DOWN" / "SIDEWAYS" |
| `actual_return_pct` | float64 | Actual return percentage |
| `resolution_timestamp` | datetime64[ns] | When resolution occurred |
| `prediction_accuracy` | bool | True if direction matches actual |
| `resolution_price` | float64 | Price at resolution time |
| `reward` | float64 | Prediction accuracy score (for training signal) |
| `used_in_training` | bool | Whether this row was used in training |

### File Organization

```
data/
├── daily/
│   ├── NSE_RELIANCE.parquet
│   ├── NSE_TCS.parquet
│   ├── NSE_HDFCBANK.parquet
│   └── ... (one file per ticker)
└── replay_buffer/
    ├── replay_buffer.parquet       # Unified 30-day rolling buffer
    └── static_baseline.parquet     # 6-month historical baseline
```

---

## 3. Key Ledger Functions

| Function | Signature | Returns | Description |
|----------|-----------|---------|-------------|
| `phase1_write` | `(row_data: dict) -> str` | UUID string | Write Phase 1 prediction data |
| `resolve_phase2` | `(timestamp: pd.Timestamp) -> int` | Count resolved | Resolve all pending rows whose 5-min window elapsed |
| `get_completed_rows` | `() -> pd.DataFrame` | DataFrame | All Phase 2-resolved rows (for training eligibility) |
| `get_agent_contributions` | `(row_id: str) -> list` | Message list | Agent contributions JSON for a given row |
| `cleanup_orphans` | `() -> int` | Count cleaned | Resolve or mark orphaned rows on startup |
| `update_replay_buffer` | `() -> None` | None | Append resolved rows, trim to 30 days |
| `get_replay_buffer` | `() -> pd.DataFrame` | DataFrame | All rows in replay buffer for training |

---

## 4. Replay Buffer

### 4.1 Rolling Buffer (30-Day)

- **Location**: `data/replay_buffer/replay_buffer.parquet`
- **Strategy**: Append-only; trimming happens once per day during post-market processing
- **Window**: 30 trading days (~45 calendar days)
- **Min rows for training**: `MIN_TRAINING_ROWS = 20`

### 4.2 Mixed Replay Buffer (70/30 Split)

To prevent regime overfitting (e.g., unlearning bear-market mechanics during a bull run), training uses a mixed buffer:

| Source | Percentage | Description |
|--------|-----------|-------------|
| Rolling Buffer | 70% | Last 30 trading days |
| Static Baseline | 30% | Fixed 6-month historical Parquet covering diverse regimes |

- **Static baseline location**: `data/replay_buffer/static_baseline.parquet`
- **Static baseline is never trimmed** — only appended to when new historical data becomes available
- **Seeded on first training run** using historical data from Dhan API (or pre-built file)

---

## 5. LocalBandSDK — Agent Coordination Layer

### Class Definition

```python
class LocalBandSDK:
    """In-memory room/message simulation for agent coordination."""

    def __init__(self):
        self.rooms = {}
        self.room_counter = 0

    def create_room(self, name) -> str:
        """Create a new room. Returns room_id (e.g., 'room_1')."""

    def send_message(self, room_id, sender, text, msg_type="contribution"):
        """Send a message to a room.
        msg_type: "contribution" | "synthesis" | "system"
        """

    def get_room_history(self, room_id) -> list:
        """Return all messages in chronological order."""

    def get_history_formatted(self, room_id, max_chars=3000) -> str:
        """Return formatted history string for logging/display."""
```

### Message Structure

```json
{
  "sender": "Technical Analyst",
  "text": "RSI at 58, VWAP above current price...",
  "type": "contribution",
  "turn": 0
}
```

### Singleton Instance

```python
band = LocalBandSDK()  # Import in predict.py, dashboard.py, ledger.py, web_server.py
```

### Usage Per Prediction Cycle

```
1. room_id = band.create_room(f"cycle_{ticker}_{timestamp}")
2. For each agent:
   → result = ollama_generate(prompt)
   → band.send_message(room_id, agent_name, result)
3. Synthesizer:
   → band.send_message(room_id, "Synthesizer", final_json, msg_type="synthesis")
4. ledger.phase1_write(prediction, agent_contributions=band.get_room_history(room_id))
5. dashboard.state.update(agent_activity={ticker, agents: band.get_room_history(room_id)})
6. Rooms auto-cleaned after Phase 2 resolution
```

---

## 6. Agent Roles (Exact Prompts)

### Agent 1: Technical Analyst

```
You are a technical analysis specialist for NSE equities.
Analyze the price action, VWAP, RSI, MACD, Bollinger Bands, and ATR.
Given the market data below, predict:
1. Direction (UP/DOWN/SIDEWAYS)
2. Target return percentage
3. Confidence level (LOW/MED/HIGH)
Cite specific indicator values. Max 200 words.
```

### Agent 2: Volatility Analyst

```
You are a volatility and risk specialist for NSE equities.
Focus on ATR, Bollinger Band width, and recent volatility patterns.
Given the market data below, produce:
1. Volatility regime (LOW/MED/HIGH)
2. Risk-adjusted confidence score
3. Any anomaly or squeeze signals
Cite specific volatility metrics. Max 200 words.
```

### Agent 3: Volume Analyst

```
You are a volume and liquidity analyst for NSE equities.
Analyze volume trends, volume vs VWAP, and volume spike patterns.
Given the market data below, assess:
1. Conviction behind current price move (STRONG/WEAK/NEUTRAL)
2. Any divergence between price and volume
3. Liquidity conditions for the predicted move
Cite specific volume figures. Max 200 words.
```

### Synthesis Prompt

```
You are the Agent-NEE Prediction Synthesizer.
Below are analyses from multiple specialist agents for the same ticker.
Merge them into a single final prediction.

Agent Analyses:
{agent_analyses}

Output ONLY a valid JSON object with these exact keys:
{"direction": "UP|DOWN|SIDEWAYS", "target_return_pct": <float>, "confidence": "LOW|MED|HIGH"}
No markdown, no explanation, no extra text.
```

---

## 7. JSON Enforcement

Ollama supports two approaches for structured output:

### Option A: `format: "json"` (Simple — Recommended)

```python
response = requests.post("http://localhost:11434/api/generate", json={
    "model": "phi3:mini",
    "prompt": synthesis_prompt,
    "format": "json",
    "temperature": 0.1,
    "max_tokens": 64,
    "stream": False,
})
```

### Option B: Explicit GBNF Grammar (Strict)

```
root        ::= "{" ws "\"direction\"" ws ":" ws direction ws "," ws
                "\"target_return_pct\"" ws ":" ws number ws "," ws
                "\"confidence\"" ws ":" ws confidence ws "}"
direction   ::= "\"UP\"" | "\"DOWN\"" | "\"SIDEWAYS\""
confidence  ::= "\"LOW\"" | "\"MED\"" | "\"HIGH\""
number      ::= "-"? [0-9]+ "."? [0-9]*
ws          ::= " "
```

Both use GBNF under the hood. Option A for simplicity, Option B for bulletproof schema enforcement.

---

## 8. PredictionSquad Schema

### Class Definition

```python
class PredictionSquad:
    """Runs all agents sequentially on the same Ollama model, then synthesizes."""

    def __init__(self, band: LocalBandSDK):
        self.band = band

    def run(self, ticker_data: str, room_id: str) -> dict:
        """Execute full multi-agent prediction cycle for one ticker.
        Returns: {"direction": str, "target_return_pct": float, "confidence": str}
        """
```

### ollama_generate() Function

```python
def ollama_generate(prompt, temperature=0.3, max_tokens=64, format=None, grammar=None, n_ctx=4096):
    """Call Ollama API for generation.
    
    - Fully stateless: no context array passed (fresh KV cache per call)
    - Model stays warm in memory (no keep_alive: 0)
    - Supports format="json" or grammar="..." (GBNF)
    - n_ctx passed as options.num_ctx
    """
```

### Agent Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_CTX` | 4096 | Context window (must fit synthesizer input ~1K+ tokens) |
| `AGENT_MAX_TOKENS` | 512 | Max tokens per agent prose output |
| `SYNTH_MAX_TOKENS` | 64 | Max tokens for synthesizer compact JSON |
| `INFERENCE_TEMPERATURE` | 0.3 | Agent sampling temperature |
| `SYNTHESIS_TEMPERATURE` | 0.1 | Near-deterministic merging |
| `AGENT_TEMPERATURES` | `{technical: 0.3, volatility: 0.3, volume: 0.3}` | Per-agent overrides |

---

## 9. Dhan API Integration Contract

### Authentication

```python
from dhanhq import dhanhq
client = dhanhq(client_id=config.DHAN_CLIENT_ID, access_token=config.DHAN_ACCESS_TOKEN)
```

### Security ID Resolution

Dhan API uses numeric Security IDs, not ticker symbols. The system resolves `NSE:RELIANCE` → Security ID at startup:

```python
SECURITY_MAP_CACHE = {}  # "NSE:RELIANCE" → "12345"

def fetch_security_map():
    """Fetch and cache Security ID mappings for all configured tickers."""
    # Queries Dhan API for all securities
    # Matches by tradingSymbol + exchangeSegment == "NSE_EQ"
    # Caches in SECURITY_MAP_CACHE dict
```

### Polling

```python
def poll_tickers() -> dict[str, DataFrame | str]:
    """Fetch latest OHLCV data for all tickers.
    Returns: {ticker: DataFrame} or {ticker: "STALE"} on failure.
    """
```

### Rate Limiting

- Exponential backoff: initial 1s, max 60s
- On HTTP 429: backoff + retry
- On persistent failure: fall back to mock data

---

## 10. Mock Data Generator

### Base Price Map

| Ticker | Base Price |
|--------|-----------|
| NSE:RELIANCE | 2845.0 |
| NSE:TCS | 3920.0 |
| NSE:HDFCBANK | 1650.0 |
| NSE:INFY | 1480.0 |
| NSE:ICICIBANK | 1120.0 |
| NSE:SBIN | 780.0 |
| NSE:BHARTIARTL | 1250.0 |
| NSE:ITC | 480.0 |
| NSE:WIPRO | 510.0 |
| NSE:AXISBANK | 1080.0 |

### Output Format

Each ticker returns a DataFrame with columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`. The `high` and `low` are constrained to be consistent with `open` and `close`.

---

## 11. Indicators Contract

### compute_indicators() Function

**Input**: DataFrame with columns `[open, high, low, close, volume]`
**Output**: Same DataFrame with indicator columns appended

| Indicator | Column(s) | Parameters |
|-----------|-----------|------------|
| VWAP | `vwap` | Daily reset, volume-weighted |
| RSI | `rsi_14` | 14-period |
| MACD | `macd`, `macd_signal` | 12, 26, 9 |
| Bollinger Bands | `bb_upper`, `bb_middle`, `bb_lower` | 20-period, 2 std |
| ATR | `atr_14` | 14-period |
| SMA 50 | `sma_50` | 50-period (if len >= 50) |
| SMA 200 | `sma_200` | 200-period (if len >= 200) |

### format_market_data_for_agent()

Formats the latest indicators row as a readable string for LLM prompts:

```
Ticker: NSE:RELIANCE
Price: O=2845.00 H=2862.00 L=2838.00 C=2855.00
Volume: 1200000
VWAP: 2850.00
RSI(14): 58.0
MACD: 12.00 (signal: 8.50)
Bollinger Bands: 2820.00 - 2850.00 - 2880.00
ATR(14): 18.50
```

### VWAP Daily Reset

VWAP resets at the start of each trading day:
`VWAP = Σ(Price_i × Volume_i) / Σ(Volume_i)` where Price = (H + L + C) / 3

---

## 12. Training Data Formation

### format_training_pair()

Only rows where prediction was **correct** are used as training examples:

```python
def format_training_pair(resolved_row) -> dict:
    """Returns {"prompt": str, "completion": str}"""
    # Prompt: market data string + "Predict direction, return%, confidence"
    # Completion: JSON with actual_direction, actual_return_pct, original confidence
```

### balance_classes()

Ensures balanced UP/DOWN/SIDEWAYS distribution:
- Counts occurrences of each class
- Samples min_count from each class
- Shuffles the result

### Adapter Directory Structure

```
models/adapters/
├── latest.txt              # Points to current active adapter
├── current_version.txt     # Path of latest trained adapter
├── previous_version.txt    # Path of previous adapter (for rollback)
├── v1/                     # Adapter checkpoint 1
├── v2/                     # Adapter checkpoint 2
└── ... (max 10)
```

### Validation Rollback

After each training run:
1. Save new adapter to `adapters/v{version}/`
2. Update `current_version.txt` with new path
3. Update `previous_version.txt` with old path
4. Evaluate validation loss
5. If val_loss increased → restore previous adapter
6. Delete oldest adapters when count > `MAX_ADAPTERS_TO_KEEP`
7. Update `latest.txt` with current active adapter

---

## 13. Integration Contracts

### dhan_client.py ↔ market.py
- `market.py` calls `dhan_client.safe_poll_tickers()` during trading hours
- Returns `dict[str, DataFrame | "STALE"]`
- Falls back to `generate_mock_data()` if Dhan API unavailable

### indicators.py ↔ predict.py
- `indicators.format_market_data_for_agent(ticker, row)` returns formatted string
- `predict.PredictionSquad.run()` receives this string as `ticker_data`
- The string is the **only** market data the agents see

### predict.py ↔ ledger.py
- `ledger.phase1_write()` expects a flat dict with all schema columns
- `agent_contributions` must be JSON-serialized list of LocalBandSDK messages
- `row_id` UUID is generated by the caller (not ledger)

### agents.py ↔ dashboard.py
- `agents.band.get_room_history(room_id)` returns agent contributions
- `dashboard.state.update(agent_activity=...)` stores in shared DashboardState
- `web_server.py` reads DashboardState and broadcasts over WebSocket

### main.py ↔ learn.py
- `learn.py` is standalone, called via `subprocess.run()`
- Receives `--date YYYY-MM-DD` as command-line argument
- Returns exit code 0 on success (training may be skipped internally)
- No model unload/reload needed — Ollama runs independently

### main.py ↔ Ollama
- `main.py` verifies Ollama running on startup via `verify_ollama()`
- All inference through Ollama REST API at `http://localhost:11434`
- Ollama manages model process lifecycle independently
- If Ollama down, system exits with clear error message
