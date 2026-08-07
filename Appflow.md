# Agent-NEE — Application Flow

**Full Name**: Agentic Supervised Fine-Tuning — Neural Execution Engine for Financial Analytics
**Version**: 1.0
**Flow Granularity**: Startup → Prediction Cycle → Post-Market → Shutdown

---

## 1. Startup Sequence

When the user runs `main.py`, the following 9-step sequence executes:

```
Step 1: Load Configuration
   → config.py loads environment variables (DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
   → Validates required constants (tickers, model name, paths)
   → Creates runtime directories if missing (logs/, data/, models/, charts/)
   ↓
Step 2: Setup Logging
   → utils.setup_logging() creates file + console handlers
   → File: logs/bifas_nexus.log (daily rotation, 30-day retention)
   → Console: INFO level, simplified format
   ↓
Step 3: Hardware Probe
   → auto_detect_training() attempts to import torch
   → Checks: CUDA available? VRAM >= 12 GB? CUDA >= 12.1?
   → Sets: training_enabled = True/False
   → Logs: "Training ENABLED" or "Training disabled: <reason>"
   ↓
Step 4: Verify Ollama
   → GET http://localhost:11434/api/tags
   → Checks if phi3:mini (or configured model) is in model list
   → If Ollama not running: CRITICAL log + SystemExit(1)
   → If model not pulled: WARNING log + instructions
   ↓
Step 5: Initialize Dhan Client
   → If credentials present: dhan_client.fetch_security_map()
   → Maps NSE:RELIANCE → Security ID, NSE:TCS → Security ID, etc.
   → If credentials missing: INFO log, mock data will be used
   ↓
Step 6: Determine Active Tickers
   → is_cpu = not (training_enabled or _has_gpu_for_inference())
   → CPU mode: active_tickers = first 3 tickers
   → GPU mode: active_tickers = all 10 tickers
   → Logs: "Active tickers: 3/10 (CPU mode)" or "10/10 (GPU mode)"
   ↓
Step 7: Initialize PredictionSquad
   → Creates PredictionSquad instance with LocalBandSDK singleton
   → squad = predict.PredictionSquad(agents.band)
   ↓
Step 8: Cleanup Orphaned Ledger Rows
   → ledger.cleanup_orphans() scans for unresolved Phase 1 rows
   → Rows older than 1 hour: marked as failed
   → Rows 5 min - 1 hour: resolved with current prices
   ↓
Step 9: Start Web Server + Event Loop
   → Daemon thread: uvicorn.run(app, host="0.0.0.0", port=8080)
   → Auto-opens browser: http://localhost:8080/web/index.html
   → Enters event_loop() — runs until SIGINT or market close
```

---

## 2. Prediction Event Loop

The main prediction loop runs every 5 seconds during market hours.

```
┌─────────────────────────────────────────────────────────────┐
│                    EVENT LOOP (every 5s)                      │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Step A: Check Trading Hours                           │   │
│  │   → market.is_within_trading_hours(now)               │   │
│  │   → If outside hours: sleep(60), continue             │   │
│  │   → If holiday/weekend: sleep(60), continue           │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Step B: Check Cycle Timing                            │   │
│  │   → If last_prediction_time is None OR                │   │
│  │     (now - last) >= 300 seconds:                      │   │
│  │     → Proceed to Step C                               │   │
│  │   → Else: sleep(1), continue                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Step C: Resolve Pending Phase 2 Predictions           │   │
│  │   → ledger.resolve_phase2(now)                        │   │
│  │   → For each unresolved row where elapsed >= 300s:    │   │
│  │     → Fetch current price                             │   │
│  │     → Calculate actual_return_pct                     │   │
│  │     → Determine actual_direction (UP/DOWN/SIDEWAYS)   │   │
│  │     → Match by UUID, write Phase 2 columns            │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Step D: Poll Market Data                              │   │
│  │   → dhan_client.safe_poll_tickers()                   │   │
│  │   → Returns dict[ticker, DataFrame | STALE]           │   │
│  │   → Falls back to generate_mock_data() if needed      │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Step E: Compute Indicators                            │   │
│  │   → For each active ticker (not STALE):               │   │
│  │     → indicators.compute_indicators(candles)          │   │
│  │     → VWAP, RSI(14), MACD, BB, ATR, SMA              │   │
│  │     → Skip if insufficient data (< 30 rows)           │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Step F: Run PredictionSquad                           │   │
│  │   → For each active ticker with valid indicators:     │   │
│  │     → Format market data string for agents            │   │
│  │     → Create LocalBandSDK room                        │   │
│  │     → squad.run(ticker_data, room_id)                 │   │
│  │       → Agent 1: Technical Analyst (Ollama call)      │   │
│  │       → Agent 2: Volatility Analyst (Ollama call)     │   │
│  │       → Agent 3: Volume Analyst (Ollama call)         │   │
│  │       → Synthesizer: merge → JSON output              │   │
│  │     → Returns: {direction, target_return_pct,         │   │
│  │                  confidence}                           │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Step G: Write Phase 1 to Ledger                       │   │
│  │   → ledger.phase1_write({                             │   │
│  │       row_id: UUID,                                   │   │
│  │       timestamp: now,                                 │   │
│  │       ticker: ticker,                                 │   │
│  │       ...indicators,                                  │   │
│  │       ...prediction,                                  │   │
│  │       agent_contributions: JSON(room_history),        │   │
│  │       band_room_id: room_id,                          │   │
│  │       mode: "training_enabled" | "inference_only",    │   │
│  │       data_source: "dhan_api" | "mock"                │   │
│  │     })                                                │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Step H: Update DashboardState                         │   │
│  │   → dashboard.state.update(                           │   │
│  │       market_data={...},                              │   │
│  │       predictions={...},                              │   │
│  │       agent_activity={...},                           │   │
│  │       latency={...},                                  │   │
│  │       status={...}                                    │   │
│  │     )                                                 │   │
│  │   → Web server reads this and broadcasts via WS       │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ▼                                    │
│                    sleep(1) — loop repeats                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Market Hours Gate

### Trading Schedule

| Day | Prediction Window | Duration | Notes |
|-----|------------------|----------|-------|
| Monday | 09:25 - 15:15 IST | 5h 50m | Full day |
| Tuesday | 09:25 - 15:15 IST | 5h 50m | Full day |
| Wednesday | 09:25 - 15:15 IST | 5h 50m | Full day |
| Thursday | 09:25 - 15:15 IST | 5h 50m | Full day |
| Friday | 09:25 - 15:00 IST | 5h 35m | Early close |
| Saturday | Closed | - | Weekend |
| Sunday | Closed | - | Weekend |
| Indian Holidays | Closed | - | Per `holidays.INDIA` |

### Gate Logic

```
is_within_trading_hours(dt):
   1. Check dt.weekday() >= 5 → return False (weekend)
   2. Check dt.date() in holidays.INDIA → return False (holiday)
   3. Get time_str = dt.strftime("%H:%M")
   4. If Friday: return "09:25" <= time_str < "15:00"
   5. Else: return "09:25" <= time_str < "15:15"
```

### Pre-Market Warmup
At 09:15 IST, the system begins fetching data but does NOT make predictions until 09:25 IST. This ensures indicators have fresh data when trading begins.

---

## 4. Overlapping Cycle Mechanism

Predictions are made every 5 minutes, but each prediction is resolved 5 minutes later. Multiple cycles are active simultaneously:

```
Time:    09:25:00    09:25:05    09:25:10    09:25:15    ...    09:30:00    09:30:05
         │           │           │           │                  │           │
Cycle:   C1         C2          C3          C4                 C1 resolve  C2 resolve
Phase:   Phase 1    Phase 1     Phase 1     Phase 1            Phase 2     Phase 2
         │           │           │           │                  │           │
         ▼           ▼           ▼           ▼                  ▼           ▼
Ledger:  write C1   write C2    write C3    write C4   ...    resolve C1  resolve C2
```

**Key behaviors:**
- Every `resolve_phase2()` call resolves ALL cycles whose 5-minute window has elapsed
- New Phase 1 writes happen every 300 seconds (5 minutes)
- The 5-second loop granularity ensures prompt Phase 2 resolution
- UUID matching prevents row collisions when multiple cycles resolve simultaneously

---

## 5. Phase 2 Resolution Flow

When a prediction's 5-minute window elapses:

```
resolve_phase2(current_timestamp):
   1. Load all Phase 1 rows where Phase 2 columns are empty
   2. For each unresolved row:
      a. elapsed = (current_timestamp - row.timestamp).total_seconds()
      b. If elapsed < 300: skip (not ready yet)
      c. Fetch current_price for row.ticker
      d. actual_return = (current_price - row.close) / row.close * 100
      e. If actual_return > 0.25: actual_direction = "UP"
      f. If actual_return < -0.25: actual_direction = "DOWN"
      g. Else: actual_direction = "SIDEWAYS"
      h. prediction_accuracy = (row.prediction_direction == actual_direction)
      i. Write Phase 2 columns via UUID match:
         - actual_direction
         - actual_return_pct
         - resolution_timestamp
         - prediction_accuracy
         - resolution_price
         - reward
         - used_in_training = False
```

---

## 6. Mock Data Fallback Flow

```
safe_poll_tickers():
   │
   ├─ Check: DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN set?
   │   ├─ No → log "Dhan credentials missing. Using mock data."
   │   │       → return generate_mock_data()
   │   │
   │   └─ Yes → try poll_dhan_api()
   │            ├─ Success → return data
   │            └─ Exception → log "Dhan API failed: {e}. Falling back to mock data."
   │                          → return generate_mock_data()
```

### STALE Ticker Handling

When a specific ticker fails to return data:
1. Mark that ticker as `STALE` (not all tickers)
2. No prediction made for that ticker this cycle
3. Dashboard shows warning icon for that ticker
4. If STALE persists for 10 consecutive cycles → mark as `DISCONNECTED`
5. On `DISCONNECTED`: skip all predictions for that ticker until next successful poll

---

## 7. Post-Market Flow

At 15:30 IST (after market close):

```
POST-MARKET SEQUENCE:
   │
   ├─ Step 1: Generate Charts
   │   → charts.generate_all(today)
   │   → Creates: accuracy_trend.png, pred_vs_actual.png, latency_trend.png
   │
   ├─ Step 2: Check Training Eligibility
   │   → If NOT training_enabled:
   │   │   → log "Post-market training skipped (inference-only mode)"
   │   │   → Done
   │   │
   │   → If training_enabled:
   │       → completed = ledger.get_completed_rows()
   │       → If len(completed) < 20:
   │       │   → log "Training skipped: {len(completed)} rows < 20"
   │       │   → Done
   │       │
   │       → If len(completed) >= 20:
   │           → log "Post-market training starting ({len(completed)} rows)..."
   │           → subprocess.run(["python", "learn.py", "--date", today])
   │           → log "Training completed"
   │
   └─ Step 3: Update Replay Buffer
       → ledger.update_replay_buffer()
       → Append today's resolved rows
       → Trim to 30 trading days
```

---

## 8. Orphaned Row Cleanup

On startup, the system cleans up Phase 1 rows that were never resolved (due to crash, shutdown, etc.):

```
cleanup_orphans():
   → Load all Phase 1 rows where Phase 2 columns are empty
   → For each orphaned row:
      → elapsed = (now - row.timestamp).total_seconds()
      → If elapsed > 3600 (1 hour):
      │   → Mark as failed (reason: "orphan_timeout")
      │
      → If elapsed > 300 (5 min) but < 3600:
      │   → Resolve now using current market prices
      │
      → If elapsed < 300:
          → Leave as-is (still within resolution window)
```

---

## 9. Signal Handling & Shutdown

### Signal Handlers

```
SIGINT (Ctrl+C)  → KeyboardInterrupt → graceful shutdown
SIGTERM           → KeyboardInterrupt → graceful shutdown
```

### Shutdown Sequence

```
shutdown():
   1. log "Shutting down Agent-NEE..."
   2. ledger.flush() — save any pending Parquet state
   3. Uvicorn daemon thread stops automatically with main thread
   4. log "Shutdown complete."
```

---

## 10. State Management Tree

```
DashboardState (singleton in dashboard.py)
│
├── market_data: Dict[str, Any]
│   └── Per ticker: {open, high, low, close, volume, vwap, rsi_14, macd, bb_upper, bb_lower, atr_14}
│
├── predictions: Dict[str, Any]
│   └── Per ticker: {direction, target_return_pct, confidence}
│
├── agent_activity: Dict[str, Any]
│   └── {ticker, agents: [{name, text, latency_ms, type}]}
│
├── latency: Dict[str, Any]
│   └── {inference_avg_ms, inference_last_ms, data_fetch_avg_ms, indicators_avg_ms, total_cycle_avg_ms, next_cycle_in_s}
│
└── status: Dict[str, Any]
    └── {mode, training, uptime, cycles, predictions, accuracy, data_source, active_tickers, ollama_model}
```

### Thread Safety
- All writes go through `DashboardState.update()` with `threading.Lock`
- All reads go through `DashboardState.get_snapshot()` which returns a deep copy
- Write latency: < 1ms (fast dict update)
- Read latency: < 1ms (dict copy)

---

## 11. Edge Cases & Error States

| Scenario | Behavior | Module |
|----------|----------|--------|
| Dhan API credentials missing | Auto-fallback to mock data, log INFO | dhan_client.py |
| Dhan API returns empty data | Mark ticker as STALE, skip prediction | dhan_client.py |
| Ollama not running | Startup check fails, CRITICAL log, SystemExit(1) | main.py |
| Ollama request timeout | 3 retries with backoff, then mark cycle FAILED | predict.py |
| Ollama returns gibberish JSON | Retry with lower temperature, then FAILED | predict.py |
| CUDA OOM during training | Reduce batch size, double grad accumulation, retry | learn.py |
| Parquet write conflict | 3 retries with 0.5s delay | ledger.py |
| LoRA adapter load failure | Fall back to base model (no adapter) | learn.py |
| Agent inference failure | Skip agent, log, synthesize with remaining agents | predict.py |
| WebSocket broadcast failure | Log DEBUG, remove client from broadcast list | web_server.py |
| WebSocket client disconnect | Remove client from connection manager | web_server.py |
| Uvicorn port already in use | CRITICAL log, exit | main.py |
| Market closes mid-cycle | Respect FRIDAY_EARLY_CLOSE, stop loop | market.py |
| Holiday detected mid-session | Stop prediction loop, log "Holiday detected" | market.py |
| All tickers STALE simultaneously | Dashboard shows "No data" warning, no predictions | dashboard.py |
| KeyboardInterrupt during training | Catch in main.py, log, skip training for the day | main.py |
