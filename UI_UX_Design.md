# Agent-NEE — UI/UX Design Specifications

**Full Name**: Agentic Supervised Fine-Tuning — Neural Execution Engine for Financial Analytics
**Version**: 1.0
**Theme**: Terminal Dark (Green-on-Black)
**Font**: JetBrains Mono (Google Fonts CDN)

---

## 1. Architecture Overview

The dashboard is a **web-based single-page application** served via FastAPI + WebSocket. The Python backend runs a lightweight HTTP server that pushes real-time updates to browser clients at 1 FPS.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        WEB DASHBOARD                                  │
│                  web_server.py (FastAPI + Uvicorn)                    │
│                                                                      │
│  ┌─────────────────────┐    ┌──────────────────────────────────────┐ │
│  │  REST API           │    │  WebSocket (/ws)                     │ │
│  │  /api/status        │    │  Broadcasts JSON every ~1s:          │ │
│  │  /api/config        │    │  {market, predictions, agent_activity│ │
│  │  /api/history       │    │   latency, status}                   │ │
│  └─────────────────────┘    └──────────────┬───────────────────────┘ │
│                                            │                          │
└────────────────────────────────────────────┼──────────────────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │   Browser Client (port 8080) │
                              │   http://localhost:8080      │
                              │                              │
                               │  ┌─────────────────────────┐ │
                               │  │  index.html             │ │
                               │  │  css/styles.css         │ │
                               │  │  js/app.js              │ │
                               │  │  js/charts.js           │ │
                               │  │  js/candlestick.js      │ │
                               │  └─────────────────────────┘ │
                               └─────────────────────────────┘
```

### Frontend File Structure

```
Agent-NEE/
├── web/
│   ├── index.html          # Single-page dashboard layout
│   ├── css/
│   │   └── styles.css      # Terminal green-on-black stylesheet
│   └── js/
│       ├── app.js          # WebSocket client + DOM updates
│       ├── charts.js       # Chart.js (accuracy, scatter, latency)
│       └── candlestick.js  # Lightweight Charts (TradingView) live candlestick
```

### Frontend Dependencies (CDN — no npm)

| Library | Source | Purpose |
|---------|--------|---------|
| Chart.js | `https://cdn.jsdelivr.net/npm/chart.js` | AI charts (accuracy, scatter, latency) |
| Lightweight Charts | `https://unpkg.com/lightweight-charts@4.2.1/...` | Live candlestick chart (TradingView) |
| JetBrains Mono | `https://fonts.googleapis.com/css2?family=JetBrains+Mono` | Terminal monospace font |

---

## 2. Design System — Terminal Dark Theme

### 2.1 Color Palette

| Token | Hex | Usage |
|-------|-----|-------|
| `--bg-primary` | `#0a0a0a` | Page background (near-black) |
| `--bg-panel` | `#111111` | Panel background |
| `--bg-hover` | `#1a1a1a` | Hover state |
| `--text-primary` | `#00ff41` | Primary terminal green |
| `--text-dim` | `#00aa2a` | Dimmed text |
| `--text-muted` | `#005f14` | Muted/border lines |
| `--accent-green` | `#00ff41` | Up / positive |
| `--accent-red` | `#ff3355` | Down / negative |
| `--accent-amber` | `#ffb000` | Warning / SIDEWAYS |
| `--border-color` | `#00ff41` | Thin borders (1px solid) |
| `--glow-green` | `0 0 6px rgba(0, 255, 65, 0.3)` | Subtle glow on headings |

### 2.2 Typography

- **Font**: `'JetBrains Mono', 'Courier New', monospace` — loaded from Google Fonts
- **All text**: Monospace for terminal aesthetic
- **Headings**: Subtle green text-shadow glow (`0 0 6px rgba(0, 255, 65, 0.3)`)
- **Numbers**: Right-aligned in monospace for column alignment
- **Font weights**: 400 (regular), 600 (semibold), 700 (bold)

### 2.3 Panel Style

- **Background**: `#111111`
- **Border**: `1px solid #00ff41`
- **Border radius**: `0` — no rounded corners anywhere (true terminal aesthetic)
- **Data rows**: Alternating `#111` / `#0d0d0d` for subtle separation
- **Scrollbar**: Thin green-on-black styled
- **No shadows**: Flat design, borders only

---

## 3. Page Layout

```
┌───────────────────────────────────────────────────────────────────────────┐
│  STATUS BAR  [MODE: INFERENCE ONLY] [Uptime: 2h 34m] [Acc: 62.3%]       │
├─────────────────────────────────┬────────────────────────────────────────┤
│                                 │  ┌─────────────────────────────────┐  │
│  LIVE CANDLESTICK CHART         │  │  PREDICTION PANEL               │  │
│  (Lightweight Charts)           │  │  RELIANCE → UP  +1.2%  HIGH     │  │
│                                 │  │  TCS     → DOWN -0.5%  MED      │  │
│  [RELIANCE]  ▲ 2,855  ▼ 2,838  │  │  HDFCBANK→ UP   +0.8%  HIGH     │  │
│  ┌───────────────────────────┐ │  │  INFY    → SIDE  0.0%  LOW      │  │
│  │        CANDLES            │ │  │  ICICIBANK→ UP  +1.1%  MED      │  │
│  │   ╱╲   ╱╲                │ │  │  ... (5 more)                     │  │
│  │  ╱  ╲ ╱  ╲               │ │  └─────────────────────────────────┘  │
│  │ ╱    ╲╱    ╲              │ │  ┌─────────────────────────────────┐  │
│  │╱              ╲            │ │  │  AGENT ACTIVITY                │  │
│  │  ██  ██  ██  ██  ██       │ │  │  [TECH] Tech Analyst (0.8s)   │  │
│  │  VOLUME HISTOGRAM          │ │  │   RSI at 58, VWAP above...    │  │
│  └───────────────────────────┘ │  │  [VOL] Vol Analyst (0.7s)      │  │
│  Ticker: ◄ RELIANCE ►         │  │   ATR at 1.2%, BB narrowing    │  │
│                                 │  │  [SYN] Synthesizer (0.5s)      │  │
│                                 │  │   → UP, +1.2%, HIGH           │  │
│                                 │  └─────────────────────────────────┘  │
├─────────────────────────────────┴────────────────────────────────────────┤
│  AI CHARTS (Chart.js — terminal themed)                                  │
│  ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐    │
│  │  Accuracy Trend    │ │  Pred vs Actual    │ │  Latency Trend     │    │
│  │  (rolling 20 line) │ │  (scatter plot)    │ │  (multi-line)      │    │
│  └────────────────────┘ └────────────────────┘ └────────────────────┘    │
└───────────────────────────────────────────────────────────────────────────┘
```

### Layout Grid

- **CSS Grid**: Two-column layout
  - Left column (60%): Candlestick chart
  - Right column (40%): Prediction panel + Agent activity panel (stacked)
- **Full width below**: AI Charts section (3 charts side by side)
- **Top**: Status bar (single line, full width)

---

## 4. Component Specifications

### 4.1 Status Bar (Terminal Header)

Single-line terminal prompt style showing system-level information:

```
[MODE: INFERENCE ONLY] [UPTIME: 2h 34m] [CYCLES: 1,847] [ACC: 62.3%] [DATA: MOCK] [TKRS: 3/10]
```

| Badge | Source Field | Style |
|-------|-------------|-------|
| MODE | `status.mode` | Green badge, uppercase |
| UPTIME | `status.uptime` | Dim green text |
| CYCLES | `status.cycles` | Monospace number |
| ACC | `status.accuracy` | Green if > 50%, red if < 50% |
| DATA | `status.data_source` | Amber warning if MOCK |
| TKRS | `status.active_tickers` | Green if 10/10, amber if < 10 |

### 4.2 Live Candlestick Panel (Lightweight Charts — TradingView)

The centerpiece of the dashboard. Uses **Lightweight Charts** library by TradingView.

| Property | Value |
|----------|-------|
| **Data source** | WebSocket `market` data, per-ticker OHLCV |
| **Candle colors** | Green body (`#00ff41`) = price up, Red body (`#ff3355`) = price down |
| **Volume histogram** | Rendered below candles in same pane, matching candle direction colors |
| **Grid lines** | Horizontal: dim green (`#005f14`), no vertical grid lines |
| **Crosshair** | Green vertical/horizontal line on hover |
| **Ticker selector** | Strip below chart with buttons per active ticker |
| **Price header** | Above chart: ticker name, last price (green/red), daily change % |
| **History buffer** | Last 100 candles maintained client-side |
| **Update cadence** | Every WebSocket push (~1s) |
| **Background** | Transparent (inherits `--bg-panel`) |

### 4.3 Prediction Panel

Terminal-styled cards showing latest prediction per ticker.

| Element | Style |
|---------|-------|
| **Direction badge** | `[UP]` green, `[DN]` red, `[--]` amber for SIDEWAYS |
| **Return %** | Green text if positive, red if negative — monospace with sign |
| **Confidence** | `LOW` amber, `MED` yellow, `HIGH` bright green |
| **Ticker name** | Bright green, clickable — switches candlestick view |
| **Confidence bar** | Visual bar: `████████` (HIGH), `████` (MED), `██` (LOW) |
| **Update cadence** | Instant after each `PredictionSquad.run()` completes |

```
┌─────────────────────────────────────┐
│  PREDICTIONS                        │
│  RELIANCE  [UP]   +1.24%  ████████ │
│  TCS       [DN]   -0.52%  ████     │
│  HDFCBANK  [UP]   +0.81%  ██████   │
│  INFY      [--]    0.00%  ██       │
│  ICICIBANK [UP]   +1.13%  ███████  │
└─────────────────────────────────────┘
```

### 4.4 Agent Activity Panel

Collapsible terminal cards showing per-agent contributions for the selected ticker.

| Element | Style |
|---------|-------|
| **Agent cards** | Name badge (green), latency display, prose text |
| **Agent prefixes** | `[TECH]` blue-green, `[VOL]` amber, `[SYN]` bright green |
| **Expandable** | Last message visible; click to expand full text in terminal scroll area |
| **Update cadence** | Instant after each `PredictionSquad.run()` |

```
┌─────────────────────────────────────┐
│  AGENT ACTIVITY — RELIANCE          │
│  ┌───────────────────────────────┐  │
│  │ [TECH] Tech Analyst  0.8s    │  │
│  │ RSI at 58, VWAP above current│  │
│  │ price, MACD positive cross... │  │
│  ├───────────────────────────────┤  │
│  │ [VOL]  Vol Analyst   0.7s    │  │
│  │ ATR at 1.2%, BB width...     │  │
│  ├───────────────────────────────┤  │
│  │ [SYN]  Synthesizer   0.5s    │  │
│  │ → UP, +1.2%, HIGH confidence │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

### 4.5 AI Charts Section (Chart.js)

Three real-time charts rendered client-side using Chart.js, styled to match terminal theme.

| Chart | Type | Data Source | Details |
|-------|------|-------------|---------|
| **Accuracy Trend** | Line (rolling 20-period) | WebSocket `status.accuracy` history | Green line on dark bg |
| **Pred vs Actual** | Scatter | Resolved Phase 2 rows from REST API | Green dots (correct) / Red dots (wrong) |
| **Latency Trend** | Multi-line | WebSocket `latency` history | Inference green, fetch amber, total cyan |

#### Chart.js Theme

- **Gridlines**: Green (`#005f14`)
- **Background**: Transparent
- **Dataset lines**: Green (`#00ff41`)
- **Scatter dots**: Green (correct) / Red (wrong)
- **Tooltips**: Black bg, green border, monospace font
- **Axis labels**: Monospace, dim green

---

## 5. index.html Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent-NEE Terminal</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="css/styles.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://unpkg.com/lightweight-charts@4.2.1/dist/lightweight-charts.standalone.production.js"></script>
</head>
<body>
    <!-- Status Bar -->
    <header id="status-bar">
        <span id="mode-badge" class="terminal-badge">MODE: INFERENCE ONLY</span>
        <span id="uptime">UPTIME: 0h 0m</span>
        <span id="accuracy">ACC: --</span>
        <span id="cycles">CYCLES: 0</span>
        <span id="data-source" class="terminal-warn">DATA: MOCK</span>
        <span id="ticker-count">TKRS: 0/10</span>
    </header>

    <!-- Main Layout: Candlestick Left + Right Panels -->
    <main id="dashboard">
        <section id="candlestick-section">
            <div id="candlestick-header">
                <span id="active-ticker-label">RELIANCE</span>
                <span id="ticker-price" class="price-up">2,855.00</span>
                <span id="ticker-change" class="price-up">+12.50 (0.44%)</span>
            </div>
            <div id="candlestick-chart"></div>
            <div id="ticker-strip">
                <!-- ticker buttons populated by app.js -->
            </div>
        </section>
        <aside id="right-panel">
            <section id="prediction-panel"><!-- populated by app.js --></section>
            <section id="agent-panel"><!-- populated by app.js --></section>
        </aside>
    </main>

    <!-- AI Charts -->
    <section id="charts-section">
        <div class="chart-container">
            <div class="chart-label">ACCURACY TREND (20)</div>
            <canvas id="accuracy-chart"></canvas>
        </div>
        <div class="chart-container">
            <div class="chart-label">PRED vs ACTUAL</div>
            <canvas id="scatter-chart"></canvas>
        </div>
        <div class="chart-container">
            <div class="chart-label">LATENCY (ms)</div>
            <canvas id="latency-chart"></canvas>
        </div>
    </section>

    <script src="js/app.js"></script>
    <script src="js/candlestick.js"></script>
    <script src="js/charts.js"></script>
</body>
</html>
```

---

## 6. WebSocket Message Format

The server broadcasts the following JSON structure every ~1 second:

```json
{
  "market": {
    "RELIANCE": {
      "open": 2845.0, "high": 2862.0, "low": 2838.0, "close": 2855.0,
      "volume": 1200000, "vwap": 2850.0, "rsi_14": 58.0, "macd": 12.0,
      "bb_upper": 2880.0, "bb_lower": 2820.0, "atr_14": 18.5
    },
    "TCS": { "...": "..." }
  },
  "predictions": {
    "RELIANCE": { "direction": "UP", "target_return_pct": 1.2, "confidence": "HIGH" },
    "TCS": { "direction": "DOWN", "target_return_pct": -0.5, "confidence": "MED" }
  },
  "agent_activity": {
    "ticker": "RELIANCE",
    "agents": [
      { "name": "Technical Analyst", "text": "RSI at 58...", "latency_ms": 800 },
      { "name": "Volatility Analyst", "text": "ATR at 1.2%...", "latency_ms": 700 },
      { "name": "Volume Analyst", "text": "Volume 15% above avg...", "latency_ms": 600 },
      { "name": "Synthesizer", "text": "→ UP, +1.2%, HIGH", "latency_ms": 500, "type": "synthesis" }
    ]
  },
  "latency": {
    "inference_avg_ms": 850,
    "inference_last_ms": 920,
    "data_fetch_avg_ms": 120,
    "indicators_avg_ms": 15,
    "total_cycle_avg_ms": 1015,
    "next_cycle_in_s": 3.9
  },
  "status": {
    "mode": "INFERENCE_ONLY",
    "training": "Disabled (no CUDA)",
    "uptime": "2h 34m",
    "cycles": 1847,
    "predictions": 18470,
    "accuracy": 62.3,
    "data_source": "mock",
    "active_tickers": "3/10",
    "ollama_model": "phi3:mini"
  }
}
```

---

## 7. DashboardState (Backend Data Container)

The original Rich terminal dashboard is refactored into a pure data container:

```python
@dataclass
class DashboardState:
    market_data: Dict[str, Any] = field(default_factory=dict)
    predictions: Dict[str, Any] = field(default_factory=dict)
    agent_activity: Dict[str, Any] = field(default_factory=dict)
    latency: Dict[str, Any] = field(default_factory=dict)
    status: Dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def get_snapshot(self) -> dict:
        with self._lock:
            return {
                "market": dict(self.market_data),
                "predictions": dict(self.predictions),
                "agent_activity": dict(self.agent_activity),
                "latency": dict(self.latency),
                "status": dict(self.status),
            }
```

### Data Flow

```
Prediction Loop (main.py, every 5s)
       │
       ▼ writes via Lock
  DashboardState (thread-safe singleton in dashboard.py)
       │
       ▼ reads every 1s
  web_server.py broadcast_loop() (asyncio background task)
       │
       ▼ broadcasts JSON
  WebSocket /ws
       │
       ▼
  Browser JS (app.js) → update DOM + Chart.js + Lightweight Charts
```

---

## 8. Non-Blocking Guarantees

| Component | Thread | Blocking? |
|-----------|--------|-----------|
| Prediction loop | Main thread | Never blocks on dashboard |
| DashboardState.update() | Main thread (called from prediction loop) | < 1ms dict write with Lock |
| Uvicorn web server | Daemon thread | Runs asyncio event loop independently |
| WebSocket broadcast_loop() | asyncio task (in Uvicorn thread) | Reads DashboardState snapshot |
| Browser rendering | Browser main thread | DOM updates from WebSocket messages |

---

## 9. Keyboard Shortcuts (Browser)

| Key | Action |
|-----|--------|
| `Q` | Graceful shutdown (via REST API call) |
| `R` | Force refresh all panels |
| `D` | Toggle debug overlay (raw JSON display) |

---

## 10. JavaScript Modules

### 10.1 app.js — WebSocket Client & DOM Updates

- WebSocket connection to `ws://localhost:8080/ws` with auto-reconnect
- JSON message handler → dispatches to DOM + candlestick + chart update functions
- `renderPredictionPanel(data)` — terminal prediction cards with `[UP]`/`[DN]`/`[--]` badges
- `renderAgentPanel(data)` — collapsible agent cards with `[TECH]`/`[VOL]`/`[SYN]` prefixes
- `renderStatusBar(data)` — single-line terminal prompt header
- Ticker click handler → `candlestick.switchTicker(ticker)` + agent activity context switch
- Routes `market_data` OHLCV to `candlestick.js`
- Routes data to `charts.js` for AI chart updates

### 10.2 candlestick.js — Live Candlestick Chart

- `initCandlestick(containerId)` — creates Lightweight Charts instance on `#candlestick-chart`
- `createCandlestickSeries()` — candlestick series: green body (up), red body (down)
- `createVolumeSeries()` — volume histogram: green/red bars matching candle direction
- `updateCandles(ticker, candles)` — appends new OHLCV candle, maintains 100-candle buffer
- `switchTicker(ticker)` — clears series, loads new ticker's historical candle buffer
- `updatePriceHeader(ticker, price, change)` — updates price bar above chart
- Terminal colors: `#005f14` gridlines, `#00ff41` crosshair, monospace axis labels, transparent bg

### 10.3 charts.js — AI Performance Charts

- Chart.js global defaults set to terminal theme (transparent bg, green gridlines, monospace)
- `initAccuracyChart(canvasId)` — line chart, rolling 20-period, green line
- `initScatterChart(canvasId)` — scatter, green dots (correct) / red dots (wrong)
- `initLatencyChart(canvasId)` — multi-line: inference green, fetch amber, total cyan
- `updateCharts(wsData)` — push new data points from WebSocket stream

---

## 11. Static Baseline Charts (matplotlib)

Post-market, `charts.py` (matplotlib) generates saved `.png` reports in `charts/`. These are separate from the live Chart.js charts in the browser.

| Chart | File | Description |
|-------|------|-------------|
| Accuracy Trend | `accuracy_trend.png` | Rolling accuracy over time |
| Pred vs Actual | `pred_vs_actual.png` | Scatter of predicted vs actual returns |
| Latency Trend | `latency_trend.png` | Inference latency over time |
