/* Agent-NEE — Main WebSocket Client & DOM Updates */

let ws = null;
let reconnectTimer = null;
const WS_URL = `ws://${window.location.host}/ws`;

// ─── HTML Escaping (XSS Prevention) ──────────────────────────────────

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ─── WebSocket Connection ────────────────────────────────────────────

function connectWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) return;

    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        console.log('WebSocket connected');
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleMessage(data);
        } catch (e) {
            console.error('Failed to parse WS message:', e);
        }
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected. Reconnecting...');
        reconnectTimer = setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        ws.close();
    };

    setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send('ping');
        }
    }, 30000);
}

// ─── Message Handler ─────────────────────────────────────────────────

function handleMessage(data) {
    if (data.market) renderMarketData(data.market);
    if (data.predictions) renderPredictionPanel(data.predictions);
    if (data.agent_activity) renderAgentPanel(data.agent_activity);
    if (data.status) renderStatusBar(data.status);
    if (data.latency || data.status) updateCharts(data);

    if (data.market) {
        for (const [ticker, mdata] of Object.entries(data.market)) {
            const candles = [{
                timestamp: new Date().toISOString(),
                open: mdata.open, high: mdata.high,
                low: mdata.low, close: mdata.close,
                volume: mdata.volume || 0,
            }];
            updateCandles(ticker, candles);
        }
    }
}

// ─── Status Bar ──────────────────────────────────────────────────────

function renderStatusBar(status) {
    const el = (id) => document.getElementById(id);

    if (status.mode) {
        const badge = el('mode-badge');
        badge.textContent = `MODE: ${escapeHtml(status.mode.replace(/_/g, ' ').toUpperCase())}`;
    }
    if (status.uptime) el('uptime').textContent = `UPTIME: ${escapeHtml(status.uptime)}`;
    if (status.accuracy !== undefined) {
        const accEl = el('accuracy');
        accEl.textContent = `ACC: ${Number(status.accuracy).toFixed(1)}%`;
        accEl.style.color = status.accuracy > 50 ? '#00ff41' : '#ff3355';
    }
    if (status.cycles !== undefined) el('cycles').textContent = `CYCLES: ${Number(status.cycles)}`;
    if (status.data_source) {
        const dsEl = el('data-source');
        dsEl.textContent = `DATA: ${escapeHtml(status.data_source.toUpperCase())}`;
        dsEl.style.color = status.data_source === 'mock' ? '#ffb000' : '#00ff41';
    }
    if (status.active_tickers) el('ticker-count').textContent = `TKRS: ${escapeHtml(status.active_tickers)}`;
}

// ─── Prediction Panel ────────────────────────────────────────────────

function renderPredictionPanel(predictions) {
    const container = document.getElementById('predictions-list');
    if (!container) return;

    const strip = document.getElementById('ticker-strip');
    if (strip.children.length === 0) {
        for (const ticker of Object.keys(predictions)) {
            const btn = document.createElement('button');
            btn.textContent = ticker.replace('NSE:', '');
            btn.onclick = () => switchTicker(ticker);
            strip.appendChild(btn);
        }
    }

    let html = '';
    for (const [ticker, pred] of Object.entries(predictions)) {
        const dir = escapeHtml((pred.direction || 'SIDEWAYS').toUpperCase());
        const dirClass = `direction-${dir}`;
        const dirLabel = dir === 'DOWN' ? 'DN' : dir === 'SIDEWAYS' ? '--' : 'UP';
        const returnPct = Number(pred.target_return_pct) || 0;
        const returnSign = returnPct >= 0 ? '+' : '';
        const returnClass = returnPct >= 0 ? 'price-up' : 'price-down';
        const conf = escapeHtml((pred.confidence || 'LOW').toUpperCase());
        const barLen = conf === 'HIGH' ? 8 : conf === 'MED' ? 4 : 2;
        const bar = '\u2588'.repeat(barLen);
        const safeTicker = escapeHtml(ticker);
        const shortTicker = escapeHtml(ticker.replace('NSE:', ''));

        html += `
            <div class="prediction-row" onclick="switchTicker('${safeTicker}')">
                <span class="ticker-name">${shortTicker}</span>
                <span class="direction ${dirClass}">[${dirLabel}]</span>
                <span class="return-pct ${returnClass}">${returnSign}${returnPct.toFixed(2)}%</span>
                <span class="confidence-bar">${bar}</span>
            </div>`;
    }
    container.innerHTML = html;
}

// ─── Agent Activity Panel ────────────────────────────────────────────

function renderAgentPanel(activity) {
    const container = document.getElementById('agents-list');
    if (!container) return;

    const ticker = escapeHtml(activity.ticker || 'N/A');
    const agents = activity.agents || [];

    let html = `<div style="padding:4px 12px;color:#005f14;font-size:11px;">${ticker}</div>`;

    for (const agent of agents) {
        const name = escapeHtml(agent.sender || agent.name || 'Unknown');
        const prefix = name.includes('Technical') ? 'TECH' :
                       name.includes('Volatility') ? 'VOL' :
                       name.includes('Volume') ? 'VOL' :
                       name.includes('Synthesizer') ? 'SYN' : '---';
        const prefixClass = `prefix-${prefix}`;
        const latency = agent.latency_ms ? `${(agent.latency_ms/1000).toFixed(1)}s` : '--';
        const text = escapeHtml((agent.text || '').substring(0, 200));

        html += `
            <div class="agent-card">
                <div class="agent-header">
                    <span class="agent-name"><span class="${prefixClass}">[${prefix}]</span> ${name}</span>
                    <span class="agent-latency">${latency}</span>
                </div>
                <div class="agent-text" onclick="this.classList.toggle('expanded')">${text}</div>
            </div>`;
    }
    container.innerHTML = html;
}

// ─── Market Data ─────────────────────────────────────────────────────

function renderMarketData(market) {
    for (const [ticker, data] of Object.entries(market)) {
        if (data.close !== undefined) {
            const change = data.close - data.open;
            updatePriceHeader(ticker, data.close, change);
        }
    }
}

// ─── Init ────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();

    document.addEventListener('keydown', (e) => {
        if (e.key === 'r' || e.key === 'R') {
            location.reload();
        }
    });
});
