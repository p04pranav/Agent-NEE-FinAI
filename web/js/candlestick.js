/* Agent-NEE — Candlestick Chart (TradingView Lightweight Charts) */

let candlestickChart = null;
let candleSeries = null;
let volumeSeries = null;
let activeTicker = "RELIANCE";
let candleBuffers = {};

function initCandlestick(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    candlestickChart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: container.clientHeight,
        layout: {
            background: { type: 'solid', color: '#111111' },
            textColor: '#00aa2a',
            fontFamily: "'JetBrains Mono', monospace",
        },
        grid: {
            vertLines: { color: '#005f14' },
            horzLines: { color: '#005f14' },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: { color: '#00ff41', width: 1, style: 0 },
            horzLine: { color: '#00ff41', width: 1, style: 0 },
        },
        timeScale: {
            borderColor: '#005f14',
            timeVisible: true,
            secondsVisible: false,
        },
        rightPriceScale: {
            borderColor: '#005f14',
        },
    });

    // Candlestick series
    candleSeries = candlestickChart.addCandlestickSeries({
        upColor: '#00ff41',
        downColor: '#ff3355',
        borderUpColor: '#00ff41',
        borderDownColor: '#ff3355',
        wickUpColor: '#00ff41',
        wickDownColor: '#ff3355',
    });

    // Volume series
    volumeSeries = candlestickChart.addHistogramSeries({
        color: '#00ff41',
        priceFormat: { type: 'volume' },
        priceScaleId: '',
    });
    volumeSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
    });

    // Resize handler
    const ro = new ResizeObserver(() => {
        candlestickChart.applyOptions({
            width: container.clientWidth,
            height: container.clientHeight,
        });
    });
    ro.observe(container);
}

function updateCandles(ticker, candles) {
    if (!candleSeries) return;

    if (!candleBuffers[ticker]) {
        candleBuffers[ticker] = [];
    }

    const buffer = candleBuffers[ticker];

    for (const c of candles) {
        const time = Math.floor(new Date(c.timestamp).getTime() / 1000);
        const candle = {
            time: time,
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close,
        };

        // Update or append
        const last = buffer[buffer.length - 1];
        if (last && last.time === time) {
            buffer[buffer.length - 1] = candle;
        } else {
            buffer.push(candle);
        }

        // Keep buffer size
        if (buffer.length > 100) buffer.shift();
    }

    if (ticker === activeTicker) {
        candleSeries.setData(buffer);
        // Volume data
        const volData = candles.map(c => ({
            time: Math.floor(new Date(c.timestamp).getTime() / 1000),
            value: c.volume,
            color: c.close >= c.open ? 'rgba(0,255,65,0.3)' : 'rgba(255,51,85,0.3)',
        }));
        volumeSeries.setData(volData);
    }
}

function switchTicker(ticker) {
    activeTicker = ticker;
    document.getElementById('active-ticker-label').textContent = ticker.replace('NSE:', '');

    if (candleBuffers[ticker] && candleSeries) {
        candleSeries.setData(candleBuffers[ticker]);
    }
}

function updatePriceHeader(ticker, price, change) {
    if (ticker !== activeTicker) return;
    const priceEl = document.getElementById('ticker-price');
    const changeEl = document.getElementById('ticker-change');

    priceEl.textContent = price.toFixed(2);
    priceEl.className = change >= 0 ? 'price-up' : 'price-down';

    const sign = change >= 0 ? '+' : '';
    changeEl.textContent = `${sign}${change.toFixed(2)} (${sign}${(change/price*100).toFixed(2)}%)`;
    changeEl.className = change >= 0 ? 'price-up' : 'price-down';
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    initCandlestick('candlestick-chart');
});
