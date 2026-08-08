/* Agent-NEE — AI Performance Charts (Chart.js) */

let accuracyChart = null;
let scatterChart = null;
let latencyChart = null;

const MAX_DATA_POINTS = 50;

// Chart.js terminal theme defaults
Chart.defaults.color = '#00aa2a';
Chart.defaults.font.family = "'JetBrains Mono', monospace";
Chart.defaults.font.size = 10;

const GRID_COLOR = '#005f14';
const GRID_OPTIONS = {
    color: GRID_COLOR,
    drawBorder: false,
};

function initAccuracyChart(canvasId) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    accuracyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Accuracy %',
                data: [],
                borderColor: '#00ff41',
                backgroundColor: 'rgba(0,255,65,0.1)',
                borderWidth: 1.5,
                pointRadius: 0,
                fill: true,
                tension: 0.3,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { display: false },
                y: {
                    min: 0, max: 100,
                    grid: GRID_OPTIONS,
                    ticks: { callback: v => v + '%' },
                },
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#000',
                    borderColor: '#00ff41',
                    borderWidth: 1,
                },
            },
        },
    });
}

function initScatterChart(canvasId) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    scatterChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Correct',
                data: [],
                backgroundColor: 'rgba(0,255,65,0.6)',
                pointRadius: 3,
            }, {
                label: 'Wrong',
                data: [],
                backgroundColor: 'rgba(255,51,85,0.6)',
                pointRadius: 3,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'Predicted %' },
                    grid: GRID_OPTIONS,
                },
                y: {
                    title: { display: true, text: 'Actual %' },
                    grid: GRID_OPTIONS,
                },
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: { boxWidth: 8, padding: 4 },
                },
                tooltip: {
                    backgroundColor: '#000',
                    borderColor: '#00ff41',
                    borderWidth: 1,
                },
            },
        },
    });
}

function initLatencyChart(canvasId) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    latencyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Inference',
                data: [],
                borderColor: '#00ff41',
                borderWidth: 1.5,
                pointRadius: 0,
            }, {
                label: 'Total',
                data: [],
                borderColor: '#00aaff',
                borderWidth: 1.5,
                pointRadius: 0,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { display: false },
                y: {
                    grid: GRID_OPTIONS,
                    ticks: { callback: v => v + 'ms' },
                },
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: { boxWidth: 8, padding: 4 },
                },
                tooltip: {
                    backgroundColor: '#000',
                    borderColor: '#00ff41',
                    borderWidth: 1,
                },
            },
        },
    });
}

function updateCharts(wsData) {
    const now = new Date().toLocaleTimeString();

    // Accuracy
    if (accuracyChart && wsData.status && wsData.status.accuracy !== undefined) {
        const acc = wsData.status.accuracy;
        accuracyChart.data.labels.push(now);
        accuracyChart.data.datasets[0].data.push(acc);
        if (accuracyChart.data.labels.length > MAX_DATA_POINTS) {
            accuracyChart.data.labels.shift();
            accuracyChart.data.datasets[0].data.shift();
        }
        accuracyChart.update('none');
    }

    // Latency
    if (latencyChart && wsData.latency) {
        latencyChart.data.labels.push(now);
        latencyChart.data.datasets[0].data.push(wsData.latency.inference_avg_ms || 0);
        latencyChart.data.datasets[1].data.push(wsData.latency.total_cycle_avg_ms || 0);
        if (latencyChart.data.labels.length > MAX_DATA_POINTS) {
            latencyChart.data.labels.shift();
            latencyChart.data.datasets[0].data.shift();
            latencyChart.data.datasets[1].data.shift();
        }
        latencyChart.update('none');
    }

    // Scatter (from predictions)
    if (scatterChart && wsData.predictions) {
        for (const [ticker, pred] of Object.entries(wsData.predictions)) {
            if (pred.actual_return_pct !== undefined) {
                const point = {
                    x: pred.target_return_pct || 0,
                    y: pred.actual_return_pct,
                };
                const dataset = pred.prediction_accuracy ? 0 : 1;
                scatterChart.data.datasets[dataset].data.push(point);
                // Keep limited points
                if (scatterChart.data.datasets[dataset].data.length > 100) {
                    scatterChart.data.datasets[dataset].data.shift();
                }
            }
        }
        scatterChart.update('none');
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    initAccuracyChart('accuracy-chart');
    initScatterChart('scatter-chart');
    initLatencyChart('latency-chart');
});
