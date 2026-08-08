"""
Agent-NEE FinAI — Static Chart Generation (matplotlib)
Post-market PNG reports: accuracy trend, pred vs actual, latency trend.
"""

import logging
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config

logger = logging.getLogger("agent_nee")

plt.rcParams.update({
    'figure.facecolor': '#0a0a0a',
    'axes.facecolor': '#111111',
    'axes.edgecolor': '#00ff41',
    'axes.labelcolor': '#00ff41',
    'text.color': '#00ff41',
    'xtick.color': '#00aa2a',
    'ytick.color': '#00aa2a',
    'grid.color': '#005f14',
    'grid.alpha': 0.3,
    'font.family': 'monospace',
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
})


def generate_accuracy_trend(output_path: Path = None):
    """Generate rolling accuracy trend chart."""
    output_path = output_path or config.CHARTS_DIR / "accuracy_trend.png"

    buffer_path = config.REPLAY_DIR / "replay_buffer.parquet"
    if not buffer_path.exists():
        logger.warning("No replay buffer for accuracy chart")
        return

    df = pd.read_parquet(buffer_path)
    if df.empty or "prediction_accuracy" not in df.columns:
        return

    resolved = df[df["actual_direction"].notna()]
    if resolved.empty:
        return

    resolved["correct"] = resolved["prediction_accuracy"].astype(float)
    rolling_acc = resolved["correct"].rolling(window=20, min_periods=1).mean() * 100

    fig, ax = plt.subplots()
    ax.plot(rolling_acc.values, color='#00ff41', linewidth=1.5)
    ax.fill_between(range(len(rolling_acc)), rolling_acc.values, alpha=0.1, color='#00ff41')
    ax.set_ylabel('Accuracy %')
    ax.set_xlabel('Prediction #')
    ax.set_title('Rolling Accuracy Trend (20-period)', color='#00ff41', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='#ff3355', linestyle='--', alpha=0.5, label='Random baseline')
    ax.legend(facecolor='#111111', edgecolor='#005f14')
    ax.grid(True)

    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Accuracy trend saved to {output_path}")


def generate_pred_vs_actual(output_path: Path = None):
    """Generate prediction vs actual scatter plot."""
    output_path = output_path or config.CHARTS_DIR / "pred_vs_actual.png"

    buffer_path = config.REPLAY_DIR / "replay_buffer.parquet"
    if not buffer_path.exists():
        return

    df = pd.read_parquet(buffer_path)
    if df.empty or "prediction_return_pct" not in df.columns:
        return

    resolved = df[df["actual_return_pct"].notna()].copy()
    if resolved.empty:
        return

    fig, ax = plt.subplots()

    correct = resolved[resolved["prediction_accuracy"] == True]
    wrong = resolved[resolved["prediction_accuracy"] == False]

    ax.scatter(correct["prediction_return_pct"], correct["actual_return_pct"],
               c='#00ff41', s=20, alpha=0.6, label='Correct')
    ax.scatter(wrong["prediction_return_pct"], wrong["actual_return_pct"],
               c='#ff3355', s=20, alpha=0.6, label='Wrong')

    lim = max(abs(resolved["prediction_return_pct"].max()),
              abs(resolved["actual_return_pct"].max())) * 1.1
    ax.plot([-lim, lim], [-lim, lim], '--', color='#005f14', alpha=0.5)

    ax.set_xlabel('Predicted Return %')
    ax.set_ylabel('Actual Return %')
    ax.set_title('Prediction vs Actual', color='#00ff41', fontweight='bold')
    ax.legend(facecolor='#111111', edgecolor='#005f14')
    ax.grid(True)

    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Pred vs actual saved to {output_path}")


def generate_latency_trend(latency_history: list = None, output_path: Path = None):
    """Generate latency trend chart from history list or log file."""
    output_path = output_path or config.CHARTS_DIR / "latency_trend.png"

    if not latency_history:
        logger.info("No latency data for chart")
        return

    fig, ax = plt.subplots()
    inference = [d.get("inference_avg_ms", 0) for d in latency_history]
    total = [d.get("total_cycle_avg_ms", 0) for d in latency_history]

    ax.plot(inference, color='#00ff41', linewidth=1.5, label='Inference')
    ax.plot(total, color='#00aaff', linewidth=1.5, label='Total Cycle')
    ax.set_ylabel('Latency (ms)')
    ax.set_xlabel('Cycle #')
    ax.set_title('Latency Trend', color='#00ff41', fontweight='bold')
    ax.legend(facecolor='#111111', edgecolor='#005f14')
    ax.grid(True)

    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Latency trend saved to {output_path}")


def generate_all(date_str: str = None, latency_history: list = None):
    """Generate all static charts."""
    logger.info("Generating static charts...")
    generate_accuracy_trend()
    generate_pred_vs_actual()
    generate_latency_trend(latency_history)
    logger.info("Static charts complete")
