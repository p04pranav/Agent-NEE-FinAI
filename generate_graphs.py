#!/usr/bin/env python3
"""Generate performance charts from actual backtest replay buffer data."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

DARK_BG = '#0a0a0a'
DARK_GREEN = '#00ff41'
DARK_DIM = '#00aa2a'
DARK_RED = '#ff3355'
DARK_AMBER = '#ffb000'
DARK_GRID = '#005f14'

LIGHT_BG = 'white'
LIGHT_BLUE = '#2563eb'
LIGHT_RED = '#dc2626'
LIGHT_GRID = '#e5e7eb'

OUT = 'visuals'
os.makedirs(OUT, exist_ok=True)


def apply_dark_style(fig, ax):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors='#cccccc', labelsize=10)
    ax.xaxis.label.set_color('#cccccc')
    ax.yaxis.label.set_color('#cccccc')
    ax.title.set_color('#cccccc')
    for spine in ax.spines.values():
        spine.set_color('#333333')
    ax.grid(True, color=DARK_GRID, linewidth=0.5, alpha=0.7)
    plt.rcParams['font.family'] = 'monospace'


def apply_light_style(fig, ax):
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)
    ax.tick_params(colors='#333333', labelsize=10)
    ax.xaxis.label.set_color('#333333')
    ax.yaxis.label.set_color('#333333')
    ax.title.set_color('#111111')
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
    ax.grid(True, color=LIGHT_GRID, linewidth=0.5, alpha=0.7)
    plt.rcParams['font.family'] = 'serif'


def load_data():
    path = 'data/replay_buffer/replay_buffer.parquet'
    df = pd.read_parquet(path)
    df['prediction_accuracy'] = df['prediction_accuracy'].astype(bool)
    df['ticker_short'] = df['ticker'].str.replace('NSE:', '')
    return df


# ── 1. Accuracy Over Time ──────────────────────────────────────────────────

def gen_accuracy_over_time(df):
    df_sorted = df.sort_values('timestamp').copy()
    df_sorted['pred_idx'] = range(len(df_sorted))

    window = max(20, len(df_sorted) // 15)
    rolling_acc = df_sorted['prediction_accuracy'].rolling(window, min_periods=5).mean() * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    apply_dark_style(fig, ax)
    ax.plot(df_sorted['pred_idx'], rolling_acc, color=DARK_GREEN, linewidth=2, zorder=3)
    ax.axhline(y=50, color=DARK_RED, linestyle='--', linewidth=1.5, label='Random Baseline (50%)', zorder=2)
    ax.fill_between(df_sorted['pred_idx'], rolling_acc, 50,
                     where=(rolling_acc >= 50), color=DARK_GREEN, alpha=0.1)
    ax.fill_between(df_sorted['pred_idx'], rolling_acc, 50,
                     where=(rolling_acc < 50), color=DARK_RED, alpha=0.1)
    ax.set_xlabel('Prediction Index')
    ax.set_ylabel(f'Rolling {window}-Pred Accuracy (%)')
    ax.set_title('Prediction Accuracy Over Time (Backtest)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(facecolor=DARK_BG, edgecolor='#333333', labelcolor='#cccccc', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{OUT}/accuracy_over_time.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [1/8] accuracy_over_time.png")


# ── 2. Agent Comparison ────────────────────────────────────────────────────

def gen_agent_comparison(df):
    conf_groups = df.groupby('prediction_confidence')['prediction_accuracy'].agg(['mean', 'count'])
    conf_groups = conf_groups.reindex(['HIGH', 'MED', 'LOW']).dropna()
    conf_groups['mean'] *= 100

    labels = [f"{idx}\n(n={int(row['count'])})" for idx, row in conf_groups.iterrows()]
    values = conf_groups['mean'].values
    colors = ['#16a34a', DARK_AMBER, LIGHT_RED][:len(values)]

    fig, ax = plt.subplots(figsize=(10, 6))
    apply_dark_style(fig, ax)
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor='#333333', linewidth=1, zorder=3)
    ax.axhline(y=50, color=DARK_RED, linestyle='--', linewidth=1.5, label='Random Baseline', zorder=2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', color='#cccccc', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by Confidence Level (Backtest)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(facecolor=DARK_BG, edgecolor='#333333', labelcolor='#cccccc', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{OUT}/agent_comparison.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [2/8] agent_comparison.png")


# ── 3. Cumulative Return ───────────────────────────────────────────────────

def gen_cumulative_return(df):
    df_sorted = df.sort_values('timestamp').copy()
    returns = df_sorted['actual_return_pct'].values / 100
    cum = np.cumsum(returns) * 100
    idx = np.arange(len(cum))

    fig, ax = plt.subplots(figsize=(10, 6))
    apply_dark_style(fig, ax)
    ax.plot(idx, cum, color=DARK_GREEN, linewidth=2, zorder=3)
    ax.fill_between(idx, cum, 0, where=(cum >= 0), color=DARK_GREEN, alpha=0.1)
    ax.fill_between(idx, cum, 0, where=(cum < 0), color=DARK_RED, alpha=0.1)
    ax.axhline(y=0, color='#666666', linestyle='-', linewidth=1, zorder=2)
    ax.set_xlabel('Prediction Index')
    ax.set_ylabel('Cumulative Return (%)')
    ax.set_title('Cumulative Return (Backtest)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUT}/cumulative_return.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [3/8] cumulative_return.png")


# ── 4. Training Loss ──────────────────────────────────────────────────────

def gen_training_loss(df):
    steps = np.arange(0, 151, 5)
    train_loss = 0.85 * np.exp(-0.015 * steps) + 0.42 + np.random.normal(0, 0.008, len(steps))
    val_loss = 0.88 * np.exp(-0.014 * steps) + 0.45 + np.random.normal(0, 0.012, len(steps))
    train_loss = np.clip(train_loss, 0.40, 0.90)
    val_loss = np.clip(val_loss, 0.43, 0.93)

    fig, ax = plt.subplots(figsize=(10, 6))
    apply_light_style(fig, ax)
    ax.plot(steps, train_loss, color=LIGHT_BLUE, linewidth=2, label='Train Loss', zorder=3)
    ax.plot(steps, val_loss, color=LIGHT_RED, linewidth=2, label='Validation Loss', zorder=3)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title('LoRA SFT Training Loss (Placeholder)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f'{OUT}/training_loss.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [4/8] training_loss.png")


# ── 5. Confidence Calibration ──────────────────────────────────────────────

def gen_confidence_calibration(df):
    conf_groups = df.groupby('prediction_confidence')['prediction_accuracy'].agg(['mean', 'count', 'std'])
    conf_groups = conf_groups.reindex(['HIGH', 'MED', 'LOW']).dropna()
    conf_groups['mean'] *= 100
    conf_groups['stderr'] = (conf_groups['std'] / conf_groups['count'].pow(0.5)) * 100

    labels = conf_groups.index.tolist()
    values = conf_groups['mean'].values
    errors = conf_groups['stderr'].values
    colors = ['#16a34a', '#ca8a04', LIGHT_RED]

    fig, ax = plt.subplots(figsize=(10, 6))
    apply_light_style(fig, ax)
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='#d1d5db',
                  linewidth=1, yerr=errors, capsize=8, error_kw={'linewidth': 1.5, 'color': '#6b7280'}, zorder=3)
    ax.axhline(y=50, color='#9ca3af', linestyle='--', linewidth=1.5, label='Random Baseline', zorder=2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#111111')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Prediction Accuracy by Confidence Level (Backtest)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f'{OUT}/confidence_calibration.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [5/8] confidence_calibration.png")


# ── 6. Accuracy by Ticker ──────────────────────────────────────────────────

def gen_accuracy_by_ticker(df):
    ticker_acc = df.groupby('ticker_short')['prediction_accuracy'].agg(['mean', 'count'])
    ticker_acc['mean'] *= 100
    ticker_acc = ticker_acc.sort_values('mean')

    tickers = ticker_acc.index.tolist()
    values = ticker_acc['mean'].values
    counts = ticker_acc['count'].values

    norm_vals = [(v - 20) / 60 for v in values]
    cmap_colors = []
    for nv in norm_vals:
        nv = max(0, min(1, nv))
        r = int(0xdc * (1 - nv) + 0x16 * nv)
        g = int(0x26 * (1 - nv) + 0xa3 * nv)
        b = int(0x26 * (1 - nv) + 0x4a * nv)
        cmap_colors.append(f'#{r:02x}{g:02x}{b:02x}')

    fig, ax = plt.subplots(figsize=(10, 6))
    apply_light_style(fig, ax)
    bars = ax.barh(tickers, values, color=cmap_colors, edgecolor='#d1d5db', linewidth=1, height=0.6, zorder=3)
    ax.axvline(x=50, color='#9ca3af', linestyle='--', linewidth=1.5, label='Random Baseline', zorder=2)
    for bar, val, cnt in zip(bars, values, counts):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}% (n={cnt})', ha='left', va='center', fontsize=9, fontweight='bold', color='#333333')
    ax.set_xlabel('Directional Accuracy (%)')
    ax.set_title('Directional Accuracy by Ticker (Backtest)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.legend(fontsize=10, framealpha=0.9, loc='lower right')
    fig.tight_layout()
    fig.savefig(f'{OUT}/accuracy_by_ticker.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [6/8] accuracy_by_ticker.png")


# ── 7. Predicted vs Actual ─────────────────────────────────────────────────

def gen_pred_vs_actual(df):
    predicted = df['prediction_return_pct'].values
    actual = df['actual_return_pct'].values

    same_dir = np.sign(predicted) == np.sign(actual)
    colors = np.where(same_dir, '#16a34a', LIGHT_RED)

    mask = ~(np.isnan(predicted) | np.isnan(actual))
    predicted = predicted[mask]
    actual = actual[mask]
    colors = colors[mask]

    r = np.corrcoef(predicted, actual)[0, 1] if len(predicted) > 2 else 0

    fig, ax = plt.subplots(figsize=(10, 6))
    apply_light_style(fig, ax)
    ax.scatter(predicted, actual, c=colors, s=18, alpha=0.6, edgecolors='none', zorder=2)
    lim = max(abs(predicted).max(), abs(actual).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color='#9ca3af', linestyle='--', linewidth=1.5, label='Perfect Prediction', zorder=3)
    ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontweight='bold', color='#111111',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#d1d5db', alpha=0.9))
    ax.set_xlabel('Predicted Return (%)')
    ax.set_ylabel('Actual Return (%)')
    ax.set_title('Predicted vs Actual Returns (Backtest)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f'{OUT}/pred_vs_actual.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [7/8] pred_vs_actual.png")


# ── 8. Latency Profile ────────────────────────────────────────────────────

def gen_latency_profile(df):
    latencies = df['inference_latency_ms'].dropna().values

    fig, ax = plt.subplots(figsize=(10, 6))
    apply_light_style(fig, ax)
    ax.hist(latencies, bins=30, color=LIGHT_BLUE, alpha=0.7, edgecolor='white', linewidth=0.5, zorder=3)
    ax.axvline(x=np.median(latencies), color=LIGHT_RED, linestyle='--', linewidth=1.5,
               label=f'Median: {np.median(latencies):.0f}ms', zorder=2)
    ax.set_xlabel('Inference Latency (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Inference Latency Distribution (Backtest)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f'{OUT}/latency_profile.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [8/8] latency_profile.png")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading backtest data...")
    df = load_data()
    print(f"  {len(df)} predictions loaded")
    print(f"  Overall accuracy: {df['prediction_accuracy'].mean()*100:.1f}%")
    print(f"\nGenerating charts...")
    gen_accuracy_over_time(df)
    gen_agent_comparison(df)
    gen_cumulative_return(df)
    gen_training_loss(df)
    gen_confidence_calibration(df)
    gen_accuracy_by_ticker(df)
    gen_pred_vs_actual(df)
    gen_latency_profile(df)
    print(f"\nDone. {len(os.listdir(OUT))} PNGs in {OUT}/")
