#!/usr/bin/env python3
"""Generate synthetic OHLCV CSV files for 10 NSE stock tickers."""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "csv")

TICKERS = {
    "RELIANCE": 2845,
    "TCS": 3920,
    "HDFCBANK": 1650,
    "INFY": 1480,
    "ICICIBANK": 1120,
    "SBIN": 780,
    "BHARTIARTL": 1250,
    "ITC": 480,
    "WIPRO": 510,
    "AXISBANK": 1080,
}

ROWS = 1000
INTERVAL_MIN = 5
START = datetime(2024, 1, 2, 9, 25, 0)


def generate_timestamps(n: int, start: datetime, interval_min: int) -> list[datetime]:
    """Generate *n* timestamps, 5-min bars, 09:25-15:15 Mon-Fri."""
    times = []
    current_day = start
    while len(times) < n:
        bar_time = current_day
        for _ in range(71):  # 71 bars per day (09:25 to 15:15 inclusive)
            if len(times) >= n:
                break
            times.append(bar_time)
            bar_time += timedelta(minutes=interval_min)
        # jump to next weekday
        current_day = current_day + timedelta(days=1)
        while current_day.weekday() >= 5:  # skip Sat/Sun
            current_day += timedelta(days=1)
        current_day = current_day.replace(hour=9, minute=25, second=0)
    return times


def generate_ohlcv(base_price: float, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Random-walk OHLCV with realistic constraints."""
    returns = rng.normal(0, 0.0015, n)  # ~0.15 % per bar
    close = base_price * np.cumprod(1 + returns)

    open_ = np.empty(n)
    open_[0] = base_price
    open_[1:] = close[:-1]

    noise = rng.uniform(0.0005, 0.003, n)
    high = np.maximum(open_, close) * (1 + noise)
    low = np.minimum(open_, close) * (1 - noise)

    volume = rng.normal(1_000_000, 200_000, n).clip(200_000).astype(int)

    return pd.DataFrame({
        "open": np.round(open_, 2),
        "high": np.round(high, 2),
        "low": np.round(low, 2),
        "close": np.round(close, 2),
        "volume": volume,
    })


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(42)
    timestamps = generate_timestamps(ROWS, START, INTERVAL_MIN)

    print(f"Generating {ROWS} rows for {len(TICKERS)} tickers into {OUTPUT_DIR}/\n")

    for ticker, base in TICKERS.items():
        df = generate_ohlcv(base, ROWS, rng)
        df.insert(0, "timestamp", [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps])
        path = os.path.join(OUTPUT_DIR, f"{ticker}.csv")
        df.to_csv(path, index=False)
        print(f"  {ticker:>12s}  base={base:>5d}  rows={len(df)}  -> {path}")

    print(f"\nDone. {len(TICKERS)} CSV files written.")


if __name__ == "__main__":
    main()
