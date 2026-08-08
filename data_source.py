"""
Agent-NEE FinAI — CSV Data Source
Loads OHLCV data from CSV files and serves them cycle-by-cycle.
"""

import logging
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger("agent_nee")

_data: dict[str, pd.DataFrame] = {}
_cursors: dict[str, int] = {}


def _load_csvs() -> None:
    """Load all CSVs from CSV_DIR into _data. Called once on first poll."""
    csv_dir = Path(config.CSV_DIR)
    loaded = 0

    for ticker in config.TICKER_SYMBOLS:
        filename = ticker.replace(":", "_") + ".csv"
        path = csv_dir / filename
        if not path.exists():
            logger.warning(f"CSV not found for {ticker}: {path}")
            continue
        df = pd.read_csv(path)
        _data[ticker] = df
        _cursors[ticker] = 0
        loaded += 1

    logger.info(f"Loaded CSV data for {loaded} tickers from {csv_dir}")


def poll_tickers() -> dict[str, pd.DataFrame]:
    """Return {ticker: DataFrame} for the current cycle.

    Advances cursor by CSV_ROWS_PER_CYCLE.
    If SIMULATION_LOOP and cursor >= len(data), reset to 0.
    If CSV not found for a ticker, skip it.
    """
    if not _data:
        _load_csvs()

    result: dict[str, pd.DataFrame] = {}
    step = config.CSV_ROWS_PER_CYCLE

    for ticker in config.TICKER_SYMBOLS:
        if ticker not in _data:
            continue

        df = _data[ticker]
        cursor = _cursors[ticker]
        chunk = df.iloc[cursor : cursor + step]
        served = len(chunk)

        if served > 0:
            result[ticker] = chunk
            _cursors[ticker] += served
        elif served == 0:
            if config.SIMULATION_LOOP:
                _cursors[ticker] = 0
                logger.info(f"CSV exhausted for {ticker}, looping")
                chunk = df.iloc[0:step]
                result[ticker] = chunk
                _cursors[ticker] = len(chunk)
            else:
                logger.debug(f"CSV exhausted for {ticker}, skipping")

    return result


def load_all() -> dict[str, pd.DataFrame]:
    """Return all loaded CSV data (no cursor advancement). Useful for tests."""
    if not _data:
        _load_csvs()
    return dict(_data)
