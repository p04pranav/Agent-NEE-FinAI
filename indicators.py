"""
Agent-NEE FinAI — Technical Indicators Engine
Computes VWAP, RSI, MACD, Bollinger Bands, ATR, SMA via pandas-ta.
"""

import logging
import pandas as pd
import pandas_ta as ta

import config

logger = logging.getLogger("agent_nee")


def compute_indicators(candles: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators on OHLCV DataFrame.
    
    Input: DataFrame with columns [timestamp, open, high, low, close, volume]
    Output: Same DataFrame with indicator columns appended.
    
    Indicators: VWAP (daily reset), RSI(14), MACD(12,26,9), BB(20,2), ATR(14), SMA(50), SMA(200)
    """
    df = candles.copy()

    # Ensure proper column names (lowercase)
    df.columns = [c.lower() for c in df.columns]

    # Need minimum rows for indicators
    if len(df) < config.MIN_INDICATOR_ROWS:
        logger.warning(f"Insufficient data ({len(df)} rows < {config.MIN_INDICATOR_ROWS}). Indicators will be NaN.")
        for col in ["vwap", "rsi_14", "macd", "macd_signal", "bb_upper", "bb_middle", "bb_lower", "atr_14", "sma_50", "sma_200"]:
            df[col] = float("nan")
        return df

    # VWAP — Volume Weighted Average Price with DAILY RESET
    # Typical price = (H + L + C) / 3
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["_date"] = df["timestamp"].dt.date
    else:
        df["_date"] = 0  # Single day

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical_price * df["volume"]

    # Group by date for daily reset
    df["vwap"] = (
        tp_vol.groupby(df["_date"]).cumsum()
        / df["volume"].groupby(df["_date"]).cumsum()
    ).round(2)
    df.drop(columns=["_date"], inplace=True)

    # RSI — 14-period
    rsi = ta.rsi(df["close"], length=14)
    df["rsi_14"] = rsi.round(2) if rsi is not None else float("nan")

    # MACD — 12, 26, 9
    macd_result = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd_result is not None and len(macd_result.columns) >= 2:
        df["macd"] = macd_result.iloc[:, 0].round(2)
        df["macd_signal"] = macd_result.iloc[:, 1].round(2)
    else:
        df["macd"] = float("nan")
        df["macd_signal"] = float("nan")

    # Bollinger Bands — 20-period, 2 std
    bb = ta.bbands(df["close"], length=20, std=2)
    if bb is not None and len(bb.columns) >= 3:
        df["bb_lower"] = bb.iloc[:, 0].round(2)
        df["bb_middle"] = bb.iloc[:, 1].round(2)
        df["bb_upper"] = bb.iloc[:, 2].round(2)
    else:
        df["bb_upper"] = float("nan")
        df["bb_middle"] = float("nan")
        df["bb_lower"] = float("nan")

    # ATR — 14-period
    atr = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["atr_14"] = atr.round(2) if atr is not None else float("nan")

    # SMA — 50 and 200 period (if enough data)
    if len(df) >= 50:
        sma50 = ta.sma(df["close"], length=50)
        df["sma_50"] = sma50.round(2) if sma50 is not None else float("nan")
    else:
        df["sma_50"] = float("nan")

    if len(df) >= 200:
        sma200 = ta.sma(df["close"], length=200)
        df["sma_200"] = sma200.round(2) if sma200 is not None else float("nan")
    else:
        df["sma_200"] = float("nan")

    return df


def format_market_data_for_agent(ticker: str, row: pd.Series) -> str:
    """Format the latest indicators row as a readable string for LLM prompts."""
    def _fmt(val, decimals=2):
        if pd.isna(val):
            return "N/A"
        return f"{val:.{decimals}f}"

    lines = [
        f"Ticker: {ticker}",
        f"Price: O={_fmt(row.get('open'))} H={_fmt(row.get('high'))} L={_fmt(row.get('low'))} C={_fmt(row.get('close'))}",
        f"Volume: {_fmt(row.get('volume'), 0)}",
        f"VWAP: {_fmt(row.get('vwap'))}",
        f"RSI(14): {_fmt(row.get('rsi_14'))}",
        f"MACD: {_fmt(row.get('macd'))} (signal: {_fmt(row.get('macd_signal'))})",
        f"Bollinger Bands: {_fmt(row.get('bb_lower'))} - {_fmt(row.get('bb_middle'))} - {_fmt(row.get('bb_upper'))}",
        f"ATR(14): {_fmt(row.get('atr_14'))}",
    ]
    return "\n".join(lines)
