"""
Agent-NEE FinAI — Two-Phase Parquet Ledger
UUID-based prediction logging with deferred accuracy resolution.
"""

import json
import uuid
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config
import utils
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("agent_nee")

# Phase 1 columns (written at prediction time)
PHASE1_COLUMNS = [
    "row_id", "timestamp", "ticker", "cycle_id",
    "open", "high", "low", "close", "volume",
    "vwap", "rsi_14", "macd", "macd_signal",
    "bb_upper", "bb_lower", "bb_middle", "atr_14",
    "prediction_direction", "prediction_return_pct", "prediction_confidence",
    "agent_contributions", "band_room_id",
    "model_version", "inference_latency_ms",
    "mode", "data_source",
]

# Phase 2 columns (added at resolution time)
PHASE2_COLUMNS = [
    "actual_direction", "actual_return_pct",
    "resolution_timestamp", "prediction_accuracy",
    "resolution_price", "reward", "used_in_training",
]

ALL_COLUMNS = PHASE1_COLUMNS + PHASE2_COLUMNS


def _get_ticker_path(ticker: str) -> Path:
    """Get Parquet file path for a ticker."""
    safe_name = ticker.replace(":", "_")
    return config.DAILY_DIR / f"{safe_name}.parquet"


def _load_ticker_df(ticker: str) -> pd.DataFrame:
    """Load existing Parquet for a ticker, or create empty DataFrame."""
    path = _get_ticker_path(ticker)
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=ALL_COLUMNS)


def _save_ticker_df(ticker: str, df: pd.DataFrame):
    """Save DataFrame to Parquet."""
    path = _get_ticker_path(ticker)
    df.to_parquet(path, index=False)


def phase1_write(row_data: dict) -> str:
    """Write Phase 1 prediction data to ledger.
    Returns: UUID string for the row.
    """
    row_id = str(uuid.uuid4())
    row_data["row_id"] = row_id

    # Ensure all Phase 2 columns exist as None/NaN
    for col in PHASE2_COLUMNS:
        if col not in row_data:
            row_data[col] = None

    ticker = row_data.get("ticker")
    if not ticker:
        raise utils.LedgerError("Missing ticker in row_data")

    # Serialize agent_contributions if it's a list
    if isinstance(row_data.get("agent_contributions"), list):
        row_data["agent_contributions"] = json.dumps(row_data["agent_contributions"])

    # Load, append, save
    df = _load_ticker_df(ticker)
    new_row = pd.DataFrame([row_data])
    df = pd.concat([df, new_row], ignore_index=True)
    _save_ticker_df(ticker, df)

    logger.debug(f"Phase 1 write: {ticker} row_id={row_id}")
    return row_id


def resolve_phase2(current_timestamp: pd.Timestamp, current_prices: dict[str, float] = None) -> int:
    """Resolve all pending Phase 1 rows whose 5-minute window has elapsed.
    
    Args:
        current_timestamp: Current time for resolution
        current_prices: Dict mapping ticker -> current price. If None, uses entry price (degraded mode).
    
    Returns: Count of resolved rows.
    """
    resolved_count = 0
    current_prices = current_prices or {}

    for ticker in config.TICKER_SYMBOLS:
        df = _load_ticker_df(ticker)
        if df.empty:
            continue

        # Find unresolved rows (Phase 2 columns are NaN/None)
        unresolved_mask = df["actual_direction"].isna()
        unresolved = df[unresolved_mask]

        if unresolved.empty:
            continue

        for idx, row in unresolved.iterrows():
            ts = pd.to_datetime(row["timestamp"])
            elapsed = (current_timestamp - ts).total_seconds()

            if elapsed < config.INTERVAL_SECONDS:
                continue  # Not ready yet

            # Get fresh current price
            entry_price = float(row["close"])
            current_price = current_prices.get(ticker, entry_price)

            # Calculate actual return
            if entry_price > 0:
                actual_return = (current_price - entry_price) / entry_price * 100
            else:
                actual_return = 0.0

            # Determine direction
            if actual_return > config.SIDEWAYS_THRESHOLD_PCT:
                actual_direction = "UP"
            elif actual_return < -config.SIDEWAYS_THRESHOLD_PCT:
                actual_direction = "DOWN"
            else:
                actual_direction = "SIDEWAYS"

            # Check accuracy
            pred_direction = str(row.get("prediction_direction", ""))
            accuracy = pred_direction == actual_direction

            # Write Phase 2 columns
            df.at[idx, "actual_direction"] = actual_direction
            df.at[idx, "actual_return_pct"] = round(actual_return, 4)
            df.at[idx, "resolution_timestamp"] = current_timestamp
            df.at[idx, "prediction_accuracy"] = accuracy
            df.at[idx, "resolution_price"] = round(current_price, 2)
            df.at[idx, "reward"] = 1.0 if accuracy else 0.0
            df.at[idx, "used_in_training"] = False

            resolved_count += 1

        _save_ticker_df(ticker, df)

    if resolved_count > 0:
        logger.info(f"Phase 2 resolved {resolved_count} rows")

    return resolved_count


def get_completed_rows() -> pd.DataFrame:
    """Get all Phase 2-resolved rows across all tickers (for training eligibility)."""
    all_rows = []
    for ticker in config.TICKER_SYMBOLS:
        df = _load_ticker_df(ticker)
        if df.empty:
            continue
        resolved = df[df["actual_direction"].notna()]
        if not resolved.empty:
            all_rows.append(resolved)

    if not all_rows:
        return pd.DataFrame(columns=ALL_COLUMNS)
    return pd.concat(all_rows, ignore_index=True)


def get_agent_contributions(row_id: str) -> list:
    """Get agent contributions JSON for a given row."""
    for ticker in config.TICKER_SYMBOLS:
        df = _load_ticker_df(ticker)
        if df.empty:
            continue
        match = df[df["row_id"] == row_id]
        if not match.empty:
            raw = match.iloc[0].get("agent_contributions", "[]")
            if isinstance(raw, str):
                return json.loads(raw)
            return raw
    return []


def cleanup_orphans() -> int:
    """Resolve or mark orphaned Phase 1 rows on startup."""
    now = pd.Timestamp.now(tz=IST)
    cleaned = 0

    for ticker in config.TICKER_SYMBOLS:
        df = _load_ticker_df(ticker)
        if df.empty:
            continue

        unresolved_mask = df["actual_direction"].isna()
        unresolved = df[unresolved_mask]

        for idx, row in unresolved.iterrows():
            ts = pd.to_datetime(row["timestamp"])
            elapsed = (now - ts).total_seconds()

            if elapsed > 3600:  # > 1 hour — mark expired
                df.at[idx, "actual_direction"] = "EXPIRED"
                df.at[idx, "actual_return_pct"] = 0.0
                df.at[idx, "resolution_timestamp"] = now
                df.at[idx, "prediction_accuracy"] = False
                df.at[idx, "resolution_price"] = 0.0
                df.at[idx, "reward"] = 0.0
                df.at[idx, "used_in_training"] = False
                cleaned += 1
            elif elapsed > config.INTERVAL_SECONDS:
                # Resolve with entry price (best available — no fresh data on startup)
                entry_price = float(row["close"])
                df.at[idx, "actual_direction"] = "SIDEWAYS"
                df.at[idx, "actual_return_pct"] = 0.0
                df.at[idx, "resolution_timestamp"] = now
                df.at[idx, "prediction_accuracy"] = False
                df.at[idx, "resolution_price"] = entry_price
                df.at[idx, "reward"] = 0.0
                df.at[idx, "used_in_training"] = False
                cleaned += 1

        _save_ticker_df(ticker, df)

    if cleaned > 0:
        logger.info(f"Cleaned up {cleaned} orphaned rows")
    return cleaned


def update_replay_buffer():
    """Append resolved rows to replay buffer, trim to 30 trading days."""
    completed = get_completed_rows()
    if completed.empty:
        return

    buffer_path = config.REPLAY_DIR / "replay_buffer.parquet"

    if buffer_path.exists():
        existing = pd.read_parquet(buffer_path)
        combined = pd.concat([existing, completed], ignore_index=True)
    else:
        combined = completed

    # Deduplicate by row_id
    combined = combined.drop_duplicates(subset=["row_id"], keep="last")

    # Trim to 30 trading days (~45 calendar days)
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"])
        # Normalize tz: strip tz info for comparison
        if combined["timestamp"].dt.tz is not None:
            combined["timestamp"] = combined["timestamp"].dt.tz_localize(None)
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=45)
        combined = combined[combined["timestamp"] >= cutoff]

    combined.to_parquet(buffer_path, index=False)
    logger.info(f"Replay buffer updated: {len(combined)} rows")


def get_replay_buffer() -> pd.DataFrame:
    """Load the replay buffer for training."""
    buffer_path = config.REPLAY_DIR / "replay_buffer.parquet"
    if buffer_path.exists():
        return pd.read_parquet(buffer_path)
    return pd.DataFrame(columns=ALL_COLUMNS)


def flush():
    """Flush any pending state (no-op for Parquet as writes are immediate)."""
    logger.debug("Ledger flush called (Parquet writes are immediate)")
