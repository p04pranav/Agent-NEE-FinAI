"""
Agent-NEE FinAI — Integration Tests
Covers: config, mock data pipeline, ledger round-trip, Agent SDK, dashboard,
market hours, indicators, predict module, charts, learn module.
"""

import datetime
import json
import math
import os
import random
import tempfile
import threading
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from zoneinfo import ZoneInfo

# Ensure project root is on sys.path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import config
import data_source
import indicators
import ledger
import utils
from agents import LocalBandSDK
from charts import generate_accuracy_trend, generate_latency_trend
from web_server import DashboardState
from learn import balance_classes, format_training_pair
from main import IST, is_trading_day, is_within_trading_hours, is_premarket_time
from predict import PredictionSquad

# ─── Helpers ──────────────────────────────────────────────────────────


def _make_ohlcv_df(n: int = 100, base_price: float = 2500.0) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame with n rows."""
    now = datetime.datetime.now(IST)
    rows = []
    price = base_price
    for i in range(n):
        ts = now - datetime.timedelta(minutes=(n - i) * 5)
        change = random.gauss(0, 0.02) * price
        o = price
        c = price + change
        h = max(o, c) * (1 + abs(random.gauss(0, 0.005)))
        l = min(o, c) * (1 - abs(random.gauss(0, 0.005)))
        vol = int(1_000_000 * (1 + random.gauss(0, 0.3)))
        rows.append({
            "timestamp": ts,
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
            "volume": vol,
        })
        price = c
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Config loads correctly
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConfig:
    """Verify all config constants have expected types and values."""

    def test_paths_are_pathlib(self):
        assert isinstance(config.BASE_DIR, Path)
        assert isinstance(config.LOG_DIR, Path)
        assert isinstance(config.DATA_DIR, Path)
        assert isinstance(config.DAILY_DIR, Path)
        assert isinstance(config.REPLAY_DIR, Path)

    def test_ticker_symbols_list(self):
        assert isinstance(config.TICKER_SYMBOLS, list)
        assert len(config.TICKER_SYMBOLS) == 10
        for t in config.TICKER_SYMBOLS:
            assert t.startswith("NSE:")

    def test_model_name_string(self):
        assert isinstance(config.MODEL_NAME, str)
        assert ":" in config.MODEL_NAME  # e.g. "phi3:mini"

    def test_ollama_url_string(self):
        assert config.OLLAMA_BASE_URL.startswith("http")
        assert config.OLLAMA_API_GENERATE.endswith("/api/generate")

    def test_inference_params_numeric(self):
        assert isinstance(config.INFERENCE_TEMPERATURE, float)
        assert 0 < config.INFERENCE_TEMPERATURE <= 2
        assert isinstance(config.INFERENCE_TOP_P, float)
        assert 0 < config.INFERENCE_TOP_P <= 1
        assert isinstance(config.INFERENCE_MAX_TOKENS, int)
        assert config.INFERENCE_MAX_TOKENS > 0

    def test_interval_constants(self):
        assert config.INTERVAL_MINUTES == 5
        assert config.INTERVAL_SECONDS == 300

    def test_time_windows_format(self):
        for val in [config.PREDICTION_START, config.PREDICTION_END,
                     config.FRIDAY_EARLY_CLOSE, config.PREMARKET_WARMUP_TIME]:
            assert isinstance(val, str)
            h, m = val.split(":")
            assert 0 <= int(h) <= 23
            assert 0 <= int(m) <= 59

    def test_mock_base_prices(self):
        assert isinstance(config.MOCK_BASE_PRICES, dict)
        assert len(config.MOCK_BASE_PRICES) == 10
        for k, v in config.MOCK_BASE_PRICES.items():
            assert k.startswith("NSE:")
            assert isinstance(v, float)
            assert v > 0

    def test_web_config(self):
        assert isinstance(config.WEB_HOST, str)
        assert isinstance(config.WEB_PORT, int)

    def test_training_params(self):
        assert config.LORA_R == 16
        assert config.LORA_ALPHA == 16
        assert config.LORA_DROPOUT == 0.0
        assert isinstance(config.LORA_TARGET_MODULES, list)
        assert config.SFT_LEARNING_RATE > 0
        assert config.SFT_NUM_EPOCHS > 0

    def test_agent_roles(self):
        assert config.N_AGENTS == 3
        assert len(config.AGENT_ROLES_TO_RUN) == 3
        assert config.BAND_ENABLED is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Mock data pipeline — generate → compute_indicators → format
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMockDataPipeline:
    """End-to-end: data_source.load_all() → compute_indicators() → format_market_data_for_agent()"""

    def test_generate_mock_data_returns_all_tickers(self):
        data = data_source.load_all()
        assert isinstance(data, dict)
        assert len(data) == len(config.TICKER_SYMBOLS)
        for ticker in config.TICKER_SYMBOLS:
            assert ticker in data
            assert isinstance(data[ticker], pd.DataFrame)

    def test_mock_data_has_correct_columns(self):
        data = data_source.load_all()
        for ticker, df in data.items():
            for col in ["timestamp", "open", "high", "low", "close", "volume"]:
                assert col in df.columns, f"{ticker} missing column {col}"

    def test_mock_data_row_count(self):
        data = data_source.load_all()
        for ticker, df in data.items():
            assert len(df) == 1000

    def test_mock_data_ohlc_relationships(self):
        """High >= max(open,close) and Low <= min(open,close) for every candle."""
        data = data_source.load_all()
        for ticker, df in data.items():
            for _, row in df.iterrows():
                assert row["high"] >= max(row["open"], row["close"]), f"{ticker}: high < max(o,c)"
                assert row["low"] <= min(row["open"], row["close"]), f"{ticker}: low > min(o,c)"

    def test_compute_indicators_adds_columns(self):
        df = _make_ohlcv_df(100)
        result = indicators.compute_indicators(df)
        expected_cols = {"vwap", "rsi_14", "macd", "macd_signal",
                         "bb_upper", "bb_middle", "bb_lower", "atr_14", "sma_50", "sma_200"}
        assert expected_cols.issubset(set(result.columns))

    def test_compute_indicators_insufficient_rows(self):
        """With < MIN_INDICATOR_ROWS, indicator columns should be NaN."""
        df = _make_ohlcv_df(10)
        result = indicators.compute_indicators(df)
        assert result["rsi_14"].isna().all()

    def test_format_market_data_for_agent(self):
        df = _make_ohlcv_df(100)
        result = indicators.compute_indicators(df)
        latest = result.iloc[-1]
        text = indicators.format_market_data_for_agent("NSE:RELIANCE", latest)
        assert isinstance(text, str)
        assert "NSE:RELIANCE" in text
        assert "RSI" in text
        assert "VWAP" in text
        assert "MACD" in text

    def test_full_pipeline_integration(self):
        """data_source.load_all → compute_indicators → format_market_data_for_agent."""
        data = data_source.load_all()
        ticker = "NSE:RELIANCE"
        df = data[ticker]
        df_ind = indicators.compute_indicators(df)
        latest = df_ind.iloc[-1]
        text = indicators.format_market_data_for_agent(ticker, latest)
        assert len(text) > 50  # Should be a substantial string
        assert "Price:" in text
        assert "Volume:" in text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Ledger round-trip — phase1_write → _load_ticker_df → resolve_phase2
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLedgerRoundTrip:
    """UUID-based round-trip through the two-phase Parquet ledger."""

    @pytest.fixture(autouse=True)
    def _tmp_dirs(self, tmp_path):
        """Redirect DAILY_DIR and REPLAY_DIR to tmp for test isolation."""
        self._orig_daily = config.DAILY_DIR
        self._orig_replay = config.REPLAY_DIR
        config.DAILY_DIR = tmp_path / "daily"
        config.REPLAY_DIR = tmp_path / "replay"
        config.DAILY_DIR.mkdir(parents=True, exist_ok=True)
        config.REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        yield
        config.DAILY_DIR = self._orig_daily
        config.REPLAY_DIR = self._orig_replay

    def test_phase1_write_returns_uuid(self):
        now = pd.Timestamp.now(tz=IST)
        row_id = ledger.phase1_write({
            "timestamp": now,
            "ticker": "NSE:RELIANCE",
            "cycle_id": "test_cycle",
            "open": 2845.0, "high": 2850.0, "low": 2840.0, "close": 2845.0,
            "volume": 1_000_000,
            "vwap": 2845.0, "rsi_14": 55.0,
            "macd": 0.5, "macd_signal": 0.3,
            "bb_upper": 2860.0, "bb_lower": 2830.0, "bb_middle": 2845.0, "atr_14": 15.0,
            "prediction_direction": "UP",
            "prediction_return_pct": 0.5,
            "prediction_confidence": "MED",
            "agent_contributions": [],
            "band_room_id": "room_1",
            "model_version": "phi3:mini",
            "inference_latency_ms": 150.0,
            "mode": "test",
            "data_source": "test",
        })
        # Valid UUID
        uuid.UUID(row_id)

    def test_load_ticker_df_returns_written_row(self):
        now = pd.Timestamp.now(tz=IST)
        row_id = ledger.phase1_write({
            "timestamp": now, "ticker": "NSE:TCS", "cycle_id": "c1",
            "open": 3920.0, "high": 3925.0, "low": 3915.0, "close": 3920.0,
            "volume": 500_000, "vwap": 3920.0, "rsi_14": 60.0,
            "macd": 1.0, "macd_signal": 0.8,
            "bb_upper": 3940.0, "bb_lower": 3900.0, "bb_middle": 3920.0, "atr_14": 20.0,
            "prediction_direction": "DOWN", "prediction_return_pct": -0.3,
            "prediction_confidence": "HIGH",
            "agent_contributions": [], "band_room_id": "room_2",
            "model_version": "phi3:mini", "inference_latency_ms": 200.0,
            "mode": "test", "data_source": "test",
        })
        df = ledger._load_ticker_df("NSE:TCS")
        assert not df.empty
        match = df[df["row_id"] == row_id]
        assert len(match) == 1
        assert match.iloc[0]["ticker"] == "NSE:TCS"

    def test_resolve_phase2_updates_actuals(self):
        """Write phase1, wait > INTERVAL_SECONDS, resolve with a price change."""
        now = pd.Timestamp.now(tz=IST)
        entry_ts = now - pd.Timedelta(seconds=config.INTERVAL_SECONDS + 60)

        row_id = ledger.phase1_write({
            "timestamp": entry_ts, "ticker": "NSE:HDFCBANK", "cycle_id": "c_resolve",
            "open": 1650.0, "high": 1655.0, "low": 1645.0, "close": 1650.0,
            "volume": 800_000, "vwap": 1650.0, "rsi_14": 50.0,
            "macd": 0.0, "macd_signal": 0.0,
            "bb_upper": 1660.0, "bb_lower": 1640.0, "bb_middle": 1650.0, "atr_14": 10.0,
            "prediction_direction": "UP", "prediction_return_pct": 0.5,
            "prediction_confidence": "MED",
            "agent_contributions": [], "band_room_id": "room_r",
            "model_version": "phi3:mini", "inference_latency_ms": 100.0,
            "mode": "test", "data_source": "test",
        })

        # Resolve with a price that should be UP (1650 → 1660 = +0.606%)
        resolved = ledger.resolve_phase2(now, current_prices={"NSE:HDFCBANK": 1660.0})
        assert resolved >= 1

        df = ledger._load_ticker_df("NSE:HDFCBANK")
        row = df[df["row_id"] == row_id].iloc[0]
        assert row["actual_direction"] == "UP"
        assert row["resolution_price"] == 1660.0
        assert row["prediction_accuracy"] == True  # predicted UP, actual UP (numpy bool)
        assert row["reward"] == 1.0

    def test_resolve_phase2_down_prediction(self):
        now = pd.Timestamp.now(tz=IST)
        entry_ts = now - pd.Timedelta(seconds=config.INTERVAL_SECONDS + 60)

        ledger.phase1_write({
            "timestamp": entry_ts, "ticker": "NSE:INFY", "cycle_id": "c_down",
            "open": 1480.0, "high": 1485.0, "low": 1475.0, "close": 1480.0,
            "volume": 600_000, "vwap": 1480.0, "rsi_14": 45.0,
            "macd": -0.5, "macd_signal": -0.3,
            "bb_upper": 1490.0, "bb_lower": 1470.0, "bb_middle": 1480.0, "atr_14": 12.0,
            "prediction_direction": "DOWN", "prediction_return_pct": -0.5,
            "prediction_confidence": "HIGH",
            "agent_contributions": [], "band_room_id": "room_d",
            "model_version": "phi3:mini", "inference_latency_ms": 120.0,
            "mode": "test", "data_source": "test",
        })

        resolved = ledger.resolve_phase2(now, current_prices={"NSE:INFY": 1470.0})
        assert resolved >= 1

        df = ledger._load_ticker_df("NSE:INFY")
        assert (df["actual_direction"] == "DOWN").any()

    def test_phase1_missing_ticker_raises(self):
        with pytest.raises(utils.LedgerError):
            ledger.phase1_write({"timestamp": pd.Timestamp.now(tz=IST)})

    def test_agent_contributions_serialization(self):
        """List agent_contributions should be JSON-serialized."""
        now = pd.Timestamp.now(tz=IST)
        ledger.phase1_write({
            "timestamp": now, "ticker": "NSE:SBIN", "cycle_id": "c_json",
            "open": 780.0, "high": 785.0, "low": 775.0, "close": 780.0,
            "volume": 400_000, "vwap": 780.0, "rsi_14": 50.0,
            "macd": 0.0, "macd_signal": 0.0,
            "bb_upper": 790.0, "bb_lower": 770.0, "bb_middle": 780.0, "atr_14": 8.0,
            "prediction_direction": "SIDEWAYS", "prediction_return_pct": 0.1,
            "prediction_confidence": "LOW",
            "agent_contributions": [{"sender": "TA", "text": "neutral", "type": "contribution", "turn": 0}],
            "band_room_id": "room_j", "model_version": "phi3:mini",
            "inference_latency_ms": 80.0, "mode": "test", "data_source": "test",
        })
        df = ledger._load_ticker_df("NSE:SBIN")
        raw = df.iloc[0]["agent_contributions"]
        parsed = json.loads(raw)
        assert isinstance(parsed, list)
        assert parsed[0]["sender"] == "TA"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Agent SDK — LocalBandSDK lifecycle
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLocalBandSDK:
    """create_room → send_message → get_room_history → cleanup_room"""

    def test_create_room_returns_id(self):
        sdk = LocalBandSDK()
        rid = sdk.create_room("test_room")
        assert rid.startswith("room_")
        assert rid in sdk.rooms

    def test_send_and_get_history(self):
        sdk = LocalBandSDK()
        rid = sdk.create_room("r1")
        sdk.send_message(rid, "AgentA", "Bullish signal", "contribution")
        sdk.send_message(rid, "AgentB", "Bearish signal", "contribution")
        history = sdk.get_room_history(rid)
        assert len(history) == 2
        assert history[0]["sender"] == "AgentA"
        assert history[1]["sender"] == "AgentB"
        assert history[0]["turn"] == 0
        assert history[1]["turn"] == 1

    def test_message_types(self):
        sdk = LocalBandSDK()
        rid = sdk.create_room("r_types")
        sdk.send_message(rid, "Sys", "hello", "system")
        sdk.send_message(rid, "Synth", "result", "synthesis")
        history = sdk.get_room_history(rid)
        assert history[0]["type"] == "system"
        assert history[1]["type"] == "synthesis"

    def test_send_to_nonexistent_room_is_noop(self):
        sdk = LocalBandSDK()
        sdk.send_message("room_999", "Agent", "msg")  # Should not raise
        assert sdk.get_room_history("room_999") == []

    def test_cleanup_room(self):
        sdk = LocalBandSDK()
        rid = sdk.create_room("r_cleanup")
        sdk.send_message(rid, "A", "msg")
        sdk.cleanup_room(rid)
        assert rid not in sdk.rooms
        assert sdk.get_room_history(rid) == []

    def test_room_counter_increments(self):
        sdk = LocalBandSDK()
        r1 = sdk.create_room("a")
        r2 = sdk.create_room("b")
        assert r1 != r2
        assert int(r2.split("_")[1]) == int(r1.split("_")[1]) + 1

    def test_get_history_formatted(self):
        sdk = LocalBandSDK()
        rid = sdk.create_room("r_fmt")
        sdk.send_message(rid, "TA", "RSI overbought")
        sdk.send_message(rid, "VA", "Volume spike")
        fmt = sdk.get_history_formatted(rid)
        assert "TA" in fmt
        assert "RSI overbought" in fmt
        assert "VA" in fmt

    def test_cleanup_old_rooms(self):
        sdk = LocalBandSDK()
        for i in range(5):
            sdk.create_room(f"r{i}")
        sdk.cleanup_old_rooms(max_rooms=3)
        assert len(sdk.rooms) == 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Dashboard state — concurrent writes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDashboardConcurrency:
    """Concurrent writes from multiple threads don't corrupt data."""

    def test_concurrent_updates_no_crash(self):
        state = DashboardState()
        errors = []

        def writer(prefix, count):
            try:
                for i in range(count):
                    state.update(
                        market_data={f"{prefix}_{i}": {"close": float(i)}},
                        predictions={f"{prefix}_{i}": {"direction": "UP"}},
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"t{tid}", 100)) for tid in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_snapshot_is_dict(self):
        state = DashboardState()
        state.update(market_data={"NSE:X": {"close": 100.0}})
        snap = state.get_snapshot()
        assert isinstance(snap, dict)
        assert "market" in snap
        assert "predictions" in snap
        assert "status" in snap
        assert "uptime" in snap["status"]

    def test_snapshot_top_level_isolation(self):
        """get_snapshot returns a new top-level dict each call (shallow copy).
        Replacing a top-level key in snap1 should not affect snap2."""
        state = DashboardState()
        state.update(market_data={"NSE:Y": {"close": 200.0}})
        snap1 = state.get_snapshot()
        snap2 = state.get_snapshot()
        # Replace the whole market key in snap1
        snap1["market"] = {"NSE:Z": {"close": 999.0}}
        assert snap2["market"]["NSE:Y"]["close"] == 200.0

    def test_concurrent_snapshot_reads(self):
        state = DashboardState()
        state.update(market_data={"NSE:Z": {"close": 300.0}})
        results = []

        def reader():
            snap = state.get_snapshot()
            results.append(snap)

        threads = [threading.Thread(target=reader) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        for r in results:
            assert isinstance(r, dict)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Market hours — weekday, weekend, holiday, Friday early close
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMarketHours:
    """Test with known dates: weekday, weekend, holiday, Friday early close."""

    def test_weekday_trading_day(self):
        # Monday 2026-08-10
        dt = datetime.datetime(2026, 8, 10, 12, 0, tzinfo=IST)
        assert is_trading_day(dt) is True

    def test_saturday_not_trading_day(self):
        # Saturday 2026-08-08
        dt = datetime.datetime(2026, 8, 8, 12, 0, tzinfo=IST)
        assert is_trading_day(dt) is False

    def test_sunday_not_trading_day(self):
        # Sunday 2026-08-09
        dt = datetime.datetime(2026, 8, 9, 12, 0, tzinfo=IST)
        assert is_trading_day(dt) is False

    def test_independence_day_holiday(self):
        # Aug 15 — Indian Independence Day
        dt = datetime.datetime(2026, 8, 15, 12, 0, tzinfo=IST)
        assert is_trading_day(dt) is False

    def test_trading_hours_midday(self):
        # Wednesday 2026-08-12, 12:00 IST
        dt = datetime.datetime(2026, 8, 12, 12, 0, tzinfo=IST)
        assert is_within_trading_hours(dt) is True

    def test_trading_hours_before_open(self):
        # Wednesday 2026-08-12, 08:00 IST — before 09:25
        dt = datetime.datetime(2026, 8, 12, 8, 0, tzinfo=IST)
        assert is_within_trading_hours(dt) is False

    def test_trading_hours_after_close(self):
        # Wednesday 2026-08-12, 16:00 IST — after 15:15
        dt = datetime.datetime(2026, 8, 12, 16, 0, tzinfo=IST)
        assert is_within_trading_hours(dt) is False

    def test_friday_early_close(self):
        # Friday 2026-08-14, 15:05 IST — after Friday close (15:00)
        dt = datetime.datetime(2026, 8, 14, 15, 5, tzinfo=IST)
        assert is_within_trading_hours(dt) is False

    def test_friday_before_early_close(self):
        # Friday 2026-08-14, 14:55 IST — before Friday close (15:00)
        dt = datetime.datetime(2026, 8, 14, 14, 55, tzinfo=IST)
        assert is_within_trading_hours(dt) is True

    def test_weekend_not_within_hours(self):
        # Saturday
        dt = datetime.datetime(2026, 8, 8, 12, 0, tzinfo=IST)
        assert is_within_trading_hours(dt) is False

    def test_premarket_time(self):
        # Wednesday 2026-08-12, 09:18 IST — premarket window
        dt = datetime.datetime(2026, 8, 12, 9, 18, tzinfo=IST)
        assert is_premarket_time(dt) is True

    def test_not_premarket_before_window(self):
        # Wednesday 2026-08-12, 09:10 IST — too early
        dt = datetime.datetime(2026, 8, 12, 9, 10, tzinfo=IST)
        assert is_premarket_time(dt) is False

    def test_naive_datetime_treated_as_ist(self):
        """Naive datetime (no tzinfo) should be treated as IST."""
        dt = datetime.datetime(2026, 8, 12, 12, 0)
        assert is_within_trading_hours(dt) is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Indicators — RSI bounded 0-100, VWAP reasonable, MACD finite
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestIndicators:
    """Verify indicator value ranges and sanity."""

    @pytest.fixture()
    def computed_df(self):
        df = _make_ohlcv_df(200)
        return indicators.compute_indicators(df)

    def test_rsi_bounded_0_100(self, computed_df):
        rsi = computed_df["rsi_14"].dropna()
        assert (rsi >= 0).all(), f"RSI min = {rsi.min()}"
        assert (rsi <= 100).all(), f"RSI max = {rsi.max()}"

    def test_vwap_reasonable(self, computed_df):
        vwap = computed_df["vwap"].dropna()
        assert len(vwap) > 0
        # VWAP is a cumulative average so it can drift beyond a single candle.
        # It should be within the global price range of the dataset (with margin).
        global_low = computed_df["low"].min()
        global_high = computed_df["high"].max()
        assert vwap.min() >= global_low * 0.95, f"VWAP min {vwap.min()} below global low {global_low}"
        assert vwap.max() <= global_high * 1.05, f"VWAP max {vwap.max()} above global high {global_high}"

    def test_macd_finite(self, computed_df):
        macd = computed_df["macd"].dropna()
        macd_sig = computed_df["macd_signal"].dropna()
        assert all(math.isfinite(v) for v in macd)
        assert all(math.isfinite(v) for v in macd_sig)

    def test_bollinger_order(self, computed_df):
        """bb_lower <= bb_middle <= bb_upper for non-NaN rows."""
        for _, row in computed_df.iterrows():
            if not pd.isna(row["bb_lower"]):
                assert row["bb_lower"] <= row["bb_middle"] <= row["bb_upper"]

    def test_atr_positive(self, computed_df):
        atr = computed_df["atr_14"].dropna()
        assert (atr >= 0).all()

    def test_sma50_nan_on_short_data(self):
        df = _make_ohlcv_df(30)
        result = indicators.compute_indicators(df)
        assert result["sma_50"].isna().all()

    def test_sma50_present_on_long_data(self, computed_df):
        sma50 = computed_df["sma_50"].dropna()
        assert len(sma50) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Predict module — mock Ollama, test PredictionSquad.run()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPredictModule:
    """Mock Ollama API and verify PredictionSquad returns valid structure."""

    def _mock_ollama_generate(self, prompt, temperature=None, max_tokens=None,
                               format_json=False, grammar=None):
        """Return deterministic JSON for synthesis, plain text for agents."""
        if format_json:
            return json.dumps({
                "direction": "UP",
                "target_return_pct": 0.75,
                "confidence": "HIGH",
            })
        return "Technical analysis shows bullish momentum with RSI at 65."

    @patch("predict.ollama_generate")
    def test_squad_run_returns_valid_structure(self, mock_gen):
        mock_gen.side_effect = self._mock_ollama_generate
        sdk = LocalBandSDK()
        squad = PredictionSquad(band_sdk=sdk)
        room_id = sdk.create_room("test")

        result = squad.run("NSE:RELIANCE", "Ticker: NSE:RELIANCE\nRSI: 65", room_id)

        assert isinstance(result, dict)
        assert "direction" in result
        assert "target_return_pct" in result
        assert "confidence" in result
        assert "inference_latency_ms" in result
        assert "agent_contributions" in result

    @patch("predict.ollama_generate")
    def test_squad_run_direction_valid(self, mock_gen):
        mock_gen.side_effect = self._mock_ollama_generate
        sdk = LocalBandSDK()
        squad = PredictionSquad(band_sdk=sdk)
        room_id = sdk.create_room("test")

        result = squad.run("NSE:TCS", "data", room_id)
        assert result["direction"] in ("UP", "DOWN", "SIDEWAYS")

    @patch("predict.ollama_generate")
    def test_squad_run_confidence_valid(self, mock_gen):
        mock_gen.side_effect = self._mock_ollama_generate
        sdk = LocalBandSDK()
        squad = PredictionSquad(band_sdk=sdk)
        room_id = sdk.create_room("test")

        result = squad.run("NSE:INFY", "data", room_id)
        assert result["confidence"] in ("LOW", "MED", "HIGH")

    @patch("predict.ollama_generate")
    def test_squad_run_agents_send_messages(self, mock_gen):
        mock_gen.side_effect = self._mock_ollama_generate
        sdk = LocalBandSDK()
        squad = PredictionSquad(band_sdk=sdk)
        room_id = sdk.create_room("test")

        squad.run("NSE:HDFCBANK", "data", room_id)
        history = sdk.get_room_history(room_id)
        # 3 agents + 1 synthesizer = 4 messages
        assert len(history) == 4

    @patch("predict.ollama_generate")
    def test_squad_run_invalid_json_retry(self, mock_gen):
        """First call returns bad JSON, retry returns valid JSON."""
        call_count = {"n": 0}

        def side_effect(prompt, **kwargs):
            call_count["n"] += 1
            if kwargs.get("format_json"):
                if call_count["n"] <= 4:  # First synthesis attempt
                    return "not valid json"
                return json.dumps({"direction": "UP", "target_return_pct": 0.5, "confidence": "MED"})
            return "Agent analysis"

        mock_gen.side_effect = side_effect
        sdk = LocalBandSDK()
        squad = PredictionSquad(band_sdk=sdk)
        room_id = sdk.create_room("test")

        result = squad.run("NSE:SBIN", "data", room_id)
        assert result["direction"] in ("UP", "DOWN", "SIDEWAYS")

    @patch("predict.ollama_generate")
    def test_squad_run_clamps_invalid_direction(self, mock_gen):
        """Direction not in valid set should be clamped to SIDEWAYS."""
        def side_effect(prompt, **kwargs):
            if kwargs.get("format_json"):
                return json.dumps({"direction": "INVALID", "target_return_pct": 0.0, "confidence": "LOW"})
            return "analysis"

        mock_gen.side_effect = side_effect
        sdk = LocalBandSDK()
        squad = PredictionSquad(band_sdk=sdk)
        room_id = sdk.create_room("test")

        result = squad.run("NSE:ITC", "data", room_id)
        assert result["direction"] == "SIDEWAYS"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Charts — generate_accuracy_trend() on empty data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCharts:
    """Charts generate without error, even on empty/missing data."""

    @pytest.fixture(autouse=True)
    def _tmp_dirs(self, tmp_path):
        self._orig_replay = config.REPLAY_DIR
        self._orig_charts = config.CHARTS_DIR
        config.REPLAY_DIR = tmp_path / "replay"
        config.CHARTS_DIR = tmp_path / "charts"
        config.REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        yield
        config.REPLAY_DIR = self._orig_replay
        config.CHARTS_DIR = self._orig_charts

    def test_accuracy_trend_no_buffer(self):
        """No replay buffer file — should return without error."""
        generate_accuracy_trend()

    def test_accuracy_trend_empty_buffer(self):
        """Empty replay buffer — should return without error."""
        buf = config.REPLAY_DIR / "replay_buffer.parquet"
        pd.DataFrame(columns=ledger.ALL_COLUMNS).to_parquet(buf, index=False)
        generate_accuracy_trend()

    def test_accuracy_trend_with_data(self):
        """Populated buffer — should produce a PNG file."""
        rows = []
        for i in range(50):
            rows.append({
                "row_id": str(uuid.uuid4()),
                "timestamp": pd.Timestamp.now(tz=IST) - pd.Timedelta(minutes=5 * (50 - i)),
                "ticker": "NSE:RELIANCE",
                "cycle_id": f"c{i}",
                "open": 2845.0, "high": 2850.0, "low": 2840.0, "close": 2845.0,
                "volume": 1000000, "vwap": 2845.0, "rsi_14": 55.0,
                "macd": 0.5, "macd_signal": 0.3,
                "bb_upper": 2860.0, "bb_lower": 2830.0, "bb_middle": 2845.0, "atr_14": 15.0,
                "prediction_direction": random.choice(["UP", "DOWN", "SIDEWAYS"]),
                "prediction_return_pct": 0.5, "prediction_confidence": "MED",
                "agent_contributions": "[]", "band_room_id": f"room_{i}",
                "model_version": "phi3:mini", "inference_latency_ms": 100.0,
                "mode": "test", "data_source": "test",
                "actual_direction": random.choice(["UP", "DOWN", "SIDEWAYS"]),
                "actual_return_pct": random.uniform(-1, 1),
                "resolution_timestamp": pd.Timestamp.now(tz=IST),
                "prediction_accuracy": random.choice([True, False]),
                "resolution_price": 2845.0, "reward": random.choice([0.0, 1.0]),
                "used_in_training": False,
            })
        buf = config.REPLAY_DIR / "replay_buffer.parquet"
        pd.DataFrame(rows).to_parquet(buf, index=False)

        output = config.CHARTS_DIR / "test_accuracy.png"
        generate_accuracy_trend(output_path=output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_latency_trend_with_data(self):
        history = [
            {"inference_avg_ms": 100 + i * 5, "total_cycle_avg_ms": 200 + i * 10}
            for i in range(20)
        ]
        output = config.CHARTS_DIR / "test_latency.png"
        generate_latency_trend(latency_history=history, output_path=output)
        assert output.exists()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Learn module — format_training_pair, balance_classes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLearnModule:
    """format_training_pair returns correct structure; balance_classes balances."""

    def _make_resolved_row(self, direction="UP", return_pct=1.0):
        return {
            "row_id": str(uuid.uuid4()),
            "ticker": "NSE:RELIANCE",
            "open": 2845.0, "high": 2850.0, "low": 2840.0, "close": 2845.0,
            "volume": 1000000, "vwap": 2845.0, "rsi_14": 55.0,
            "macd": 0.5, "macd_signal": 0.3,
            "bb_upper": 2860.0, "bb_lower": 2830.0, "bb_middle": 2845.0, "atr_14": 15.0,
            "prediction_direction": direction,
            "prediction_return_pct": return_pct,
            "prediction_confidence": "MED",
            "actual_direction": direction,
            "actual_return_pct": return_pct,
            "prediction_accuracy": True,
            "used_in_training": False,
        }

    def test_format_training_pair_structure(self):
        row = self._make_resolved_row("UP", 0.75)
        pair = format_training_pair(row)
        assert isinstance(pair, dict)
        assert "prompt" in pair
        assert "completion" in pair
        assert "row_id" in pair

    def test_format_training_pair_prompt_contains_ticker(self):
        row = self._make_resolved_row()
        pair = format_training_pair(row)
        assert "NSE:RELIANCE" in pair["prompt"]

    def test_format_training_pair_completion_is_json(self):
        row = self._make_resolved_row("DOWN", -0.5)
        pair = format_training_pair(row)
        comp = json.loads(pair["completion"])
        assert comp["direction"] == "DOWN"
        assert comp["target_return_pct"] == -0.5
        assert "confidence" in comp

    def test_format_training_pair_preserves_row_id(self):
        row = self._make_resolved_row()
        pair = format_training_pair(row)
        assert pair["row_id"] == row["row_id"]

    def test_balance_classes_balances(self):
        """With 10 UP, 5 DOWN, 2 SIDEWAYS, output should have equal counts."""
        rows = (
            [self._make_resolved_row("UP", 0.5)] * 10
            + [self._make_resolved_row("DOWN", -0.5)] * 5
            + [self._make_resolved_row("SIDEWAYS", 0.1)] * 2
        )
        balanced = balance_classes(rows)
        by_dir = {"UP": 0, "DOWN": 0, "SIDEWAYS": 0}
        for r in balanced:
            by_dir[r["actual_direction"]] += 1
        # All classes should have min(10,5,2) = 2
        assert by_dir["UP"] == 2
        assert by_dir["DOWN"] == 2
        assert by_dir["SIDEWAYS"] == 2

    def test_balance_classes_empty_input(self):
        assert balance_classes([]) == []

    def test_balance_classes_single_class(self):
        """Only one class — balanced result should be empty (min of non-empty = itself, but other classes empty)."""
        rows = [self._make_resolved_row("UP", 0.5)] * 5
        balanced = balance_classes(rows)
        # When one class is empty, non_empty = [UP_list], min_count = 5
        # But only UP has rows, so balanced should have 5 UP rows
        # Wait: the code does `for cls_rows in by_class.values(): if cls_rows: balanced.extend(...)`
        # So all non-empty classes get min_count items. Since only UP is non-empty, result = 5 UP.
        assert len(balanced) == 5

    def test_balance_classes_preserves_row_ids(self):
        rows = [self._make_resolved_row(d) for d in ["UP", "DOWN", "SIDEWAYS"]]
        balanced = balance_classes(rows)
        ids_in = {r["row_id"] for r in rows}
        ids_out = {r["row_id"] for r in balanced}
        assert ids_out.issubset(ids_in)
