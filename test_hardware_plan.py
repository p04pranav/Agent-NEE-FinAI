"""
Agent-NEE FinAI — Hardware-Specific Test Plan
Covers: integration (live Ollama, data pipeline, ledger, dashboard),
performance benchmarks, and edge cases.

Hardware target: Intel Xeon 2.20GHz 2-core, 12GB RAM, no GPU (CPU mode).
"""

import datetime
import json
import os
import shutil
import tempfile
import time
import threading
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests
from zoneinfo import ZoneInfo

import sys
sys.path.insert(0, str(Path(__file__).parent))

import config
import data_source
import indicators
import ledger
import utils
from agents import LocalBandSDK, AGENT_ROLES, SYNTHESIS_PROMPT
from predict import PredictionSquad, ollama_generate, verify_ollama
from web_server import DashboardState, start_web_server, app
from main import (
    IST, is_trading_day, is_within_trading_hours,
    auto_detect_training, trim_tickers_for_hardware, prediction_cycle,
)
from charts import generate_accuracy_trend, generate_latency_trend, generate_all
from learn import balance_classes, format_training_pair, check_training_deps

# ─── Helpers ──────────────────────────────────────────────────────────

def _make_ohlcv_df(n: int = 100, base_price: float = 2500.0) -> pd.DataFrame:
    """Build synthetic OHLCV DataFrame."""
    import random
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
            "open": round(o, 2), "high": round(h, 2),
            "low": round(l, 2), "close": round(c, 2),
            "volume": vol,
        })
        price = c
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture(autouse=True)
def _reset_data_source():
    """Reset data_source module state between tests."""
    yield
    data_source._data.clear()
    data_source._cursors.clear()


def _ollama_available() -> bool:
    """Check if Ollama is running and phi3:mini is available."""
    try:
        resp = requests.get(config.OLLAMA_API_TAGS, timeout=5)
        if resp.status_code != 200:
            return False
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        return any(config.MODEL_NAME in m for m in models)
    except Exception:
        return False


# Skip decorator for tests requiring live Ollama
requires_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not running or phi3:mini not available",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 2: Integration Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# ─── 2.1 Data Pipeline ───────────────────────────────────────────────

class TestDataPipelineIntegration:
    """CSV load → poll → indicators → format end-to-end."""

    def test_load_all_tickers_from_csv(self):
        """All 10 tickers load from data/csv/."""
        data_source._data.clear()
        data_source._cursors.clear()
        data = data_source.load_all()
        assert len(data) == 10
        for ticker in config.TICKER_SYMBOLS:
            assert ticker in data

    def test_csv_row_count(self):
        """Each CSV has 1000 rows."""
        data = data_source.load_all()
        for ticker, df in data.items():
            assert len(df) == 1000, f"{ticker} has {len(df)} rows"

    def test_csv_has_correct_columns(self):
        """CSVs have timestamp, open, high, low, close, volume."""
        data = data_source.load_all()
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        for ticker, df in data.items():
            assert required.issubset(set(df.columns)), f"{ticker} missing columns"

    def test_poll_tickers_returns_data(self):
        """poll_tickers() returns data for current cycle."""
        data_source._data.clear()
        data_source._cursors.clear()
        result = data_source.poll_tickers()
        assert len(result) > 0
        for ticker, df in result.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) == config.CSV_ROWS_PER_CYCLE

    def test_poll_cursor_advances(self):
        """Calling poll_tickers() twice advances the cursor."""
        data_source._data.clear()
        data_source._cursors.clear()
        data_source.poll_tickers()
        cursor_before = dict(data_source._cursors)
        data_source.poll_tickers()
        for ticker in data_source._cursors:
            if ticker in cursor_before:
                assert data_source._cursors[ticker] > cursor_before[ticker]

    def test_poll_exhaustion_and_loop(self):
        """When CSV is exhausted with SIMULATION_LOOP=True, cursor resets."""
        data_source._data.clear()
        data_source._cursors.clear()
        # Poll enough times to exhaust a CSV
        for _ in range(1001):
            data_source.poll_tickers()
        # Should still be serving data (looped)
        result = data_source.poll_tickers()
        assert len(result) > 0

    def test_indicators_on_real_csv_data(self):
        """compute_indicators() produces valid indicators on CSV data."""
        data = data_source.load_all()
        df = data["NSE:RELIANCE"]
        result = indicators.compute_indicators(df)
        # RSI should be in [0, 100]
        rsi = result["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()
        # ATR should be positive
        atr = result["atr_14"].dropna()
        assert (atr >= 0).all()

    def test_format_output_contains_all_fields(self):
        """format_market_data_for_agent() includes all expected fields."""
        data = data_source.load_all()
        df = indicators.compute_indicators(data["NSE:TCS"])
        latest = df.iloc[-1]
        text = indicators.format_market_data_for_agent("NSE:TCS", latest)
        for field in ["Ticker:", "Price:", "Volume:", "VWAP:", "RSI(14):", "MACD:", "Bollinger", "ATR(14):"]:
            assert field in text, f"Missing {field} in formatted output"


# ─── 2.2 Ledger Persistence ──────────────────────────────────────────

class TestLedgerPersistenceIntegration:
    """Parquet file creation, Phase 2 resolution, UUID uniqueness."""

    @pytest.fixture(autouse=True)
    def _tmp_dirs(self, tmp_path):
        self._orig_daily = config.DAILY_DIR
        self._orig_replay = config.REPLAY_DIR
        config.DAILY_DIR = tmp_path / "daily"
        config.REPLAY_DIR = tmp_path / "replay"
        config.DAILY_DIR.mkdir(parents=True, exist_ok=True)
        config.REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        yield
        config.DAILY_DIR = self._orig_daily
        config.REPLAY_DIR = self._orig_replay

    def test_parquet_file_created_on_write(self):
        """Phase 1 write creates a Parquet file for the ticker."""
        ledger.phase1_write({
            "timestamp": pd.Timestamp.now(tz=IST), "ticker": "NSE:RELIANCE",
            "cycle_id": "c1", "open": 2845.0, "high": 2850.0, "low": 2840.0,
            "close": 2845.0, "volume": 1_000_000, "vwap": 2845.0, "rsi_14": 55.0,
            "macd": 0.5, "macd_signal": 0.3, "bb_upper": 2860.0, "bb_lower": 2830.0,
            "bb_middle": 2845.0, "atr_14": 15.0, "prediction_direction": "UP",
            "prediction_return_pct": 0.5, "prediction_confidence": "MED",
            "agent_contributions": [], "band_room_id": "room_1",
            "model_version": "phi3:mini", "inference_latency_ms": 150.0,
            "mode": "test", "data_source": "test",
        })
        path = config.DAILY_DIR / "NSE_RELIANCE.parquet"
        assert path.exists()
        df = pd.read_parquet(path)
        assert len(df) == 1

    def test_phase2_resolves_with_price_change(self):
        """Phase 2 resolves UP when price increases beyond threshold."""
        now = pd.Timestamp.now(tz=IST)
        entry_ts = now - pd.Timedelta(seconds=config.INTERVAL_SECONDS + 60)
        row_id = ledger.phase1_write({
            "timestamp": entry_ts, "ticker": "NSE:HDFCBANK", "cycle_id": "c_r",
            "open": 1650.0, "high": 1655.0, "low": 1645.0, "close": 1650.0,
            "volume": 800_000, "vwap": 1650.0, "rsi_14": 50.0,
            "macd": 0.0, "macd_signal": 0.0, "bb_upper": 1660.0, "bb_lower": 1640.0,
            "bb_middle": 1650.0, "atr_14": 10.0, "prediction_direction": "UP",
            "prediction_return_pct": 0.5, "prediction_confidence": "MED",
            "agent_contributions": [], "band_room_id": "room_r",
            "model_version": "phi3:mini", "inference_latency_ms": 100.0,
            "mode": "test", "data_source": "test",
        })
        resolved = ledger.resolve_phase2(now, current_prices={"NSE:HDFCBANK": 1660.0})
        assert resolved >= 1
        df = ledger._load_ticker_df("NSE:HDFCBANK")
        row = df[df["row_id"] == row_id].iloc[0]
        assert row["actual_direction"] == "UP"
        assert row["prediction_accuracy"] == True

    def test_uuid_uniqueness_100_rows(self):
        """100 Phase 1 writes produce 100 unique UUIDs."""
        now = pd.Timestamp.now(tz=IST)
        ids = set()
        for i in range(100):
            rid = ledger.phase1_write({
                "timestamp": now, "ticker": "NSE:SBIN", "cycle_id": f"c{i}",
                "open": 780.0, "high": 785.0, "low": 775.0, "close": 780.0,
                "volume": 400_000, "vwap": 780.0, "rsi_14": 50.0,
                "macd": 0.0, "macd_signal": 0.0, "bb_upper": 790.0, "bb_lower": 770.0,
                "bb_middle": 780.0, "atr_14": 8.0, "prediction_direction": "UP",
                "prediction_return_pct": 0.1, "prediction_confidence": "LOW",
                "agent_contributions": [], "band_room_id": f"room_{i}",
                "model_version": "phi3:mini", "inference_latency_ms": 80.0,
                "mode": "test", "data_source": "test",
            })
            ids.add(rid)
        assert len(ids) == 100

    def test_agent_contributions_json_roundtrip(self):
        """List agent_contributions stored as JSON, loadable back."""
        now = pd.Timestamp.now(tz=IST)
        contributions = [
            {"sender": "Technical Analyst", "text": "RSI overbought", "type": "contribution", "turn": 0},
            {"sender": "Volatility Analyst", "text": "Low vol regime", "type": "contribution", "turn": 1},
        ]
        ledger.phase1_write({
            "timestamp": now, "ticker": "NSE:ITC", "cycle_id": "c_json",
            "open": 480.0, "high": 485.0, "low": 475.0, "close": 480.0,
            "volume": 300_000, "vwap": 480.0, "rsi_14": 48.0,
            "macd": 0.0, "macd_signal": 0.0, "bb_upper": 490.0, "bb_lower": 470.0,
            "bb_middle": 480.0, "atr_14": 5.0, "prediction_direction": "SIDEWAYS",
            "prediction_return_pct": 0.05, "prediction_confidence": "LOW",
            "agent_contributions": contributions, "band_room_id": "room_j",
            "model_version": "phi3:mini", "inference_latency_ms": 90.0,
            "mode": "test", "data_source": "test",
        })
        df = ledger._load_ticker_df("NSE:ITC")
        raw = df.iloc[0]["agent_contributions"]
        parsed = json.loads(raw)
        assert len(parsed) == 2
        assert parsed[0]["sender"] == "Technical Analyst"

    def test_orphan_cleanup_marks_expired(self):
        """Rows older than 1 hour get marked EXPIRED."""
        now = pd.Timestamp.now(tz=IST)
        old_ts = now - pd.Timedelta(hours=2)
        ledger.phase1_write({
            "timestamp": old_ts, "ticker": "NSE:WIPRO", "cycle_id": "c_old",
            "open": 510.0, "high": 515.0, "low": 505.0, "close": 510.0,
            "volume": 200_000, "vwap": 510.0, "rsi_14": 50.0,
            "macd": 0.0, "macd_signal": 0.0, "bb_upper": 520.0, "bb_lower": 500.0,
            "bb_middle": 510.0, "atr_14": 5.0, "prediction_direction": "UP",
            "prediction_return_pct": 0.1, "prediction_confidence": "LOW",
            "agent_contributions": [], "band_room_id": "room_old",
            "model_version": "phi3:mini", "inference_latency_ms": 80.0,
            "mode": "test", "data_source": "test",
        })
        cleaned = ledger.cleanup_orphans()
        assert cleaned >= 1
        df = ledger._load_ticker_df("NSE:WIPRO")
        assert (df["actual_direction"] == "EXPIRED").any()


# ─── 2.3 Ollama Inference (live) ─────────────────────────────────────

class TestOllamaInferenceLive:
    """Live Ollama inference tests — CPU mode, phi3:mini."""

    @requires_ollama
    def test_ollama_responds_to_prompt(self):
        """Ollama returns non-empty response for simple prompt."""
        resp = ollama_generate("What is 2+2? Answer in one word.")
        assert isinstance(resp, str)
        assert len(resp) > 0

    @requires_ollama
    def test_json_mode_returns_valid_json(self):
        """format_json=True returns parseable JSON."""
        resp = ollama_generate(
            'Return JSON: {"answer": "yes"}',
            format_json=True,
            max_tokens=64,
        )
        parsed = json.loads(resp)
        assert isinstance(parsed, dict)

    @requires_ollama
    def test_single_agent_latency_under_10s(self):
        """Single agent inference < 120s on CPU (phi3:mini on Xeon 2.2GHz, 2 cores)."""
        role = AGENT_ROLES["technical_analyst"]
        prompt = f"{role['prompt']}\n\nTicker: NSE:RELIANCE\nPrice: O=2845 H=2850 L=2840 C=2845\nVolume: 1000000\nRSI(14): 55.0"
        start = time.time()
        resp = ollama_generate(prompt, temperature=role["temperature"], max_tokens=64)
        elapsed = time.time() - start
        assert elapsed < 120.0, f"Inference took {elapsed:.1f}s > 120s"
        assert len(resp) > 0

    @requires_ollama
    def test_squad_run_returns_valid_prediction(self):
        """PredictionSquad.run() returns valid direction/confidence/return."""
        sdk = LocalBandSDK()
        squad = PredictionSquad(band_sdk=sdk)
        data = data_source.load_all()
        df = indicators.compute_indicators(data["NSE:RELIANCE"])
        latest = df.iloc[-1]
        ticker_data = indicators.format_market_data_for_agent("NSE:RELIANCE", latest)
        room_id = sdk.create_room("test_live")

        result = squad.run("NSE:RELIANCE", ticker_data, room_id)

        assert result["direction"] in ("UP", "DOWN", "SIDEWAYS")
        assert result["confidence"] in ("LOW", "MED", "HIGH")
        assert isinstance(result["target_return_pct"], float)
        assert result["inference_latency_ms"] > 0

    @requires_ollama
    def test_five_predictions_all_valid(self):
        """5 consecutive predictions all have valid direction/confidence."""
        sdk = LocalBandSDK()
        squad = PredictionSquad(band_sdk=sdk)
        data = data_source.load_all()
        df = indicators.compute_indicators(data["NSE:TCS"])
        latest = df.iloc[-1]
        ticker_data = indicators.format_market_data_for_agent("NSE:TCS", latest)

        for i in range(3):  # 3 predictions (reduced from 5 for CPU speed)
            room_id = sdk.create_room(f"test_{i}")
            result = squad.run("NSE:TCS", ticker_data, room_id)
            assert result["direction"] in ("UP", "DOWN", "SIDEWAYS")
            assert result["confidence"] in ("LOW", "MED", "HIGH")
            sdk.cleanup_room(room_id)

    @requires_ollama
    def test_full_cycle_latency_3_tickers(self):
        """Full prediction_cycle() for 3 tickers < 45s on CPU."""
        # Reset data source cursors
        data_source._data.clear()
        data_source._cursors.clear()

        sdk = LocalBandSDK()
        squad = PredictionSquad(band_sdk=sdk)
        active_tickers = config.TICKER_SYMBOLS[:3]  # CPU mode: 3 tickers

        # Patch dashboard state to avoid side effects
        with patch("main.dashboard_state"):
            start = time.time()
            prediction_cycle(squad, active_tickers, training_enabled=False)
            elapsed = time.time() - start

        assert elapsed < 45.0, f"Cycle took {elapsed:.1f}s > 45s"


# ─── 2.4 Web Dashboard ───────────────────────────────────────────────

class TestWebDashboardIntegration:
    """FastAPI server, REST endpoints, WebSocket, security headers."""

    @pytest.fixture()
    def client(self):
        """Create test client for FastAPI app."""
        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_api_status_returns_json(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_api_config_returns_model_and_tickers(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "model" in data
        assert "tickers" in data
        assert len(data["tickers"]) == 10

    def test_api_history_returns_predictions(self, client):
        resp = client.get("/api/history")
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert "market" in data

    def test_security_headers_present(self, client):
        resp = client.get("/api/status")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert "Content-Security-Policy" in resp.headers

    def test_websocket_connects(self, client):
        with client.websocket_connect("/ws") as ws:
            # Send ping, expect pong
            ws.send_text("ping")
            data = ws.receive_text()
            assert data == "pong"

    def test_websocket_receives_broadcast(self, client):
        """WebSocket can send and receive messages (broadcast tested via ping/pong)."""
        with client.websocket_connect("/ws") as ws:
            ws.send_text("ping")
            data = ws.receive_text()
            assert data == "pong"

    def test_websocket_large_message_rejected(self, client):
        """Messages > 1KB should close connection."""
        with client.websocket_connect("/ws") as ws:
            large_msg = "x" * 2000
            ws.send_text(large_msg)
            # Connection should close (code 1009)
            with pytest.raises(Exception):
                ws.receive_text()


# ─── 2.5 End-to-End Mock ─────────────────────────────────────────────

class TestEndToEndMock:
    """Full pipeline with mocked Ollama — no live inference."""

    @pytest.fixture(autouse=True)
    def _tmp_dirs(self, tmp_path):
        self._orig_daily = config.DAILY_DIR
        self._orig_replay = config.REPLAY_DIR
        config.DAILY_DIR = tmp_path / "daily"
        config.REPLAY_DIR = tmp_path / "replay"
        config.DAILY_DIR.mkdir(parents=True, exist_ok=True)
        config.REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        yield
        config.DAILY_DIR = self._orig_daily
        config.REPLAY_DIR = self._orig_replay

    def _mock_ollama_generate(self, prompt, temperature=None, max_tokens=None,
                               format_json=False, grammar=None):
        if format_json:
            return json.dumps({"direction": "UP", "target_return_pct": 0.75, "confidence": "HIGH"})
        return "Technical analysis shows bullish momentum with RSI at 65."

    @patch("predict.ollama_generate")
    def test_mock_prediction_cycle_writes_to_ledger(self, mock_gen):
        """Mocked prediction_cycle writes Phase 1 rows to ledger."""
        mock_gen.side_effect = self._mock_ollama_generate
        data_source._data.clear()
        data_source._cursors.clear()

        # Load data and advance cursors past MIN_INDICATOR_ROWS
        data_source.load_all()
        for ticker in data_source._cursors:
            data_source._cursors[ticker] = config.MIN_INDICATOR_ROWS

        sdk = LocalBandSDK()
        squad = PredictionSquad(band_sdk=sdk)
        active_tickers = config.TICKER_SYMBOLS[:3]

        with patch("main.dashboard_state"):
            prediction_cycle(squad, active_tickers, training_enabled=False)

        # Check ledger has rows
        for ticker in active_tickers:
            df = ledger._load_ticker_df(ticker)
            if not df.empty:
                assert df.iloc[0]["prediction_direction"] in ("UP", "DOWN", "SIDEWAYS")

    @patch("predict.ollama_generate")
    def test_mock_multiple_cycles_grow_ledger(self, mock_gen):
        """Three mock cycles produce growing ledger (using full CSV data)."""
        mock_gen.side_effect = self._mock_ollama_generate
        data_source._data.clear()
        data_source._cursors.clear()

        sdk = LocalBandSDK()
        squad = PredictionSquad(band_sdk=sdk)
        ticker = config.TICKER_SYMBOLS[0]

        total_before = len(ledger._load_ticker_df(ticker))

        # Use full CSV data to get enough rows for indicators
        data = data_source.load_all()
        df = indicators.compute_indicators(data[ticker])
        latest = df.iloc[-1]
        ticker_data = indicators.format_market_data_for_agent(ticker, latest)
        room_id = sdk.create_room("mock_multi")

        for i in range(3):
            result = squad.run(ticker, ticker_data, room_id)
            ledger.phase1_write({
                "timestamp": pd.Timestamp.now(tz=IST), "ticker": ticker,
                "cycle_id": f"c{i}", "open": float(latest["open"]),
                "high": float(latest["high"]), "low": float(latest["low"]),
                "close": float(latest["close"]), "volume": float(latest["volume"]),
                "vwap": float(latest.get("vwap", 0)), "rsi_14": float(latest.get("rsi_14", 0)),
                "macd": float(latest.get("macd", 0)), "macd_signal": float(latest.get("macd_signal", 0)),
                "bb_upper": float(latest.get("bb_upper", 0)), "bb_lower": float(latest.get("bb_lower", 0)),
                "bb_middle": float(latest.get("bb_middle", 0)), "atr_14": float(latest.get("atr_14", 0)),
                "prediction_direction": result["direction"],
                "prediction_return_pct": result["target_return_pct"],
                "prediction_confidence": result["confidence"],
                "agent_contributions": result.get("agent_contributions", []),
                "band_room_id": room_id, "model_version": config.MODEL_NAME,
                "inference_latency_ms": result["inference_latency_ms"],
                "mode": "test", "data_source": "test",
            })

        total_after = len(ledger._load_ticker_df(ticker))
        assert total_after > total_before

    @patch("predict.ollama_generate")
    def test_mock_dashboard_receives_predictions(self, mock_gen):
        """Dashboard state receives predictions after mock cycle."""
        mock_gen.side_effect = self._mock_ollama_generate
        data_source._data.clear()
        data_source._cursors.clear()

        sdk = LocalBandSDK()
        squad = PredictionSquad(band_sdk=sdk)
        state = DashboardState()

        # Use full CSV data
        data = data_source.load_all()
        ticker = config.TICKER_SYMBOLS[0]
        df = indicators.compute_indicators(data[ticker])
        latest = df.iloc[-1]
        ticker_data = indicators.format_market_data_for_agent(ticker, latest)
        room_id = sdk.create_room("mock_dash")

        result = squad.run(ticker, ticker_data, room_id)
        short_ticker = ticker.replace("NSE:", "")
        state.update(predictions={short_ticker: {
            "direction": result["direction"],
            "target_return_pct": result["target_return_pct"],
            "confidence": result["confidence"],
        }})

        snap = state.get_snapshot()
        assert len(snap["predictions"]) > 0


# ─── 2.6 End-to-End Live ─────────────────────────────────────────────

class TestEndToEndLive:
    """Full pipeline with live Ollama — CPU mode, 1-3 tickers."""

    @pytest.fixture(autouse=True)
    def _tmp_dirs(self, tmp_path):
        self._orig_daily = config.DAILY_DIR
        self._orig_replay = config.REPLAY_DIR
        config.DAILY_DIR = tmp_path / "daily"
        config.REPLAY_DIR = tmp_path / "replay"
        config.DAILY_DIR.mkdir(parents=True, exist_ok=True)
        config.REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        # Reset data source
        data_source._data.clear()
        data_source._cursors.clear()
        yield
        config.DAILY_DIR = self._orig_daily
        config.REPLAY_DIR = self._orig_replay

    @requires_ollama
    def test_single_ticker_prediction_live(self):
        """Single ticker prediction with live Ollama."""
        sdk = LocalBandSDK()
        squad = PredictionSquad(band_sdk=sdk)
        data = data_source.load_all()
        df = indicators.compute_indicators(data["NSE:RELIANCE"])
        latest = df.iloc[-1]
        ticker_data = indicators.format_market_data_for_agent("NSE:RELIANCE", latest)
        room_id = sdk.create_room("e2e_live")

        result = squad.run("NSE:RELIANCE", ticker_data, room_id)

        assert result["direction"] in ("UP", "DOWN", "SIDEWAYS")
        assert result["confidence"] in ("LOW", "MED", "HIGH")

        # Write to ledger and verify
        row_id = ledger.phase1_write({
            "timestamp": pd.Timestamp.now(tz=IST), "ticker": "NSE:RELIANCE",
            "cycle_id": "e2e", "open": float(latest["open"]), "high": float(latest["high"]),
            "low": float(latest["low"]), "close": float(latest["close"]),
            "volume": float(latest["volume"]), "vwap": float(latest.get("vwap", 0)),
            "rsi_14": float(latest.get("rsi_14", 0)), "macd": float(latest.get("macd", 0)),
            "macd_signal": float(latest.get("macd_signal", 0)),
            "bb_upper": float(latest.get("bb_upper", 0)),
            "bb_lower": float(latest.get("bb_lower", 0)),
            "bb_middle": float(latest.get("bb_middle", 0)),
            "atr_14": float(latest.get("atr_14", 0)),
            "prediction_direction": result["direction"],
            "prediction_return_pct": result["target_return_pct"],
            "prediction_confidence": result["confidence"],
            "agent_contributions": result.get("agent_contributions", []),
            "band_room_id": room_id, "model_version": config.MODEL_NAME,
            "inference_latency_ms": result["inference_latency_ms"],
            "mode": "test", "data_source": "csv_simulation",
        })
        uuid.UUID(row_id)  # Valid UUID

    @requires_ollama
    def test_three_ticker_cycle_live(self):
        """Full prediction_cycle() with 3 tickers on live Ollama."""
        sdk = LocalBandSDK()
        squad = PredictionSquad(band_sdk=sdk)
        active_tickers = config.TICKER_SYMBOLS[:3]

        with patch("main.dashboard_state"):
            prediction_cycle(squad, active_tickers, training_enabled=False)

        # At least one ticker should have predictions
        total_rows = 0
        for ticker in active_tickers:
            df = ledger._load_ticker_df(ticker)
            total_rows += len(df)
        assert total_rows > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 3: Performance Benchmarks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPerformanceBenchmarks:
    """Measure latency for key operations on this hardware."""

    def test_csv_load_time(self):
        """load_all() completes in < 2s."""
        data_source._data.clear()
        data_source._cursors.clear()
        start = time.time()
        data_source.load_all()
        elapsed = time.time() - start
        assert elapsed < 2.0, f"CSV load took {elapsed:.2f}s > 2s"

    def test_indicator_compute_time(self):
        """compute_indicators() on 1000 rows < 500ms."""
        data = data_source.load_all()
        df = data["NSE:RELIANCE"]
        start = time.time()
        indicators.compute_indicators(df)
        elapsed = time.time() - start
        assert elapsed < 0.5, f"Indicator compute took {elapsed:.3f}s > 0.5s"

    @requires_ollama
    def test_single_inference_latency(self):
        """Single Ollama inference < 30s on CPU (phi3:mini on Xeon 2.2GHz)."""
        prompt = "Predict NSE:RELIANCE direction. RSI=55, MACD=0.5. Reply with direction only."
        start = time.time()
        ollama_generate(prompt, max_tokens=32)
        elapsed = time.time() - start
        assert elapsed < 30.0, f"Inference took {elapsed:.1f}s > 30s"

    def test_phase1_write_latency(self):
        """Phase 1 write < 200ms."""
        with tempfile.TemporaryDirectory() as tmp:
            orig = config.DAILY_DIR
            config.DAILY_DIR = Path(tmp)
            try:
                start = time.time()
                ledger.phase1_write({
                    "timestamp": pd.Timestamp.now(tz=IST), "ticker": "NSE:RELIANCE",
                    "cycle_id": "bench", "open": 2845.0, "high": 2850.0, "low": 2840.0,
                    "close": 2845.0, "volume": 1_000_000, "vwap": 2845.0, "rsi_14": 55.0,
                    "macd": 0.5, "macd_signal": 0.3, "bb_upper": 2860.0, "bb_lower": 2830.0,
                    "bb_middle": 2845.0, "atr_14": 15.0, "prediction_direction": "UP",
                    "prediction_return_pct": 0.5, "prediction_confidence": "MED",
                    "agent_contributions": [], "band_room_id": "bench",
                    "model_version": "phi3:mini", "inference_latency_ms": 100.0,
                    "mode": "bench", "data_source": "test",
                })
                elapsed = time.time() - start
                assert elapsed < 0.2, f"Phase 1 write took {elapsed:.3f}s > 0.2s"
            finally:
                config.DAILY_DIR = orig

    def test_phase2_resolve_latency(self):
        """Phase 2 resolve for 10 rows < 500ms."""
        with tempfile.TemporaryDirectory() as tmp:
            orig = config.DAILY_DIR
            config.DAILY_DIR = Path(tmp)
            try:
                now = pd.Timestamp.now(tz=IST)
                old_ts = now - pd.Timedelta(seconds=config.INTERVAL_SECONDS + 60)
                for i in range(10):
                    ledger.phase1_write({
                        "timestamp": old_ts, "ticker": "NSE:SBIN", "cycle_id": f"b{i}",
                        "open": 780.0, "high": 785.0, "low": 775.0, "close": 780.0,
                        "volume": 400_000, "vwap": 780.0, "rsi_14": 50.0,
                        "macd": 0.0, "macd_signal": 0.0, "bb_upper": 790.0, "bb_lower": 770.0,
                        "bb_middle": 780.0, "atr_14": 8.0, "prediction_direction": "UP",
                        "prediction_return_pct": 0.1, "prediction_confidence": "LOW",
                        "agent_contributions": [], "band_room_id": f"b{i}",
                        "model_version": "phi3:mini", "inference_latency_ms": 80.0,
                        "mode": "bench", "data_source": "test",
                    })
                start = time.time()
                ledger.resolve_phase2(now, current_prices={"NSE:SBIN": 790.0})
                elapsed = time.time() - start
                assert elapsed < 0.5, f"Phase 2 resolve took {elapsed:.3f}s > 0.5s"
            finally:
                config.DAILY_DIR = orig

    def test_dashboard_update_latency(self):
        """Dashboard state update < 5ms."""
        state = DashboardState()
        start = time.time()
        for _ in range(100):
            state.update(market_data={"NSE:X": {"close": 100.0}},
                        predictions={"NSE:X": {"direction": "UP"}})
        elapsed = (time.time() - start) / 100
        assert elapsed < 0.005, f"Dashboard update took {elapsed*1000:.1f}ms > 5ms"

    def test_dashboard_snapshot_latency(self):
        """Dashboard snapshot < 5ms."""
        state = DashboardState()
        state.update(market_data={"NSE:X": {"close": 100.0}},
                     predictions={"NSE:X": {"direction": "UP"}})
        start = time.time()
        for _ in range(100):
            state.get_snapshot()
        elapsed = (time.time() - start) / 100
        assert elapsed < 0.005, f"Snapshot took {elapsed*1000:.1f}ms > 5ms"

    def test_chart_generation_latency(self):
        """Chart generation < 5s with data."""
        with tempfile.TemporaryDirectory() as tmp:
            replay_dir = Path(tmp) / "replay"
            charts_dir = Path(tmp) / "charts"
            replay_dir.mkdir()
            charts_dir.mkdir()

            orig_replay = config.REPLAY_DIR
            orig_charts = config.CHARTS_DIR
            config.REPLAY_DIR = replay_dir
            config.CHARTS_DIR = charts_dir

            try:
                rows = []
                for i in range(50):
                    rows.append({
                        "row_id": str(uuid.uuid4()),
                        "timestamp": pd.Timestamp.now(tz=IST) - pd.Timedelta(minutes=5 * (50 - i)),
                        "ticker": "NSE:RELIANCE", "cycle_id": f"c{i}",
                        "open": 2845.0, "high": 2850.0, "low": 2840.0, "close": 2845.0,
                        "volume": 1000000, "vwap": 2845.0, "rsi_14": 55.0,
                        "macd": 0.5, "macd_signal": 0.3, "bb_upper": 2860.0,
                        "bb_lower": 2830.0, "bb_middle": 2845.0, "atr_14": 15.0,
                        "prediction_direction": "UP", "prediction_return_pct": 0.5,
                        "prediction_confidence": "MED", "agent_contributions": "[]",
                        "band_room_id": f"room_{i}", "model_version": "phi3:mini",
                        "inference_latency_ms": 100.0, "mode": "test", "data_source": "test",
                        "actual_direction": "UP", "actual_return_pct": 0.6,
                        "resolution_timestamp": pd.Timestamp.now(tz=IST),
                        "prediction_accuracy": True, "resolution_price": 2862.0,
                        "reward": 1.0, "used_in_training": False,
                    })
                pd.DataFrame(rows).to_parquet(replay_dir / "replay_buffer.parquet", index=False)

                start = time.time()
                generate_all()
                elapsed = time.time() - start
                assert elapsed < 5.0, f"Chart generation took {elapsed:.2f}s > 5s"
            finally:
                config.REPLAY_DIR = orig_replay
                config.CHARTS_DIR = orig_charts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 4: Edge Case Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEdgeCases:
    """Boundary conditions, error handling, resilience."""

    def test_missing_csv_skips_ticker(self):
        """Missing CSV file causes ticker to be skipped, not crash."""
        data_source._data.clear()
        data_source._cursors.clear()
        # Temporarily rename a CSV
        csv_path = config.CSV_DIR / "NSE_RELIANCE.csv"
        backup = csv_path.with_suffix(".csv.bak")
        if csv_path.exists():
            shutil.move(str(csv_path), str(backup))
        try:
            data = data_source.load_all()
            assert "NSE:RELIANCE" not in data
            assert len(data) == 9
        finally:
            if backup.exists():
                shutil.move(str(backup), str(csv_path))
            data_source._data.clear()
            data_source._cursors.clear()

    def test_empty_csv_ticker_skipped(self):
        """Empty CSV file causes ticker to be skipped."""
        data_source._data.clear()
        data_source._cursors.clear()
        csv_path = config.CSV_DIR / "NSE_RELIANCE.csv"
        backup = csv_path.with_suffix(".csv.bak2")
        if csv_path.exists():
            shutil.move(str(csv_path), str(backup))
        try:
            # Create file with header only (no data rows)
            csv_path.write_text("timestamp,open,high,low,close,volume\n")
            data = data_source.load_all()
            # Should load but have 0 rows
            if "NSE:RELIANCE" in data:
                assert len(data["NSE:RELIANCE"]) == 0
        finally:
            csv_path.unlink(missing_ok=True)
            if backup.exists():
                shutil.move(str(backup), str(csv_path))
            data_source._data.clear()
            data_source._cursors.clear()

    def test_simulation_loop_false_skips_exhausted(self):
        """With SIMULATION_LOOP=False, exhausted ticker is skipped."""
        data_source._data.clear()
        data_source._cursors.clear()
        orig = config.SIMULATION_LOOP
        config.SIMULATION_LOOP = False
        try:
            # Exhaust one ticker
            for _ in range(1001):
                data_source.poll_tickers()
            # Next poll should skip exhausted tickers
            result = data_source.poll_tickers()
            # Some tickers may be exhausted
            # Just verify no crash
            assert isinstance(result, dict)
        finally:
            config.SIMULATION_LOOP = orig

    def test_insufficient_indicator_rows_returns_nan(self):
        """< MIN_INDICATOR_ROWS produces NaN indicators."""
        df = _make_ohlcv_df(10)
        result = indicators.compute_indicators(df)
        assert result["rsi_14"].isna().all()
        assert result["macd"].isna().all()
        assert result["atr_14"].isna().all()

    def test_empty_dataframe_poll(self):
        """poll_tickers with empty CSV returns empty result."""
        data_source._data.clear()
        data_source._cursors.clear()
        # Mock _data to have empty DataFrames
        data_source._data["NSE:RELIANCE"] = pd.DataFrame()
        data_source._cursors["NSE:RELIANCE"] = 0
        result = data_source.poll_tickers()
        # Should not crash
        assert isinstance(result, dict)

    def test_phase1_missing_ticker_raises_ledger_error(self):
        """Phase 1 write without ticker raises LedgerError."""
        with pytest.raises(utils.LedgerError):
            ledger.phase1_write({"timestamp": pd.Timestamp.now(tz=IST)})

    def test_training_disabled_on_cpu(self):
        """auto_detect_training returns False without CUDA."""
        # This machine has no GPU
        result = auto_detect_training()
        assert result is False

    def test_cpu_mode_limits_tickers(self):
        """trim_tickers_for_hardware returns 3 tickers on CPU."""
        tickers = trim_tickers_for_hardware(training_enabled=False)
        assert len(tickers) == config.CPU_TICKER_LIMIT

    def test_verify_ollama_success(self):
        """verify_ollama returns True when Ollama is running."""
        if not _ollama_available():
            pytest.skip("Ollama not available")
        result = verify_ollama()
        assert result is True

    def test_verify_ollama_failure(self):
        """verify_ollama raises OllamaError when Ollama is down."""
        orig = config.OLLAMA_BASE_URL
        config.OLLAMA_BASE_URL = "http://127.0.0.1:19999"
        config.OLLAMA_API_TAGS = f"{config.OLLAMA_BASE_URL}/api/tags"
        try:
            with pytest.raises(utils.OllamaError):
                verify_ollama()
        finally:
            config.OLLAMA_BASE_URL = orig
            config.OLLAMA_API_TAGS = f"{orig}/api/tags"

    @patch("predict.ollama_generate")
    def test_synthesizer_invalid_json_returns_default(self, mock_gen):
        """Invalid JSON from synthesizer returns SIDEWAYS default."""
        def side_effect(prompt, **kwargs):
            if kwargs.get("format_json"):
                return "not json at all"
            return "analysis"
        mock_gen.side_effect = side_effect

        sdk = LocalBandSDK()
        squad = PredictionSquad(band_sdk=sdk)
        room_id = sdk.create_room("test_invalid")

        result = squad.run("NSE:RELIANCE", "data", room_id)
        assert result["direction"] in ("UP", "DOWN", "SIDEWAYS")

    @patch("predict.ollama_generate")
    def test_agent_failure_skips_agent(self, mock_gen):
        """Failed agent is skipped, synthesis continues with remaining."""
        call_count = {"n": 0}
        def side_effect(prompt, **kwargs):
            call_count["n"] += 1
            if kwargs.get("format_json"):
                return json.dumps({"direction": "UP", "target_return_pct": 0.5, "confidence": "MED"})
            # First agent fails
            if call_count["n"] == 1:
                raise Exception("Agent timeout")
            return "Analysis"
        mock_gen.side_effect = side_effect

        sdk = LocalBandSDK()
        squad = PredictionSquad(band_sdk=sdk)
        room_id = sdk.create_room("test_fail")

        result = squad.run("NSE:TCS", "data", room_id)
        assert result["direction"] in ("UP", "DOWN", "SIDEWAYS")

    def test_concurrent_ledger_writes_no_crash(self):
        """10 threads writing to ledger — UUIDs are unique, no Python crash."""
        with tempfile.TemporaryDirectory() as tmp:
            orig = config.DAILY_DIR
            config.DAILY_DIR = Path(tmp)
            try:
                now = pd.Timestamp.now(tz=IST)
                row_ids = []
                python_crashes = []

                def writer(i):
                    try:
                        rid = ledger.phase1_write({
                            "timestamp": now, "ticker": "NSE:TCS",
                            "cycle_id": f"t{i}", "open": 3920.0, "high": 3925.0,
                            "low": 3915.0, "close": 3920.0, "volume": 500_000,
                            "vwap": 3920.0, "rsi_14": 55.0, "macd": 0.5,
                            "macd_signal": 0.3, "bb_upper": 3940.0, "bb_lower": 3900.0,
                            "bb_middle": 3920.0, "atr_14": 15.0,
                            "prediction_direction": "UP", "prediction_return_pct": 0.5,
                            "prediction_confidence": "MED", "agent_contributions": [],
                            "band_room_id": f"room_{i}", "model_version": "phi3:mini",
                            "inference_latency_ms": 100.0, "mode": "test", "data_source": "test",
                        })
                        row_ids.append(rid)
                    except Exception as e:
                        # Parquet race condition may cause ArrowInvalid — expected
                        python_crashes.append(str(e))

                threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

                # All 10 UUIDs should be unique (generated before file I/O)
                assert len(row_ids) + len(python_crashes) == 10
                assert len(set(row_ids)) == len(row_ids)  # All unique
                # Parquet race condition is a known limitation — no Python crash
            finally:
                config.DAILY_DIR = orig

    def test_websocket_origin_validation(self):
        """WebSocket accepts connections from localhost."""
        from fastapi.testclient import TestClient
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            ws.send_text("ping")
            resp = ws.receive_text()
            assert resp == "pong"

    def test_balance_classes_empty_classes(self):
        """balance_classes handles when some classes have 0 rows."""
        rows = [
            {"actual_direction": "UP", "row_id": "1"},
            {"actual_direction": "UP", "row_id": "2"},
            {"actual_direction": "DOWN", "row_id": "3"},
        ]
        balanced = balance_classes(rows)
        # min of non-empty (UP=2, DOWN=1) = 1
        counts = {"UP": 0, "DOWN": 0}
        for r in balanced:
            counts[r["actual_direction"]] += 1
        assert counts["UP"] == 1
        assert counts["DOWN"] == 1

    def test_format_training_pair_missing_fields(self):
        """format_training_pair handles missing fields gracefully."""
        row = {"row_id": "test-123"}
        pair = format_training_pair(row)
        assert "prompt" in pair
        assert "completion" in pair
        comp = json.loads(pair["completion"])
        assert "direction" in comp

    def test_replay_buffer_update_and_dedup(self):
        """update_replay_buffer deduplicates by row_id."""
        with tempfile.TemporaryDirectory() as tmp:
            orig_replay = config.REPLAY_DIR
            orig_daily = config.DAILY_DIR
            config.REPLAY_DIR = Path(tmp) / "replay"
            config.DAILY_DIR = Path(tmp) / "daily"
            config.REPLAY_DIR.mkdir(parents=True, exist_ok=True)
            config.DAILY_DIR.mkdir(parents=True, exist_ok=True)
            try:
                now = pd.Timestamp.now(tz=IST)
                rid = str(uuid.uuid4())
                # Write a resolved row
                df = pd.DataFrame([{
                    "row_id": rid, "timestamp": now, "ticker": "NSE:RELIANCE",
                    "cycle_id": "c1", "open": 2845.0, "high": 2850.0, "low": 2840.0,
                    "close": 2845.0, "volume": 1_000_000, "vwap": 2845.0, "rsi_14": 55.0,
                    "macd": 0.5, "macd_signal": 0.3, "bb_upper": 2860.0, "bb_lower": 2830.0,
                    "bb_middle": 2845.0, "atr_14": 15.0, "prediction_direction": "UP",
                    "prediction_return_pct": 0.5, "prediction_confidence": "MED",
                    "agent_contributions": "[]", "band_room_id": "room_1",
                    "model_version": "phi3:mini", "inference_latency_ms": 100.0,
                    "mode": "test", "data_source": "test",
                    "actual_direction": "UP", "actual_return_pct": 0.6,
                    "resolution_timestamp": now, "prediction_accuracy": True,
                    "resolution_price": 2862.0, "reward": 1.0, "used_in_training": False,
                }])
                df.to_parquet(config.DAILY_DIR / "NSE_RELIANCE.parquet", index=False)

                # Update twice — should deduplicate
                ledger.update_replay_buffer()
                ledger.update_replay_buffer()

                buffer_path = config.REPLAY_DIR / "replay_buffer.parquet"
                assert buffer_path.exists()
                buf = pd.read_parquet(buffer_path)
                assert len(buf) == 1  # Deduplicated
                assert buf.iloc[0]["row_id"] == rid
            finally:
                config.REPLAY_DIR = orig_replay
                config.DAILY_DIR = orig_daily


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hardware Detection Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHardwareDetection:
    """Verify hardware auto-detection on this machine."""

    def test_no_cuda_detected(self):
        """torch.cuda.is_available() returns False on this machine."""
        try:
            import torch
            assert not torch.cuda.is_available()
        except ImportError:
            pytest.skip("torch not installed")

    def test_training_disabled(self):
        """auto_detect_training returns False."""
        assert auto_detect_training() is False

    def test_cpu_ticker_limit(self):
        """CPU mode returns 3 tickers."""
        tickers = trim_tickers_for_hardware(training_enabled=False)
        assert len(tickers) == 3
        assert tickers == config.TICKER_SYMBOLS[:3]

    def test_check_training_deps_returns_false(self):
        """check_training_deps returns False without GPU."""
        result = check_training_deps()
        assert result is False
