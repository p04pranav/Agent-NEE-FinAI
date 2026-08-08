"""
Agent-NEE FinAI — Main Orchestrator
Entry point: hardware probe, event loop, market gate, prediction cycle.
"""

import sys
import time
import signal
import logging
import threading
import subprocess
import datetime
import webbrowser

import pandas as pd
from zoneinfo import ZoneInfo
import holidays

import config
import utils
import data_source
import indicators
import ledger
import charts
from agents import band
from predict import PredictionSquad, verify_ollama
from web_server import state as dashboard_state, start_web_server

IST = ZoneInfo("Asia/Kolkata")
INDIA_HOLIDAYS = holidays.India()
logger = logging.getLogger("agent_nee")


# ─── Market Hours (merged from market.py) ────────────────────────────

def is_trading_day(dt: datetime.datetime) -> bool:
    if dt.weekday() >= 5:
        return False
    if dt.date() in INDIA_HOLIDAYS:
        return False
    return True

def is_within_trading_hours(dt: datetime.datetime) -> bool:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    elif dt.tzinfo != IST:
        dt = dt.astimezone(IST)
    if not is_trading_day(dt):
        return False
    time_str = dt.strftime("%H:%M")
    if dt.weekday() == 4:
        return config.PREDICTION_START <= time_str < config.FRIDAY_EARLY_CLOSE
    return config.PREDICTION_START <= time_str < config.PREDICTION_END

def is_premarket_time(dt: datetime.datetime) -> bool:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    elif dt.tzinfo != IST:
        dt = dt.astimezone(IST)
    if not is_trading_day(dt):
        return False
    time_str = dt.strftime("%H:%M")
    return config.PREMARKET_WARMUP_TIME <= time_str < config.PREDICTION_START


# ─── Hardware Detection ──────────────────────────────────────────────

def auto_detect_training() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("Training disabled: No CUDA available")
            return False
        vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        cuda_version = torch.version.cuda
        if vram_gb < config.TRAINING_VRAM_MIN_GB:
            logger.info(f"Training disabled: {vram_gb:.1f} GB VRAM < {config.TRAINING_VRAM_MIN_GB} GB required")
            return False
        if cuda_version and cuda_version < config.TRAINING_CUDA_MIN_VERSION:
            logger.info(f"Training disabled: CUDA {cuda_version} < {config.TRAINING_CUDA_MIN_VERSION}")
            return False
        logger.info(f"Training ENABLED: {vram_gb:.1f} GB VRAM, CUDA {cuda_version}")
        return True
    except ImportError:
        logger.info("Training disabled: torch not installed")
        return False


def trim_tickers_for_hardware(training_enabled: bool) -> list[str]:
    try:
        import torch
        is_gpu = training_enabled or torch.cuda.is_available()
    except ImportError:
        is_gpu = False
    if is_gpu:
        active = config.TICKER_SYMBOLS[:config.GPU_TICKER_LIMIT]
    else:
        active = config.TICKER_SYMBOLS[:config.CPU_TICKER_LIMIT]
    logger.info(f"{'GPU' if is_gpu else 'CPU'} mode: {len(active)}/{len(config.TICKER_SYMBOLS)} tickers")
    return active


# ─── Startup ─────────────────────────────────────────────────────────

def startup():
    log = utils.setup_logging()
    log.info("=" * 50)
    log.info("Agent-NEE FinAI — Starting")
    log.info("=" * 50)

    for d in [config.LOG_DIR, config.DATA_DIR, config.DAILY_DIR,
              config.REPLAY_DIR, config.MODELS_DIR, config.CHARTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    training_enabled = auto_detect_training()

    try:
        verify_ollama()
    except utils.OllamaError as e:
        log.critical(str(e))
        sys.exit(1)

    log.info(f"CSV simulation: {config.CSV_DIR}")

    active_tickers = trim_tickers_for_hardware(training_enabled)
    squad = PredictionSquad(band)
    ledger.cleanup_orphans()
    return training_enabled, active_tickers, squad


# ─── Prediction Cycle ────────────────────────────────────────────────

def prediction_cycle(squad: PredictionSquad, active_tickers: list[str], training_enabled: bool):
    cycle_start = time.time()
    now = pd.Timestamp.now(tz=IST)

    dashboard_state.update(status={
        "mode": "TRAINING_ENABLED" if training_enabled else "INFERENCE_ONLY",
        "data_source": "csv_simulation",
        "active_tickers": f"{len(active_tickers)}/{len(config.TICKER_SYMBOLS)}",
        "ollama_model": config.MODEL_NAME,
    })

    # Poll market data FIRST
    market_data = data_source.poll_tickers()

    # Build fresh prices for Phase 2
    current_prices = {}
    for ticker in active_tickers:
        if ticker in market_data and not isinstance(market_data[ticker], str):
            df = market_data[ticker]
            if not df.empty:
                current_prices[ticker] = float(df.iloc[-1].get("close", 0))

    # Phase 2 resolution with fresh prices
    ledger.resolve_phase2(now, current_prices)

    # Dashboard market data
    dashboard_market = {}
    for ticker in active_tickers:
        if ticker in market_data and not isinstance(market_data[ticker], str):
            df = market_data[ticker]
            if not df.empty:
                latest = df.iloc[-1]
                dashboard_market[ticker.replace("NSE:", "")] = {
                    "open": float(latest.get("open", 0)),
                    "high": float(latest.get("high", 0)),
                    "low": float(latest.get("low", 0)),
                    "close": float(latest.get("close", 0)),
                    "volume": float(latest.get("volume", 0)),
                }
    dashboard_state.update(market_data=dashboard_market)

    # Process each ticker
    dashboard_predictions = {}
    total_latency = 0

    for ticker in active_tickers:
        if ticker not in market_data or isinstance(market_data[ticker], str):
            continue
        df = market_data[ticker]
        if df.empty:
            continue

        df_ind = indicators.compute_indicators(df)
        latest = df_ind.iloc[-1]
        if pd.isna(latest.get("rsi_14")):
            continue

        ticker_data = indicators.format_market_data_for_agent(ticker, latest)
        room_id = band.create_room(f"cycle_{ticker}_{int(time.time())}")
        result = squad.run(ticker, ticker_data, room_id)
        total_latency += result.get("inference_latency_ms", 0)

        ledger.phase1_write({
            "timestamp": now, "ticker": ticker,
            "cycle_id": f"cycle_{int(time.time())}",
            "open": float(latest.get("open", 0)),
            "high": float(latest.get("high", 0)),
            "low": float(latest.get("low", 0)),
            "close": float(latest.get("close", 0)),
            "volume": float(latest.get("volume", 0)),
            "vwap": float(latest.get("vwap", 0)) if not pd.isna(latest.get("vwap")) else 0.0,
            "rsi_14": float(latest.get("rsi_14", 0)) if not pd.isna(latest.get("rsi_14")) else 0.0,
            "macd": float(latest.get("macd", 0)) if not pd.isna(latest.get("macd")) else 0.0,
            "macd_signal": float(latest.get("macd_signal", 0)) if not pd.isna(latest.get("macd_signal")) else 0.0,
            "bb_upper": float(latest.get("bb_upper", 0)) if not pd.isna(latest.get("bb_upper")) else 0.0,
            "bb_lower": float(latest.get("bb_lower", 0)) if not pd.isna(latest.get("bb_lower")) else 0.0,
            "bb_middle": float(latest.get("bb_middle", 0)) if not pd.isna(latest.get("bb_middle")) else 0.0,
            "atr_14": float(latest.get("atr_14", 0)) if not pd.isna(latest.get("atr_14")) else 0.0,
            "prediction_direction": result["direction"],
            "prediction_return_pct": result["target_return_pct"],
            "prediction_confidence": result["confidence"],
            "agent_contributions": result.get("agent_contributions", []),
            "band_room_id": room_id,
            "model_version": config.MODEL_NAME,
            "inference_latency_ms": result["inference_latency_ms"],
            "mode": "training_enabled" if training_enabled else "inference_only",
            "data_source": "csv_simulation",
        })

        short_ticker = ticker.replace("NSE:", "")
        dashboard_predictions[short_ticker] = {
            "direction": result["direction"],
            "target_return_pct": result["target_return_pct"],
            "confidence": result["confidence"],
        }
        dashboard_state.update(agent_activity={
            "ticker": short_ticker,
            "agents": result.get("agent_contributions", []),
        })
        logger.info(f"PREDICTION | {ticker} | {result['direction']} | {result['target_return_pct']:+.2f}% | {result['confidence']} | {result['inference_latency_ms']:.0f}ms")
        band.cleanup_room(room_id)

    dashboard_state.update(predictions=dashboard_predictions)
    cycle_ms = (time.time() - cycle_start) * 1000
    dashboard_state.update(latency={
        "inference_avg_ms": round(total_latency / max(len(active_tickers), 1), 1),
        "total_cycle_avg_ms": round(cycle_ms, 1),
    })
    logger.info(f"Cycle complete: {cycle_ms:.0f}ms, {len(dashboard_predictions)} predictions")


# ─── Post-Market ─────────────────────────────────────────────────────

def post_market(training_enabled: bool):
    logger.info("Post-market sequence starting...")
    try:
        charts.generate_all()
    except Exception as e:
        logger.warning(f"Chart generation failed: {e}")

    if training_enabled:
        completed = ledger.get_completed_rows()
        if len(completed) >= config.MIN_TRAINING_ROWS:
            logger.info(f"Starting training ({len(completed)} rows)...")
            today = pd.Timestamp.now(tz=IST).strftime("%Y-%m-%d")
            try:
                result = subprocess.run(
                    [sys.executable, "learn.py", "--date", today],
                    capture_output=True, text=True, timeout=3600, check=False,
                )
                if result.returncode == 0:
                    logger.info("Training completed successfully")
                else:
                    logger.warning(f"Training failed: {result.stderr[:500]}")
            except Exception as e:
                logger.warning(f"Training subprocess error: {e}")
        else:
            logger.info(f"Training skipped: {len(completed)} rows < {config.MIN_TRAINING_ROWS}")
    else:
        logger.info("Post-market training skipped (inference-only mode)")

    ledger.update_replay_buffer()
    logger.info("Post-market sequence complete")


# ─── Event Loop ──────────────────────────────────────────────────────

def event_loop(squad: PredictionSquad, active_tickers: list[str], training_enabled: bool):
    last_prediction_time = None
    last_post_market = None
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        logger.info("Shutdown signal received")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    logger.info("Event loop starting...")

    while running:
        now = pd.Timestamp.now(tz=IST)
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M")

        if current_time >= "15:30" and last_post_market != current_date and is_trading_day(now):
            post_market(training_enabled)
            last_post_market = current_date

        if not is_within_trading_hours(now):
            time.sleep(60)
            continue

        if last_prediction_time is not None:
            elapsed = (now - last_prediction_time).total_seconds()
            if elapsed < config.INTERVAL_SECONDS:
                time.sleep(1)
                continue

        try:
            prediction_cycle(squad, active_tickers, training_enabled)
            last_prediction_time = pd.Timestamp.now(tz=IST)
        except (utils.OllamaError, utils.DataSourceError, utils.LedgerError) as e:
            logger.error(f"Prediction cycle failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)

        time.sleep(1)

    logger.info("Shutting down Agent-NEE...")
    ledger.flush()
    logger.info("Shutdown complete.")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    training_enabled, active_tickers, squad = startup()
    server = start_web_server()
    threading.Thread(target=server.run, daemon=True).start()
    logger.info(f"Web server started on http://localhost:{config.WEB_PORT}")
    if config.WEB_AUTO_OPEN_BROWSER:
        webbrowser.open(f"http://localhost:{config.WEB_PORT}/web/index.html")
    event_loop(squad, active_tickers, training_enabled)


if __name__ == "__main__":
    main()
