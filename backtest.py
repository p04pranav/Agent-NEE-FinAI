#!/usr/bin/env python3
"""
Agent-NEE FinAI — Backtest Runner
Processes historical CSV data through the multi-agent prediction pipeline.
Bypasses market-hours gate for offline backtesting.
"""

import sys
import time
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path

import config
import utils
import indicators
import ledger
from agents import band, AGENT_ROLES, SYNTHESIS_PROMPT
from predict import ollama_generate, verify_ollama

logger = logging.getLogger("agent_nee")

SAMPLE_EVERY = 20       # Process every Nth row (970→490 predictions)
START_ROW = 30          # Skip first 30 rows (need indicator warmup)
PHASE2_OFFSET = 1       # Use next sampled row's close for resolution
AGENT_MAX_TOKENS_BT = 200  # Reduced from 512 for faster inference


def run_backtest():
    """Run full backtest across all tickers."""
    log = utils.setup_logging()
    log.info("=" * 60)
    log.info("Agent-NEE FinAI — BACKTEST MODE")
    log.info("=" * 60)

    # Verify Ollama
    try:
        verify_ollama()
    except Exception as e:
        log.critical(f"Ollama not ready: {e}")
        sys.exit(1)

    # Create dirs
    for d in [config.LOG_DIR, config.DATA_DIR, config.DAILY_DIR,
              config.REPLAY_DIR, config.MODELS_DIR, config.CHARTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Clean old ledger files
    for f in config.DAILY_DIR.glob("*.parquet"):
        f.unlink()
    replay_path = config.REPLAY_DIR / "replay_buffer.parquet"
    if replay_path.exists():
        replay_path.unlink()

    # Load all CSV data
    log.info("Loading CSV data...")
    all_data = {}
    for ticker in config.TICKER_SYMBOLS:
        filename = ticker.replace(":", "_") + ".csv"
        path = config.CSV_DIR / filename
        if not path.exists():
            log.warning(f"CSV not found: {path}")
            continue
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        all_data[ticker] = df
        log.info(f"  {ticker}: {len(df)} rows loaded")

    if not all_data:
        log.critical("No CSV data found!")
        sys.exit(1)

    # Compute indicators for each ticker
    log.info("Computing indicators...")
    indicator_data = {}
    for ticker, df in all_data.items():
        df_ind = indicators.compute_indicators(df)
        indicator_data[ticker] = df_ind
        log.info(f"  {ticker}: indicators computed ({len(df_ind)} rows)")

    # Sample rows for prediction
    total_predictions = 0
    total_latency = 0
    start_time = time.time()

    for ticker, df_ind in indicator_data.items():
        n_rows = len(df_ind)
        sample_indices = list(range(START_ROW, n_rows, SAMPLE_EVERY))
        log.info(f"\n{'─'*50}")
        log.info(f"Processing {ticker}: {len(sample_indices)} samples (rows {START_ROW}-{n_rows-1}, every {SAMPLE_EVERY})")
        log.info(f"{'─'*50}")

        for i, idx in enumerate(sample_indices):
            row = df_ind.iloc[idx]
            if pd.isna(row.get("rsi_14")):
                continue

            # Format data for agents
            ticker_data = indicators.format_market_data_for_agent(ticker, row)

            # Run multi-agent prediction
            room_id = band.create_room(f"backtest_{ticker}_{idx}")
            agent_analyses = []
            agent_latency = 0

            for role_key in config.AGENT_ROLES_TO_RUN:
                role = AGENT_ROLES[role_key]
                prompt = f"{role['prompt']}\n\n{ticker_data}"
                t0 = time.time()
                try:
                    response = ollama_generate(
                        prompt=prompt,
                        temperature=role["temperature"],
                        max_tokens=AGENT_MAX_TOKENS_BT,
                    )
                    lat = (time.time() - t0) * 1000
                    agent_latency += lat
                    band.send_message(room_id, role["name"], response)
                    agent_analyses.append(f"--- {role['name']} ---\n{response}")
                except Exception as e:
                    lat = (time.time() - t0) * 1000
                    agent_latency += lat
                    log.warning(f"  {ticker} row {idx} {role['prefix']} failed: {e}")
                    agent_analyses.append(f"--- {role['name']} --- [FAILED]")

            # Synthesize
            synth_prompt = SYNTHESIS_PROMPT.format(
                agent_analyses="\n\n".join(agent_analyses)
            )
            t0 = time.time()
            try:
                synth_response = ollama_generate(
                    prompt=synth_prompt,
                    temperature=config.SYNTHESIS_TEMPERATURE,
                    max_tokens=config.SYNTH_MAX_TOKENS,
                    format_json=True,
                )
                synth_latency = (time.time() - t0) * 1000
                agent_latency += synth_latency
                prediction = json.loads(synth_response)
                direction = prediction.get("direction", "SIDEWAYS").upper()
                target_return = float(prediction.get("target_return_pct", 0.0))
                confidence = prediction.get("confidence", "LOW").upper()
            except Exception as e:
                synth_latency = (time.time() - t0) * 1000
                agent_latency += synth_latency
                log.warning(f"  {ticker} row {idx} Synthesizer failed: {e}")
                direction, target_return, confidence = "SIDEWAYS", 0.0, "LOW"

            # Clamp values
            if direction not in ("UP", "DOWN", "SIDEWAYS"):
                direction = "SIDEWAYS"
            if confidence not in ("LOW", "MED", "HIGH"):
                confidence = "LOW"

            total_latency += agent_latency
            total_predictions += 1

            # Phase 1 write
            ledger.phase1_write({
                "timestamp": pd.Timestamp.now(tz=pd.Timestamp.now().tz) if pd.Timestamp.now().tz else pd.Timestamp.now(),
                "ticker": ticker,
                "cycle_id": f"backtest_{idx}",
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": float(row.get("volume", 0)),
                "vwap": float(row.get("vwap", 0)) if not pd.isna(row.get("vwap")) else 0.0,
                "rsi_14": float(row.get("rsi_14", 0)) if not pd.isna(row.get("rsi_14")) else 0.0,
                "macd": float(row.get("macd", 0)) if not pd.isna(row.get("macd")) else 0.0,
                "macd_signal": float(row.get("macd_signal", 0)) if not pd.isna(row.get("macd_signal")) else 0.0,
                "bb_upper": float(row.get("bb_upper", 0)) if not pd.isna(row.get("bb_upper")) else 0.0,
                "bb_lower": float(row.get("bb_lower", 0)) if not pd.isna(row.get("bb_lower")) else 0.0,
                "bb_middle": float(row.get("bb_middle", 0)) if not pd.isna(row.get("bb_middle")) else 0.0,
                "atr_14": float(row.get("atr_14", 0)) if not pd.isna(row.get("atr_14")) else 0.0,
                "prediction_direction": direction,
                "prediction_return_pct": target_return,
                "prediction_confidence": confidence,
                "agent_contributions": band.get_room_history(room_id),
                "band_room_id": room_id,
                "model_version": config.MODEL_NAME,
                "inference_latency_ms": round(agent_latency, 1),
                "mode": "backtest",
                "data_source": "csv_simulation",
            })

            band.cleanup_room(room_id)

            # Progress
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - start_time
                rate = total_predictions / max(elapsed, 1)
                remaining = (len(sample_indices) * len(all_data) - total_predictions) / max(rate, 0.01)
                log.info(
                    f"  [{i+1}/{len(sample_indices)}] {direction} {target_return:+.2f}% {confidence} "
                    f"({agent_latency:.0f}ms) | Total: {total_predictions} | "
                    f"ETA: {remaining/60:.1f}min"
                )

    # Phase 2 resolution: use next row's close as actual price
    log.info("\n" + "=" * 60)
    log.info("Phase 2 Resolution")
    log.info("=" * 60)

    resolved_total = 0
    for ticker, df_ind in indicator_data.items():
        df_ledger = ledger._load_ticker_df(ticker)
        if df_ledger.empty:
            continue

        unresolved_mask = df_ledger["actual_direction"].isna()
        unresolved = df_ledger[unresolved_mask]

        for idx, row in unresolved.iterrows():
            cycle_id = str(row.get("cycle_id", ""))
            if not cycle_id.startswith("backtest_"):
                continue

            # Extract the original sample index
            try:
                sample_idx = int(cycle_id.replace("backtest_", ""))
            except ValueError:
                continue

            # Get next row's close price as resolution price
            next_idx = sample_idx + SAMPLE_EVERY
            if next_idx >= len(df_ind):
                # Use last row if no next available
                next_idx = len(df_ind) - 1

            entry_price = float(row["close"])
            resolution_price = float(df_ind.iloc[next_idx]["close"])

            if entry_price > 0:
                actual_return = (resolution_price - entry_price) / entry_price * 100
            else:
                actual_return = 0.0

            if actual_return > config.SIDEWAYS_THRESHOLD_PCT:
                actual_direction = "UP"
            elif actual_return < -config.SIDEWAYS_THRESHOLD_PCT:
                actual_direction = "DOWN"
            else:
                actual_direction = "SIDEWAYS"

            pred_direction = str(row.get("prediction_direction", ""))
            accuracy = pred_direction == actual_direction

            df_ledger.at[idx, "actual_direction"] = actual_direction
            df_ledger.at[idx, "actual_return_pct"] = round(actual_return, 4)
            df_ledger.at[idx, "resolution_timestamp"] = pd.Timestamp.now()
            df_ledger.at[idx, "prediction_accuracy"] = accuracy
            df_ledger.at[idx, "resolution_price"] = round(resolution_price, 2)
            df_ledger.at[idx, "reward"] = 1.0 if accuracy else 0.0
            df_ledger.at[idx, "used_in_training"] = False
            resolved_total += 1

        ledger._save_ticker_df(ticker, df_ledger)

    log.info(f"Resolved {resolved_total} predictions")

    # Update replay buffer
    ledger.update_replay_buffer()

    # Summary
    elapsed_total = time.time() - start_time
    log.info("\n" + "=" * 60)
    log.info("BACKTEST COMPLETE")
    log.info("=" * 60)
    log.info(f"Total predictions: {total_predictions}")
    log.info(f"Total resolved: {resolved_total}")
    log.info(f"Total time: {elapsed_total/60:.1f} minutes")
    log.info(f"Avg latency: {total_latency/max(total_predictions,1):.0f}ms per prediction")
    log.info(f"Throughput: {total_predictions/max(elapsed_total,1)*60:.1f} predictions/min")

    # Quick accuracy stats
    completed = ledger.get_completed_rows()
    if not completed.empty:
        acc = completed["prediction_accuracy"].mean() * 100
        log.info(f"Overall accuracy: {acc:.1f}%")

        for conf in ["HIGH", "MED", "LOW"]:
            subset = completed[completed["prediction_confidence"] == conf]
            if not subset.empty:
                conf_acc = subset["prediction_accuracy"].mean() * 100
                log.info(f"  {conf} confidence: {conf_acc:.1f}% ({len(subset)} predictions)")

    log.info(f"\nReplay buffer: {config.REPLAY_DIR / 'replay_buffer.parquet'}")
    log.info(f"Ledger files: {config.DAILY_DIR}/")


if __name__ == "__main__":
    run_backtest()
