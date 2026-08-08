# Agent-NEE FinAI — To-Do

## Backtest & Update Performance Graphs

**Status**: Pending
**Priority**: High

### What

Run the system with real LLM inference on historical NSE data to produce
actual performance metrics and replace placeholder graphs in README.md,
SPEC.md, and Research_Paper.md.

### Prerequisites

- [ ] Ollama installed and running (`ollama serve`)
- [ ] phi3:mini model pulled (`ollama pull phi3:mini`)
- [ ] Python dependencies installed (`pip install -r requirements.txt`)
- [ ] GPU recommended (T4/3060+) for faster inference

### Steps

1. **Get real historical NSE data**
   - Download 30 trading days of 5-min OHLCV CSVs from NSE or Yahoo Finance
   - Place in `data/csv/` with format: `NSE_TICKER.csv`
   - Columns: `timestamp,open,high,low,close,volume`

2. **Run the backtest**
   ```bash
   python main.py
   ```
   - System will step through CSV data via `data_source.py`
   - Predictions logged to `data/daily/*.parquet`
   - Let it run for the full CSV duration (~30 trading days simulated)

3. **Collect results**
   - Replay buffer: `data/replay_buffer/replay_buffer.parquet`
   - Contains all predictions with actual outcomes (Phase 2 resolved)

4. **Generate real graphs**
   - Update `generate_graphs.py` to read from replay buffer instead of `np.random`
   - Run `python generate_graphs.py`
   - Replace `visuals/*.png` with real data

5. **Update documentation**
   - Remove "Backtest Pending" placeholders from README.md, SPEC.md, Research_Paper.md
   - Embed real graphs with actual measured results

### Metrics to Measure

| Metric | Source | Graph File |
|--------|--------|------------|
| Rolling 5-day accuracy | replay_buffer.prediction_accuracy | accuracy_over_time.png |
| Accuracy by confidence | replay_buffer grouped by prediction_confidence | confidence_calibration.png |
| Predicted vs actual scatter | replay_buffer.prediction_return_pct vs actual_return_pct | pred_vs_actual.png |
| Accuracy by ticker | replay_buffer grouped by ticker | accuracy_by_ticker.png |
| Cumulative return | Simulated from actual accuracy | cumulative_return.png |
| Training loss | learn.py training logs | training_loss.png |

### Validation Criteria

- [ ] Accuracy > 50% (better than random)
- [ ] HIGH confidence > MED confidence > LOW confidence (calibration works)
- [ ] At least 500 resolved predictions in replay buffer
- [ ] All 10 tickers represented
