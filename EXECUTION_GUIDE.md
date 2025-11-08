# Execution Guide - NBA Momentum Trading Backtest

## Current Status

✅ **All code modules have been created and are ready to run!**

The data acquisition is currently running in the background and will take approximately **3-4 hours** to download all 8 seasons of NBA play-by-play data.

## What Has Been Created

### Core Modules (in `src/` directory):

1. **`data_acquisition.py`** - Downloads NBA play-by-play data using nba_api
2. **`data_validation.py`** - Validates data quality and consistency
3. **`win_probability.py`** - Trains win probability estimation model
4. **`feature_engineering.py`** - Extracts momentum features from play-by-play
5. **`train_model.py`** - Trains XGBoost model to predict momentum extensions
6. **`trading_simulator.py`** - Simulates realistic trading with TP/SL and fees
7. **`backtest.py`** - Main backtesting framework
8. **`visualize_results.py`** - Creates charts and visualizations
9. **`run_full_pipeline.py`** - Orchestrates the entire pipeline

### Documentation:

- **`README.md`** - Complete project documentation
- **`EXECUTION_GUIDE.md`** - This file
- **`requirements.txt`** - Python dependencies

### Directory Structure:

```
Kalshi-method/
├── data/
│   ├── raw/              # NBA play-by-play data (downloading...)
│   └── processed/        # Will contain extracted features
├── src/                  # All Python modules (✅ COMPLETE)
├── models/               # Will contain trained models
├── results/              # Will contain backtest results
└── requirements.txt      # Dependencies (✅ INSTALLED)
```

## Execution Timeline

### Phase 1: Data Acquisition (IN PROGRESS - 3-4 hours)
Currently running in the background. Downloads 8 seasons of NBA data:
- 2017-18 through 2021-22 (training data)
- 2022-23 (validation data)
- 2023-24 (test data)
- 2024-25 (additional test data)

**Status**: ⏳ In Progress

### Phase 2: Data Validation (5-10 minutes)
Once data download completes, validate data quality:
```bash
python src/data_validation.py
```

### Phase 3: Win Probability Model (10-15 minutes)
Train model to estimate win probability:
```bash
python src/win_probability.py
```

### Phase 4: Feature Engineering (30-45 minutes)
Extract momentum features from all games:
```bash
python src/feature_engineering.py
```

### Phase 5: Momentum Model Training (15-20 minutes)
Train XGBoost model to predict run extensions:
```bash
python src/train_model.py
```

### Phase 6: Backtesting (20-30 minutes)
Run full backtest on 2023-24 season:
```bash
python src/backtest.py
```

### Phase 7: Visualization (2-3 minutes)
Generate charts and reports:
```bash
python src/visualize_results.py
```

## Easy Option: Run Everything Automatically

Instead of running each step manually, you can run the entire pipeline:

```bash
python src/run_full_pipeline.py
```

This script will:
- Check if each step is complete
- Skip steps that are already done
- Run only what's needed
- Show progress for each phase

**Total time**: ~4-6 hours (mostly waiting for data download)

## Monitoring Data Download

To check if data download is complete, look for files in `data/raw/`:

```bash
# On Windows PowerShell:
dir data\raw

# You should see files like:
#   pbp_2017_18.csv
#   pbp_2018_19.csv
#   pbp_2019_20.csv
#   ... etc
```

Each file should be approximately 50-70 MB.

## What to Expect

### After Data Download:
- 8 CSV files in `data/raw/` (~500MB total)
- ~10,000 games worth of play-by-play data
- ~2-3 million individual plays

### After Feature Engineering:
- 8 feature files in `data/processed/`
- Each row = one "game moment" where momentum could shift
- Features include: runs, scoring pace, game state, etc.

### After Model Training:
- `models/win_probability_model.pkl` (~1MB)
- `models/momentum_model.pkl` (~5-10MB)
- Console output showing model accuracy metrics

### After Backtesting:
- `results/backtest_2023_24_trades.csv` - Every trade logged
- `results/backtest_2023_24_metrics.json` - Performance summary
- `results/backtest_2023_24_bankroll.csv` - Bankroll over time

### After Visualization:
- **`results/pl_chart_2023_24.png`** - Cumulative P/L chart
- **`results/trade_distribution_2023_24.png`** - Win/loss distribution
- **`results/exit_reasons_2023_24.png`** - How trades exited (TP/SL/Game End)
- **`results/probability_changes_2023_24.png`** - Entry vs exit probabilities
- **`results/holding_time_2023_24.png`** - Trade duration analysis
- **`results/summary_report_2023_24.txt`** - Text summary of all metrics

## Key Performance Metrics

After backtesting, you'll see metrics like:

- **Total Trades**: How many trades were made
- **Win Rate**: % of profitable trades
- **Total P/L**: Net profit/loss after fees
- **ROI**: Return on investment (%)
- **Sharpe Ratio**: Risk-adjusted returns (>1.0 is good)
- **Max Drawdown**: Largest peak-to-trough decline
- **Average Win/Loss**: Average $ per winning/losing trade

## Configuration Options

You can adjust trading parameters in `src/backtest.py`:

```python
simulator_config = {
    'initial_bankroll': 10000,         # Starting capital
    'position_size_pct': 0.01,         # 1% of bankroll per trade
    'entry_fee_pct': 0.015,            # 1.5% entry fee
    'exit_fee_pct': 0.015,             # 1.5% exit fee
    'take_profit_pct': 0.08,           # Exit at +8% win prob increase
    'stop_loss_pct': -0.04,            # Exit at -4% win prob decrease
    'min_momentum_confidence': 0.65    # Minimum model confidence to enter
}
```

## Troubleshooting

### If Data Download Fails:
1. Check internet connection
2. Wait 5 minutes (may be rate limited)
3. Run `python src/data_acquisition.py` again - it will resume where it left off

### If You Run Out of Memory:
- Close other programs
- Process one season at a time (edit the season list in each script)
- Use a machine with more RAM (8GB+ recommended)

### If Models Don't Train:
- Ensure data files exist in `data/raw/`
- Check that feature files exist in `data/processed/`
- Look for error messages in the console output

## Next Steps After Completion

Once everything runs successfully:

1. **Analyze Results**: Open `results/summary_report_2023_24.txt`
2. **Review Charts**: Look at all PNG files in `results/`
3. **Examine Trades**: Open `results/backtest_2023_24_trades.csv` in Excel/Sheets
4. **Experiment**: Try different parameters in `simulator_config`
5. **Test Other Seasons**: Modify `src/backtest.py` to test on 2024-25 data

## Expected Timeline

| Phase | Time | Status |
|-------|------|--------|
| Data Acquisition | 3-4 hours | ⏳ In Progress |
| Data Validation | 5-10 min | ⏸️ Waiting |
| Win Prob Model | 10-15 min | ⏸️ Waiting |
| Feature Engineering | 30-45 min | ⏸️ Waiting |
| Model Training | 15-20 min | ⏸️ Waiting |
| Backtesting | 20-30 min | ⏸️ Waiting |
| Visualization | 2-3 min | ⏸️ Waiting |
| **TOTAL** | **~5 hours** | |

## Current Action Items

1. ⏳ **Wait for data download to complete** (~3-4 hours remaining)
2. ✅ All code is ready to execute
3. ✅ All dependencies are installed
4. 📋 When data is ready, run: `python src/run_full_pipeline.py`

## Questions?

- Check `README.md` for full documentation
- Review individual module docstrings for implementation details
- All code is commented and structured for readability

---

**Status**: System is fully implemented and ready. Waiting for data acquisition to complete.

**Estimated time to first results**: ~4-6 hours from now.

