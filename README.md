# NBA Momentum Trading Backtest System

## Overview

This system tests whether **momentum trading** works in NBA games by:
1. Detecting momentum runs in play-by-play data
2. Predicting when micro-runs (4-0) will extend to super-runs (10-0+)
3. Simulating realistic trading with position management, take-profit/stop-loss, and fees
4. Backtesting on historical seasons to generate P/L analysis

**Core Hypothesis**: Can we profit by buying "Team Wins" when a team starts a scoring run, based on momentum signals?

## Project Structure

```
Kalshi-method/
├── data/
│   ├── raw/              # Raw NBA play-by-play data
│   └── processed/        # Extracted features
├── src/
│   ├── data_acquisition.py       # Download NBA data
│   ├── data_validation.py        # Validate data quality
│   ├── win_probability.py        # Win probability model
│   ├── feature_engineering.py    # Extract momentum features
│   ├── train_model.py           # Train XGBoost model
│   ├── trading_simulator.py     # Trading simulation engine
│   ├── backtest.py              # Backtesting framework
│   ├── visualize_results.py     # Create charts
│   └── run_full_pipeline.py     # Run everything
├── models/               # Trained models
├── results/              # Backtest results and charts
└── requirements.txt      # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

The easiest way to run everything:

```bash
cd src
python run_full_pipeline.py
```

This will:
- Download 8 seasons of NBA data (~3-4 hours)
- Validate data quality
- Train win probability model
- Extract momentum features
- Train momentum prediction model
- Run backtest on 2023-24 season
- Generate visualizations

**Estimated total time**: 4-6 hours

### 3. Or Run Steps Individually

If you prefer to run steps one at a time:

```bash
# Step 1: Download data (takes 3-4 hours)
python src/data_acquisition.py

# Step 2: Validate data
python src/data_validation.py

# Step 3: Train win probability model
python src/win_probability.py

# Step 4: Extract features
python src/feature_engineering.py

# Step 5: Train momentum model
python src/train_model.py

# Step 6: Run backtest
python src/backtest.py

# Step 7: Create visualizations
python src/visualize_results.py
```

## How It Works

### 1. Data Acquisition
- Uses `nba_api` to download play-by-play data
- Covers 2017-2024 seasons (7 training + 1 test season)
- ~500MB of data, ~10,000 games

### 2. Win Probability Model
- Trained on historical games
- Predicts win probability from: score differential, time remaining, period
- Used to calculate trade entry/exit prices

### 3. Momentum Feature Engineering
Extracts features for each game moment:
- **Momentum features**: current run length, run differential
- **Scoring pace**: points scored in last 2/4 minutes
- **Game state**: time remaining, period, score differential
- **Contextual**: close game indicator, clutch time indicator

### 4. Momentum Prediction Model
- **Algorithm**: XGBoost (handles imbalanced data well)
- **Target**: Will current micro-run (4-0) extend to super-run (10-0+)?
- **Training**: 2017-2022 seasons
- **Validation**: 2022-2023 season
- **Test**: 2023-2024 season (UNSEEN)

### 5. Trading Simulation

#### Entry Rules:
- Enter when model predicts momentum extension (confidence ≥ 65%)
- Buy "Team Wins" for the team with momentum
- Close conflicting positions before entering new ones

#### Exit Rules:
- **Take Profit**: Exit when win probability increases by 8%
- **Stop Loss**: Exit when win probability drops by 4%
- **Game End**: Force close all positions

#### Realistic Constraints:
- Trading fees: 1.5% on entry, 1.5% on exit
- Position size: 1% of bankroll per trade
- Only one position per game (unless same direction)

### 6. Backtesting
- Simulates real-time conditions (no future peeking)
- Tracks bankroll evolution
- Logs all trades with entry/exit prices
- Calculates comprehensive performance metrics

### 7. Results Analysis
Generates:
- Cumulative P/L chart
- Trade distribution histogram
- Exit reason breakdown
- Win probability movement analysis
- Performance metrics (ROI, Sharpe ratio, max drawdown)

## Configuration

Edit parameters in `src/backtest.py` or `src/trading_simulator.py`:

```python
simulator_config = {
    'initial_bankroll': 10000,      # Starting capital
    'position_size_pct': 0.01,      # 1% per trade
    'entry_fee_pct': 0.015,         # 1.5% entry fee
    'exit_fee_pct': 0.015,          # 1.5% exit fee
    'take_profit_pct': 0.08,        # Exit at +8% prob change
    'stop_loss_pct': -0.04,         # Exit at -4% prob change
    'min_momentum_confidence': 0.65 # Min model confidence to enter
}
```

## Expected Results

### Success Criteria:
- **Model Precision**: >60% (ideally >70%)
- **Win Rate**: >55%
- **ROI**: Positive (break-even after fees)
- **Sharpe Ratio**: >1.0

### What to Expect:
This is an **experimental** system testing a hypothesis. Results may vary:
- ✅ If positive ROI: Momentum trading shows promise
- ❌ If negative ROI: Either the hypothesis is wrong OR parameters need tuning

## Important Notes

### This is NOT Live Trading
- This is a backtesting system only
- No real money is at risk
- Results are simulated based on historical data

### Limitations
- Win probabilities are estimated (not from real markets)
- Assumes instantaneous trade execution (30s delay simulated)
- Does not account for market depth or liquidity
- Past performance does not guarantee future results

### Data Usage
- Uses free NBA API (nba.com)
- Respects rate limits (600ms between requests)
- No paid data services required

## Troubleshooting

### Data Download Issues
If `data_acquisition.py` fails:
- Check internet connection
- NBA API may be rate limiting (wait 5 minutes and retry)
- Some games may fail - this is normal (script continues)

### Memory Issues
If you run out of memory:
- Process one season at a time
- Reduce the number of training seasons
- Use a machine with more RAM (recommended: 8GB+)

### Missing Dependencies
```bash
pip install --upgrade pandas numpy xgboost scikit-learn matplotlib seaborn nba_api
```

## Contributing

This is a research project. Feel free to:
- Experiment with different features
- Try different model architectures
- Adjust trading parameters
- Test on different seasons

## Results

After running the backtest, check:
- `results/summary_report_2023_24.txt` - Text summary
- `results/pl_chart_2023_24.png` - P/L visualization
- `results/backtest_2023_24_trades.csv` - Detailed trade log
- `results/backtest_2023_24_metrics.json` - Performance metrics

## License

This project is for educational and research purposes only.

## Acknowledgments

- NBA data provided by [nba_api](https://github.com/swar/nba_api)
- XGBoost for the gradient boosting framework
- scikit-learn for modeling utilities

---

**Questions or issues?** Check the code comments or open an issue.

**Good luck with your momentum trading research!** 🏀📈
