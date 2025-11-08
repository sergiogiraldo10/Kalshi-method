# NBA Momentum Trading Backtest - Project Summary

## 🎯 Project Goal

Test whether **momentum trading** is profitable in NBA games by:
1. Detecting scoring runs in real-time play-by-play data
2. Using machine learning to predict when runs will extend
3. Simulating realistic trades with fees, take-profit, and stop-loss
4. Backtesting on historical data to measure actual P/L

## ✅ Implementation Status: COMPLETE

All code modules have been implemented and are ready to execute!

### What's Been Built:

#### 1. Data Acquisition System (`src/data_acquisition.py`)
- Downloads 8 seasons of NBA play-by-play data using free nba_api
- Handles rate limiting and retries
- Processes ~10,000 games, ~2-3 million plays
- Status: **Running in background** (3-4 hours)

#### 2. Data Validation (`src/data_validation.py`)
- Validates timestamps, scores, and game progression
- Checks for data quality issues
- Generates validation reports
- Status: **Ready to run**

#### 3. Win Probability Model (`src/win_probability.py`)
- Logistic regression model to estimate win probability
- Trained on score differential, time remaining, and period
- Used to calculate trade entry/exit prices
- Status: **Ready to train**

#### 4. Feature Engineering (`src/feature_engineering.py`)
- Extracts momentum features from play-by-play data:
  - Current scoring runs (home/away)
  - Scoring pace (points per minute)
  - Game state (time, period, score differential)
  - Volatility indicators
- Creates training examples for the model
- Status: **Ready to run**

#### 5. Momentum Prediction Model (`src/train_model.py`)
- XGBoost classifier to predict run extensions
- Predicts: "Will a 4-0 run become a 10-0+ run?"
- Handles class imbalance
- Provides feature importance analysis
- Status: **Ready to train**

#### 6. Trading Simulator (`src/trading_simulator.py`)
- Realistic trading simulation with:
  - Position management (close conflicting positions)
  - Take profit: Exit at +8% win probability increase
  - Stop loss: Exit at -4% win probability decrease
  - Trading fees: 1.5% entry + 1.5% exit
  - Position sizing: 1% of bankroll per trade
- Tracks all trades with full details
- Calculates comprehensive metrics
- Status: **Ready to simulate**

#### 7. Backtesting Framework (`src/backtest.py`)
- Orchestrates the complete backtest
- Simulates real-time conditions (no future peeking)
- Processes games chronologically
- Integrates all models and simulators
- Status: **Ready to run**

#### 8. Results Visualization (`src/visualize_results.py`)
- Creates professional charts:
  - Cumulative P/L over time
  - Trade distribution histograms
  - Win/Loss pie charts
  - Exit reason analysis
  - Probability movement scatter plots
  - Holding time analysis
- Generates text summary reports
- Status: **Ready to visualize**

#### 9. Full Pipeline Runner (`src/run_full_pipeline.py`)
- Orchestrates all steps automatically
- Skips already-completed steps
- Shows progress for each phase
- Status: **Ready to run**

## 📊 Expected Workflow

```
Data Acquisition (3-4 hrs) 
    ↓
Data Validation (5-10 min)
    ↓
Win Probability Training (10-15 min)
    ↓
Feature Engineering (30-45 min)
    ↓
Momentum Model Training (15-20 min)
    ↓
Backtesting (20-30 min)
    ↓
Visualization (2-3 min)
    ↓
Results Analysis
```

**Total Time**: ~5-6 hours (mostly waiting for data download)

## 🔑 Key Features

### Trading System:
✅ Realistic fee structure (1.5% entry + 1.5% exit)  
✅ Position management (close before entering conflicting positions)  
✅ Take profit triggers (exit at favorable probability shifts)  
✅ Stop loss protection (limit downside risk)  
✅ Bankroll tracking  
✅ Detailed trade logging  

### Model Architecture:
✅ XGBoost for momentum prediction  
✅ Logistic regression for win probability  
✅ Feature engineering with domain knowledge  
✅ Time-series cross-validation  
✅ Class imbalance handling  

### Analysis:
✅ Win rate, ROI, Sharpe ratio  
✅ Maximum drawdown  
✅ Exit reason breakdown  
✅ Holding time analysis  
✅ Professional visualizations  

## 📁 Project Structure

```
Kalshi-method/
├── data/
│   ├── raw/              # Play-by-play data (downloading...)
│   └── processed/        # Extracted features (pending)
├── src/
│   ├── data_acquisition.py       ✅ COMPLETE
│   ├── data_validation.py        ✅ COMPLETE
│   ├── win_probability.py        ✅ COMPLETE
│   ├── feature_engineering.py    ✅ COMPLETE
│   ├── train_model.py           ✅ COMPLETE
│   ├── trading_simulator.py     ✅ COMPLETE
│   ├── backtest.py              ✅ COMPLETE
│   ├── visualize_results.py     ✅ COMPLETE
│   └── run_full_pipeline.py     ✅ COMPLETE
├── models/               # Trained models (pending)
├── results/              # Backtest results (pending)
├── README.md            ✅ COMPLETE
├── EXECUTION_GUIDE.md   ✅ COMPLETE
├── PROJECT_SUMMARY.md   ✅ COMPLETE
└── requirements.txt     ✅ COMPLETE
```

## 🎮 How to Execute

### Option 1: Automatic (Recommended)
Wait for data download to complete, then:
```bash
python src/run_full_pipeline.py
```

### Option 2: Manual (Step-by-step)
```bash
# Wait for data download, then:
python src/data_validation.py
python src/win_probability.py
python src/feature_engineering.py
python src/train_model.py
python src/backtest.py
python src/visualize_results.py
```

## 📈 Expected Outputs

After completion, you'll have:

### Models:
- `models/win_probability_model.pkl` - Win probability estimator
- `models/momentum_model.pkl` - Momentum prediction model

### Data:
- `data/raw/pbp_*.csv` - Raw play-by-play data (8 files)
- `data/processed/features_*.csv` - Extracted features (8 files)

### Results:
- `results/backtest_2023_24_trades.csv` - Every trade logged
- `results/backtest_2023_24_metrics.json` - Performance summary
- `results/pl_chart_2023_24.png` - Cumulative P/L chart
- `results/trade_distribution_2023_24.png` - Win/loss distribution
- `results/exit_reasons_2023_24.png` - Exit reason breakdown
- `results/probability_changes_2023_24.png` - Entry/exit probabilities
- `results/holding_time_2023_24.png` - Trade duration analysis
- `results/summary_report_2023_24.txt` - Text summary

## 🎯 Success Metrics

The system will be considered successful if:

| Metric | Minimum Target | Great Result |
|--------|---------------|--------------|
| Model Precision | >60% | >70% |
| Win Rate | >50% | >55% |
| Net ROI | Break-even | >5% |
| Sharpe Ratio | >0.5 | >1.0 |
| Max Drawdown | <20% | <10% |

## ⚠️ Important Notes

1. **This is NOT live trading** - It's a backtest simulation
2. **No real money at risk** - All trades are simulated
3. **Win probabilities are estimated** - Not from real betting markets
4. **Past performance doesn't guarantee future results**
5. **This is a research/educational project**

## 🔧 Customization Options

You can adjust these parameters in `src/backtest.py`:

```python
simulator_config = {
    'initial_bankroll': 10000,       # Change starting capital
    'position_size_pct': 0.01,       # Change position size
    'entry_fee_pct': 0.015,          # Adjust entry fees
    'exit_fee_pct': 0.015,           # Adjust exit fees
    'take_profit_pct': 0.08,         # Change TP threshold
    'stop_loss_pct': -0.04,          # Change SL threshold
    'min_momentum_confidence': 0.65  # Adjust entry confidence
}
```

## 📚 Documentation

- **README.md** - Full project documentation and quick start
- **EXECUTION_GUIDE.md** - Detailed execution timeline and troubleshooting
- **PROJECT_SUMMARY.md** - This file - high-level overview
- **Plan file** - Original project plan with implementation details

## 🚀 Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Data Acquisition | ⏳ In Progress | Running in background (3-4 hrs) |
| All Python Modules | ✅ Complete | 9 modules fully implemented |
| Dependencies | ✅ Installed | All packages ready |
| Documentation | ✅ Complete | 3 comprehensive guides |
| Testing | ⏸️ Pending | Awaiting data completion |

## ⏭️ Next Steps

1. ⏳ **Wait for data download** (~3-4 hours remaining)
2. 🎬 **Execute pipeline**: Run `python src/run_full_pipeline.py`
3. 📊 **Analyze results**: Review charts and metrics
4. 🔄 **Experiment**: Try different parameters
5. 📈 **Iterate**: Improve model or strategy based on findings

## 💡 Research Questions to Answer

Once the backtest completes, you'll be able to answer:

1. **Does momentum trading work?** 
   - Is the net ROI positive after fees?

2. **What runs are most predictive?**
   - Feature importance will show which features matter most

3. **How much do fees impact profitability?**
   - Compare results with different fee structures

4. **What's the optimal take-profit/stop-loss?**
   - Experiment with different TP/SL thresholds

5. **Is the strategy consistent?**
   - Test on 2024-25 season to validate

## 🏁 Conclusion

The complete NBA Momentum Trading Backtest system is **fully implemented and ready to execute**. All that remains is waiting for the data download to complete, then running the pipeline to generate results.

**Estimated time to first results**: 4-6 hours from now.

---

*System built with Python, XGBoost, scikit-learn, pandas, and nba_api.*  
*For educational and research purposes only.*

