# Ignition AI - NBA Momentum Prediction System
## The $0-Cost Project Plan

---

## Executive Summary

**Goal:** Build a predictive AI model that identifies when a "micro-run" (e.g., 4-0) will extend into a "super-run" (e.g., 12-0) in NBA games.

**Constraint:** $0 cost - no paid APIs or data services.

**Strategy:** Use historical play-by-play data to train a model, backtest on unseen data, and create a demo that "replays" predictions.

---

## Phase 1: Kalshi Investigation ✅ (Current)

**Status:** Script ready to run

**Goal:** Determine if Kalshi API is a viable free data source

**Files:**
- `kalshi_simple.py` - Quick test script
- `kalshi_investigation.py` - Detailed analysis
- `QUICKSTART.md` - How to run

**Next Action:** Run the script and evaluate results

**Expected Outcome:** Most likely, Kalshi will be too slow → proceed to Phase 2

---

## Phase 2: Data Acquisition

### 2.1 Find Historical Play-by-Play Data

**Sources to explore:**
1. **Kaggle** - NBA play-by-play datasets
   - Search: "NBA play-by-play"
   - Look for: 2017-2024 seasons
   - Format preference: CSV with timestamps

2. **nba_api (Python library)**
   - Free official NBA stats API
   - Can download historical PBP
   - May require multiple requests

3. **Basketball Reference** (scraping fallback)
   - Has detailed play logs
   - Would need custom scraper
   - Check their ToS first

**Required fields:**
- Game ID
- Timestamp / Game Clock
- Event Type (shot, turnover, foul, etc.)
- Score before/after event
- Players involved
- Home/Away team

### 2.2 Data Validation

**Checklist:**
- [ ] At least 5 seasons of data
- [ ] Play-by-play granularity (not just box scores)
- [ ] Player IDs are consistent
- [ ] Timestamps are accurate
- [ ] All regular season games included

---

## Phase 3: Feature Engineering

### 3.1 Core Features

**Momentum Features:**
- Current run (e.g., 4-0, 6-2)
- Points scored in last N possessions
- Time since last opponent score
- Run started by 3-pointer vs 2-pointer
- Consecutive stops on defense

**Game State Features:**
- Current score differential
- Time remaining in quarter
- Quarter number
- Home vs Away
- Timeout availability

**Opponent State Features:**
- Foul trouble (players with 4+ fouls)
- Recent turnovers (last 2 minutes)
- Bench usage (starters' minutes)
- Fatigue indicators (back-to-back games)

**Team Context Features:**
- Season-to-date winning %
- Last 10 games record
- Days since last game
- Injuries (if available)

### 3.2 Handling the "Aging Problem"

**Challenge:** A stat from 3 years ago is less relevant

**Strategy 1: Time-Based Weighting**
```python
weight = exp(-lambda * years_ago)
# Recent seasons get higher weight
```

**Strategy 2: Rolling Windows**
- Only use last N games for team stats
- Recalculate features game-by-game

**Strategy 3: Season Normalization**
- Normalize features within each season
- Prevents old data from dominating

### 3.3 Handling the "Trade Problem"

**Challenge:** Player trades invalidate team stats

**Strategy 1: Player-Level Features**
- Focus on players currently on court
- Not team aggregates

**Strategy 2: Roster Tracking**
- Maintain roster snapshots per game
- Rebuild team stats when roster changes

**Strategy 3: Ignore Deep History**
- Only use last 20 games for team features
- Trades have minimal impact on recent data

---

## Phase 4: Model Development

### 4.1 Model Selection

**Option 1: XGBoost (RECOMMENDED)**

**Pros:**
- Handles categorical features well
- Fast training
- Built-in feature importance
- Less prone to overfitting

**Cons:**
- Requires feature engineering

**Use case:** Start here. Works great for structured data with clear patterns.

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.01,
    subsample=0.8
)
```

**Option 2: LightGBM**

**Pros:**
- Even faster than XGBoost
- Good with large datasets
- Handles categorical features natively

**Cons:**
- Can overfit on small datasets

**Use case:** If XGBoost is too slow or dataset is huge.

**Option 3: Neural Network (LSTM/Transformer)**

**Pros:**
- Can learn temporal patterns automatically
- Might discover hidden relationships

**Cons:**
- Requires more data
- Harder to interpret
- More prone to overfitting

**Use case:** If gradient boosting models plateau and you have 7+ seasons of data.

### 4.2 Training Strategy

**Data Split:**
- Training: 2017-2022 seasons (5 years)
- Validation: 2022-2023 season (1 year)
- Test: 2023-2024 season (1 year) - UNSEEN

**Target Variable:**

Binary classification:
- **Label 1:** Micro-run extends to super-run (e.g., 4-0 → 12-0)
- **Label 0:** Micro-run stops or reverses

**Threshold tuning:**
- Try different run lengths: 8-0, 10-0, 12-0, 15-0
- Choose based on win rate vs. frequency trade-off

**Cross-Validation:**
- Time-series CV (don't shuffle randomly)
- Each fold is a continuous time period

### 4.3 Evaluation Metrics

**Primary:**
- **Precision:** Of predicted super-runs, how many actually happened?
- **Recall:** Of actual super-runs, how many did we catch?
- **F1 Score:** Balance between precision and recall

**Secondary:**
- **ROC-AUC:** Overall classification performance
- **Profit/Loss:** Simulated betting performance

**Key Insight:**
High precision > High recall for this application.
We want to be RIGHT when we predict, even if we miss some opportunities.

---

## Phase 5: Backtesting

### 5.1 Backtesting Framework

**Simulate real-time conditions:**

```python
for game in test_season:
    for moment in game.timeline:
        if is_micro_run(moment):
            prediction = model.predict(features)
            if prediction == "super_run":
                # Log prediction
                # Wait to see actual outcome
                # Calculate P/L
```

**Constraints to simulate:**
- Can only use data available BEFORE the moment
- No peeking at future events
- Realistic reaction time (e.g., 30 second delay)

### 5.2 Output

**Full-Season P/L Chart:**
- X-axis: Game number (1-82)
- Y-axis: Cumulative profit/loss
- Show: Individual bets, win/loss markers

**Summary Stats:**
- Total predictions made: X
- Correct predictions: Y (Z%)
- Final P/L: $XXX (assuming $10/bet)
- Sharpe Ratio (if applicable)
- Max drawdown

### 5.3 Success Criteria

**Minimum viable:**
- Precision > 60%
- Recall > 30%
- Positive P/L over season

**Great result:**
- Precision > 70%
- Recall > 40%
- P/L > $500 on $10 bets

---

## Phase 6: Demo Dashboard

### 6.1 Technology Stack

**Backend:**
- Python (Flask or FastAPI)
- Model inference
- Data loading

**Frontend:**
- React (or simple HTML/CSS/JS)
- Chart.js for visualizations
- Real-time updates via polling

**Deployment:**
- Local only (no cost)
- Or: Streamlit (easiest, free)
- Or: Vercel/Netlify (frontend) + free backend

### 6.2 Features

**"Hero Game" Replay:**
1. Select a game from test set (pick exciting one)
2. Show play-by-play timeline
3. Highlight moments where model predicted
4. Show whether prediction was correct
5. Animate score changes in real-time

**Visual Elements:**
- Live score ticker
- Model confidence meter
- Feature importance panel
- Current run tracker
- Prediction history log

**Demo Flow:**
```
[Game Info: LAL @ GSW, Oct 24, 2023]
[Q2, 4:32 remaining]
[Score: LAL 45, GSW 48]

[DETECTION: Current Run = 4-0 GSW]
[MODEL ANALYZING...]

Features:
  ✓ Current run: 4-0
  ✓ Opponent foul trouble: 2 players with 3 fouls
  ✓ Recent turnovers: LAL 2 in last 2 min
  ✓ Time in quarter: Early Q2
  
[PREDICTION: 78% chance of SUPER-RUN (10-0+)]
[RECOMMENDED ACTION: Monitor closely]

[Actual Outcome: GSW went on 11-0 run]
[✓ CORRECT PREDICTION]
```

### 6.3 Implementation Priority

**MVP (Minimum Viable Product):**
1. Load one pre-selected "hero game"
2. Show timeline with predictions
3. Display win/loss for each prediction

**Nice-to-Have:**
1. Select any game from test set
2. Adjust playback speed
3. Interactive feature inspection
4. Compare multiple models

---

## Phase 7: Optimization & Iteration

### 7.1 Feature Importance Analysis

**Questions to answer:**
- Which features matter most?
- Are player-specific features useful?
- Does time of game matter?

**Tools:**
- SHAP values (for XGBoost)
- Feature importance plots
- Ablation studies (remove feature, measure drop)

### 7.2 Hyperparameter Tuning

**For XGBoost:**
- Tree depth
- Learning rate
- Number of trees
- Subsample ratio

**Method:**
- Grid search on validation set
- Bayesian optimization (if needed)

### 7.3 Ensemble Methods

**If single model isn't enough:**
1. Train multiple models with different features
2. Combine predictions (voting or averaging)
3. Use different algorithms (XGBoost + LightGBM)

---

## Timeline Estimate

| Phase | Estimated Time | Dependencies |
|-------|---------------|--------------|
| **Phase 1** | 1-2 hours | None - ready to run |
| **Phase 2** | 3-5 days | Finding & downloading data |
| **Phase 3** | 1-2 weeks | Clean data, build features |
| **Phase 4** | 1-2 weeks | Model training & tuning |
| **Phase 5** | 3-5 days | Backtesting framework |
| **Phase 6** | 1 week | Dashboard development |
| **Phase 7** | Ongoing | Optional improvements |

**Total: ~5-7 weeks** (part-time work)

---

## Risk Mitigation

### Risk 1: Can't find good free data
**Mitigation:** Use nba_api to download directly from NBA's API

### Risk 2: Model doesn't learn meaningful patterns
**Mitigation:** Start simple (basic features), iterate. Basketball has real momentum effects - they can be detected.

### Risk 3: Overfitting to historical data
**Mitigation:** Strict train/test split. Time-series cross-validation. Regularization.

### Risk 4: Demo is too complex to build
**Mitigation:** Use Streamlit. It's designed for quick data apps. Can build MVP in one day.

---

## Success Metrics (Final)

**Technical Success:**
- [x] Model trains without errors
- [x] Achieves >60% precision on test set
- [x] Backtest completes on full season

**Demonstration Success:**
- [x] Dashboard loads and plays game
- [x] Predictions are shown in real-time
- [x] At least 1 "hero game" with successful predictions

**Learning Success:**
- [x] Understand how momentum affects games
- [x] Know which features matter most
- [x] Have confidence in model's viability

---

## Future Enhancements (Post-MVP)

1. **Live Integration (Future):**
   - Connect to live game feeds
   - Make real-time predictions
   - Alert system

2. **Expanded Features:**
   - Player tracking data (speed, distance)
   - Shot chart analysis
   - Coaching tendencies

3. **More Sports:**
   - Apply to NFL, NHL, MLB
   - Same momentum concepts

4. **Trading Integration:**
   - If Kalshi (or similar) is viable
   - Automated bet placement
   - Risk management

---

## Resources & References

**Python Libraries:**
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `xgboost` / `lightgbm` - Modeling
- `scikit-learn` - Preprocessing, metrics
- `matplotlib` / `seaborn` - Visualization
- `streamlit` - Dashboard (optional)

**Learning Resources:**
- XGBoost documentation
- Time-series cross-validation guides
- NBA API documentation

**Data Sources:**
- Kaggle: NBA datasets
- GitHub: nba_api
- Basketball Reference

---

## Getting Started Checklist

- [ ] Run Kalshi investigation (Phase 1)
- [ ] Review results and decide if Kalshi is viable
- [ ] If not viable: Find historical PBP data (Phase 2)
- [ ] Download and validate data
- [ ] Set up Python environment
- [ ] Install required libraries
- [ ] Begin feature engineering (Phase 3)

---

**Remember: This is an exploratory project. The goal is learning and proving the concept, not perfection.**

Good luck! 🏀🚀

