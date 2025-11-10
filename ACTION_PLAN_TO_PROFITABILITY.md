# Action Plan to Profitability

## Current Situation
- **Best backtest: -4.89%** (71 trades, 33.8% win rate)
- **Model accuracy: 32-38%** (need 45-50%)
- **Model is miscalibrated**: predicts 43% avg, reality is 32%
- **Higher confidence = worse results** (inverted relationship)

## Root Causes
1. ❌ **Model is overconfident** - predicts too many extensions
2. ❌ **Missing critical features** - no team quality, context data
3. ❌ **Class imbalance not handled** - only 32% positive samples
4. ❌ **Fees are too high** - eating 11-21% of capital

---

## Path to Profitability (3 Options)

### **OPTION 1: Improve the Model** ⭐ RECOMMENDED
**Goal**: Get win rate from 38% → 50%+

#### Steps:
1. **Add Team Context Features**
   - Team win percentage (season-to-date)
   - Offensive/Defensive rating
   - Recent form (last 5 games)
   - Home vs Away
   - Rest days (back-to-back games?)
   - Head-to-head history

2. **Add Player Context** (if available)
   - Are star players on court?
   - Injury status
   - Lineup changes

3. **Improve Run Detection**
   - Current definition: 5-0, 6-0 runs
   - Maybe focus on 6-0 ONLY (35.9% win rate vs 31.2%)
   - Add "momentum quality" (e.g., fast breaks, turnovers forced)

4. **Recalibrate Probabilities**
   - Use Platt scaling or isotonic regression
   - Current: model says 60% → reality is 29%
   - Need: model says 60% → reality is ~60%

5. **Better Training Strategy**
   - Time-series cross-validation (don't leak future data)
   - Adjust for class imbalance (32% positive class)
   - Use stratified sampling

**Expected Result**: Win rate 45-55%, profitable with 100-200 trades

---

### **OPTION 2: Trade Only the Best Signals**
**Goal**: Take fewer trades, but higher quality

#### Strategy:
- **Only trade 6-0 runs** (not 5-0)
- **Only in Q1-Q3** (Q4 has 30.1% win rate)
- **50-55% confidence range** (weird but it's what works!)
- **Asymmetric exits**: -5% SL, +25% TP
- **Small position size**: 3% of bankroll

**Expected Result**: 10-30 trades/season, +2-5% return if lucky

---

### **OPTION 3: Change the Fundamental Approach**
**Goal**: Different prediction target

Instead of predicting "will run extend?", predict:
- **"Will team win probability increase by 10%?"**
- **"Will team outscore opponent by 5+ in next 2 minutes?"**
- **"Will run last 90+ seconds?"**

These might be easier to predict and more directly tied to profitable trades.

---

## Quick Wins (Do These First)

### 1. **Focus on 6-0 runs only** (not 5-0)
   - Win rate: 35.9% vs 31.2%
   - Simple filter change

### 2. **Use 50-55% confidence range**
   - Counterintuitive, but the data shows it works better
   - Avoid 60%+ (it's worse!)

### 3. **Test asymmetric exits**
   - Stop loss: -5% (tight)
   - Take profit: +25-30% (wide)
   - Let winners run, cut losers fast

### 4. **Reduce position size to 3%**
   - Limits damage while we improve
   - Fees become less punishing

---

## Recommended Next Steps

**IMMEDIATE (today):**
1. Test "6-0 only + 50-55% confidence + asymmetric" strategy
2. If profitable → use this while improving model

**SHORT TERM (this week):**
1. Download team stats (win%, ratings)
2. Add team features to training data
3. Retrain model with better calibration
4. Backtest improved model

**MEDIUM TERM (next week):**
1. Add player lineup data if available
2. Test alternative prediction targets
3. Implement live monitoring/paper trading

---

## Success Criteria
- ✅ **Win rate: 45%+**
- ✅ **Return: +5% per season**
- ✅ **Sharpe ratio: 1.0+**
- ✅ **Max drawdown: <15%**
- ✅ **Consistent across 2-3 seasons**

---

## Fallback Plan
If after model improvements we still can't get profitable:
1. The fees might be too high for this strategy
2. NBA momentum runs might be too random to predict
3. Consider different markets (player props, halftime bets)

