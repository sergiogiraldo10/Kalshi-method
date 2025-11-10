# Momentum Trading Model - Improvements Needed

## Current Issues Found:

### 1. **RUN DETECTION IS WRONG** ❌
- **Current**: Summing ALL points in last 2 minutes (gives 20-30 points)
- **Should be**: Detecting actual momentum runs (8-0, 10-2, 12-4, etc.)
- **Example**: "8-0 run" means Team A scores 8 unanswered points

### 2. **MISSING CONTEXTUAL FEATURES** 
- We're only using raw scores
- NOT using play descriptions which tell us WHY runs happen:
  - Missed shots (defensive stops)
  - Steals & turnovers (creating fast breaks)  
  - Blocks (defensive momentum)
  - 3-pointers (spark quick runs)
  - Fouls (disrupting flow)

### 3. **KAGGLE DATA FORMAT**
- Already fixed SCORE format (0-0 vs 0 - 0)
- Need to verify play descriptions are consistent

---

## Proposed Improvements:

### PHASE 1: Fix Run Detection ✅

**New Algorithm:**
```python
# Detect if one team is on a hot streak
# Run = Team A outscores Team B significantly

Examples of VALID runs:
- 8-0 (perfect run)
- 10-2 (dominant)
- 12-4 (strong momentum)
- 6-0 (micro-run)

Run STOPS when:
- Opponent answers back (scores close to 1:1)
- Time threshold exceeded
```

**Criteria for detecting a run:**
1. One team scores 4+ points
2. That team outscores opponent by 2:1 ratio OR
3. That team scores 8+ and leads by 4+

**Run Extension (what we predict):**
- Micro-run (4-6 points) extends to bigger run (8-12+ points)
- Momentum continues with similar ratio

### PHASE 2: Add NLP Features ✅

**Parse play descriptions to extract:**

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| `opponent_misses_2min` | Missed shots by opponent | Defensive stops fuel runs |
| `steals_2min` | Steals in last 2 min | Create fast break opportunities |
| `blocks_2min` | Blocks in last 2 min | Defensive momentum |
| `turnovers_against_2min` | Opponent turnovers | Possession advantage |
| `threes_made_2min` | 3-pointers made | Quick scoring (6-9 points quickly) |
| `opponent_fouls_2min` | Opponent fouls | Disrupts their rhythm |
| `fastbreak_points_2min` | Fast break points | High-momentum scoring |
| `second_chance_points_2min` | Off rebounds | Extra possessions |

### PHASE 3: Re-Extract and Retrain

1. **Re-extract features** with fixed run detection + NLP features
2. **Retrain momentum model** with better features
3. **Backtest on 2023-24** season
4. **Analyze results** and create visualizations

---

## Expected Impact:

### Better Run Detection:
- ✅ Realistic run sizes (4-12 points, not 20-30)
- ✅ More balanced target (currently 15% positive, target 20-30%)
- ✅ Clearer signal for model to learn

### NLP Features:
- ✅ Understand WHY runs happen (defense, turnovers, hot shooting)
- ✅ Predict run continuation based on game dynamics, not just score
- ✅ Model AUC likely improves from 0.536 to 0.65+

### Trading Performance:
- ✅ More accurate entry signals
- ✅ Better timing for exits
- ✅ Positive expected value trades

---

## Questions for You:

1. **Run Definition**: Is the proposed algorithm (2:1 ratio or 8+ with 4+ lead) good? Or should it be stricter/looser?

2. **Time Window**: Currently using 2 minutes. Should we:
   - Keep 2 minutes?
   - Make it dynamic (1-3 min based on game pace)?
   - Use multiple windows (1 min, 2 min, 3 min)?

3. **NLP Features**: Do you want ALL 8 features listed above, or focus on most important (steals, blocks, misses)?

4. **Run Extension**: What should we predict?
   - Option A: "Will this 4-6 point run extend to 8-12+ points?"
   - Option B: "Will momentum continue for next X plays?"
   - Option C: "Will win probability increase by Y%?"

5. **Proceed?**: Should I implement all these changes now?

---

## Current Status:

- ✅ Data downloaded (2015-2023)
- ✅ Data validated and cleaned
- ❌ Feature extraction (NEEDS FIXING)
- ❌ Model training (NEEDS RETRAINING)
- ❌ Backtesting (PENDING)
- ❌ Visualization (PENDING)

**Estimated time for improvements: 2-3 hours (mostly feature extraction)**

