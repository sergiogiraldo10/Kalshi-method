# Quick Start Guide - Kalshi API Test

## 🚀 Fastest Way to Run

```bash
# Install the only dependency
pip install requests

# Run the simple version
python kalshi_simple.py
```

That's it! The script will guide you through the rest.

---

## What Will Happen

1. **Connection Test** (~2 seconds)
   - Script connects to Kalshi API
   - Tests both production and demo endpoints

2. **Market Search** (~3 seconds)
   - Searches for Lakers vs Hawks game
   - Shows you the market details

3. **Price Check** (~2 seconds)
   - Gets current YES bid, YES ask, last price
   - Shows trading volume and spread

4. **The Litmus Test** (~5 minutes)
   - You'll be asked if you want to proceed
   - If yes: monitors prices every 2 seconds for 5 minutes
   - If no: you can skip it
   
5. **Verdict**
   - Script tells you if market is FAST, MODERATE, or SLOW
   - Provides recommendation on viability

---

## Two Script Options

### Option 1: `kalshi_simple.py` (Recommended)
- ✅ Minimal dependencies (just `requests`)
- ✅ Easier to understand
- ✅ Faster to run
- ✅ No authentication needed

**Use this for your first test.**

### Option 2: `kalshi_investigation.py` (Advanced)
- More detailed output
- Better error handling
- Structured as a class (easier to extend)
- Can add authentication if needed

**Use this if you want to extend the code later.**

---

## Expected Results

### If Market Exists and Is Active:
```
✓ FOUND: Lakers vs Hawks - November 8, 2025
  Ticker: NBA-LAKERS-20251108
  
YES Bid:      48¢
YES Ask:      52¢
Last Price:   50¢
Spread:       4¢

VERDICT: MODERATE MARKET
Change Rate: 5.2%
```

### If Market Doesn't Exist:
```
✗ No market found for Lakers vs Hawks

Sample events found:
  1. Presidential Election 2024
  2. Fed Rate Decision November
  3. (other events...)
```

**This means:** Kalshi probably doesn't have this game, or it hasn't been listed yet.

---

## What to Do With Results

### ✅ FAST MARKET (>10% change rate)
**Action:** Kalshi might be viable, BUT:
- Consider fees
- Ask: Does "Game Winner" market reflect in-game momentum?
- Ask: Can you react faster than other traders?

**Recommendation:** Still consider the historical data approach for better control.

### ⚠️ MODERATE MARKET (1-10% change rate)
**Action:** Borderline case.
- Market moves, but slowly
- Probably not ideal for momentum plays

**Recommendation:** Proceed with historical data plan.

### ❌ SLOW MARKET (<1% change rate)
**Action:** Kalshi is NOT viable.
- Market is too static
- Too human-driven
- Can't react to live momentum

**Recommendation:** Definitely proceed with historical data plan.

---

## Troubleshooting

### Script says "Cannot connect"
- Check your internet connection
- Try again in a few minutes
- Kalshi API might be down (check status.kalshi.com)

### Script says "No market found"
- The game might not be listed yet (try closer to game time)
- Kalshi might not offer this specific game
- Double-check the Lakers are actually playing Hawks tonight

### Want to check a different game?
Edit line 88 in `kalshi_simple.py`:
```python
event = search_for_game(base_url, "Team1", "Team2")
```

---

## Next Steps After Testing

### If you want to proceed with Kalshi:
1. Create a Kalshi account
2. Get API credentials
3. Study their fee structure
4. Build a trading strategy

### If Kalshi isn't viable (most likely):
**Proceed with the "Ignition AI" original plan:**

1. **Get Data**
   - Download historical NBA play-by-play from Kaggle
   - 5-7 seasons for training
   - 1 season for testing

2. **Build Features**
   - Current run (e.g., 4-0)
   - Opponent foul trouble
   - Recent turnovers
   - Player fatigue indicators
   - Time remaining in quarter

3. **Train Model**
   - Try XGBoost first (handles categorical features well)
   - Or LightGBM (faster for large datasets)
   - Or neural network (if patterns are complex)

4. **Backtest**
   - Run on 2023-2024 season
   - Generate P/L chart
   - Calculate win rate

5. **Build Demo**
   - Web dashboard
   - "Replay" one exciting game
   - Show model making real-time calls

---

## Questions?

See the main README.md for more details.

**Remember:** This is a $0-cost investigation. No trades are made.

