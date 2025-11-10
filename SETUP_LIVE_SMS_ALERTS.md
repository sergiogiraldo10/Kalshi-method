# 📱 LIVE SMS ALERT SETUP GUIDE

## Get automatic text messages when high-confidence trade signals appear!

---

## 🚀 QUICK SETUP (5 Steps)

### **Step 1: Download 2025-26 Season Data**
```bash
python download_2025_26_season.py
```
This gets all games played so far this season to train the model.

---

### **Step 2: Extract Features**
```bash
python extract_features_2025_26.py
python add_team_features_2025_26.py
```
This processes the raw data into model features (~5 minutes).

---

### **Step 3: Train the Model**
```bash
python train_for_live_alerts.py
```
Trains on ALL historical data (2021-2026) for maximum accuracy.

---

### **Step 4: Setup Twilio (FREE)**

#### 4a. Sign up at twilio.com
- Go to https://www.twilio.com/try-twilio
- Sign up for free trial (get $15 credit)
- No credit card required for trial!

#### 4b. Get a phone number
- In Twilio console, click "Get a Trial Number"
- Copy this number (format: +12345678901)

#### 4c. Get your credentials
- In Twilio console, find your:
  - **Account SID** (starts with "AC...")
  - **Auth Token** (click to reveal)

#### 4d. Edit auto_sms_monitor.py
Open `auto_sms_monitor.py` and replace:
```python
TWILIO_ACCOUNT_SID = 'YOUR_ACCOUNT_SID_HERE'
TWILIO_AUTH_TOKEN = 'YOUR_AUTH_TOKEN_HERE'  
TWILIO_PHONE_NUMBER = '+12345678901'  # Your Twilio number
```

---

### **Step 5: Start Monitoring!**
```bash
python auto_sms_monitor.py
```

**That's it!** The system will now:
- ✅ Monitor all live NBA games
- ✅ Detect 6-0 runs in real-time
- ✅ Calculate win probability
- ✅ Send SMS when confidence ≥ 34.5%

---

## 📱 SMS ALERT FORMAT

You'll receive texts like this:

```
🏀 IGNITION AI ALERT 🏀

LAL @ GSW
Score: 45-38 LAL leading
Q2 5:30

RUN: LAL on 6-0 run
Win Prob: 62.5%
Confidence: 35.2% (TOP 20%)
Quality: 65/100

TRADE:
BUY: LAL to win
Size: $50
TP: +25% ($62.50)
SL: -5% ($47.50)

Expected: +$2.18

ACT FAST!
```

---

## ⚙️ CONFIGURATION

Edit `auto_sms_monitor.py` to customize:

```python
# Alert thresholds
MIN_CONFIDENCE = 0.345  # Top 20% (recommended)
MIN_QUALITY = 60        # Minimum quality score

# Position sizing
POSITION_SIZE_PCT = 0.05  # 5% of bankroll
TAKE_PROFIT_PCT = 0.25    # +25%
STOP_LOSS_PCT = -0.05     # -5%
```

---

## 🔍 HOW IT WORKS

### Real-Time Monitoring Loop:
```
1. Every 30 seconds:
   - Fetch all live games from NBA API
   - Get latest play-by-play for each game
   
2. For each game:
   - Detect scoring runs (6-0, 7-0, 8-0, etc.)
   - Calculate quality score (steals, blocks, 3pts)
   - Get model prediction
   
3. If high-confidence signal:
   - Calculate win probability
   - Determine position size
   - Send SMS alert to your phone
   
4. Track alerted runs (don't spam same run twice)
```

---

## 💰 COST

### Twilio Pricing:
- **Free Trial:** $15 credit (enough for ~150-200 SMS)
- **After Trial:** $0.0075 per SMS (~$0.75 per 100 messages)
- **Per Month:** ~$1-5 depending on usage

### Expected SMS Volume:
- Average night: 2-5 trade signals
- Busy night: 8-12 trade signals
- Full season: ~300-500 total signals

**Cost per season: ~$3-5** 💵

---

## ⚠️ IMPORTANT NOTES

### SMS Delivery Time:
- Usually instant (< 5 seconds)
- During high volume: up to 30 seconds
- **Momentum is time-sensitive** - have betting app ready!

### False Alerts:
- Model has 36% win rate (64% lose)
- Not every alert will be profitable
- **Profitability comes from asymmetric exits** (win big, lose small)

### Monitoring Requirements:
- Must keep script running during games
- Needs internet connection
- Uses ~50MB data/hour

### Best Practices:
1. Start script 30 min before games
2. Keep phone volume ON
3. Have betting account logged in
4. Set up one-click trade execution
5. Track all trades (wins/losses)

---

## 🐛 TROUBLESHOOTING

### "No live games" message?
- Games haven't started yet
- Check NBA schedule
- Script checks every 60 seconds

### SMS not arriving?
- Check Twilio console for errors
- Verify phone number format (+19732948219)
- Check trial account limits

### Model not found?
- Run `python train_for_live_alerts.py`
- Wait 5-10 minutes for training

### Connection errors?
- Check internet connection
- NBA API might be slow (retry in 60s)
- Rate limiting (script handles this)

---

## 📊 EXPECTED PERFORMANCE

Based on validation:
- **Win Rate:** 35-36%
- **Avg Win:** +$9 per trade
- **Avg Loss:** -$4 per trade
- **Monthly Return:** +8-15%
- **Signals per week:** 15-25

---

## 🎯 DEPLOYMENT CHECKLIST

Before going live with real money:

- [ ] Downloaded 2025-26 data
- [ ] Trained model on all historical data
- [ ] Twilio account setup
- [ ] Tested SMS (send yourself a test)
- [ ] Betting account ready
- [ ] Small bankroll ($100-500 to start)
- [ ] Tracking spreadsheet setup
- [ ] Stop-loss discipline confirmed

---

## 📞 SUPPORT

**Questions?** Check:
1. This README
2. `FINAL_VALIDATION_RESULTS.md` (strategy details)
3. `README_FINAL.md` (overview)

**SMS Issues?** 
- Twilio docs: https://www.twilio.com/docs/sms
- Trial limits: https://www.twilio.com/docs/usage/tutorials/how-to-use-your-free-trial-account

---

## 🚨 RISK WARNING

This system sends TRADE SIGNALS, not GUARANTEES.

- ⚠️ You WILL lose money on 64% of trades
- ⚠️ Past performance ≠ future results  
- ⚠️ Start small ($100-500)
- ⚠️ Never trade money you can't afford to lose
- ⚠️ Model performance may degrade over time

**This is trading, not gambling - but you can still lose!**

---

*Last updated: November 9, 2025*
*Model trained on 6,912+ NBA games*
*Validated win rate: 35.8%*

