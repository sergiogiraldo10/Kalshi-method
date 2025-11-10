# 📱 SMS OPTIONS & AUTO-START GUIDE

## ❓ **Can you just send me texts directly?**

**Short answer:** No, I'm an AI assistant - I can't send SMS myself.

**Why:** I (Cursor AI) run on your computer and don't have access to phone networks or SMS services. To send text messages, you need a **third-party SMS service**.

---

## 📲 **SMS OPTIONS (Pick One)**

### **Option 1: Twilio (RECOMMENDED)** ⭐
**Best for:** Reliable, professional SMS delivery

**Pros:**
- ✅ $15 free credit (trial)
- ✅ Most reliable delivery
- ✅ ~$0.0075 per SMS ($0.75 per 100 texts)
- ✅ Easy setup (5 minutes)

**Cons:**
- ❌ Requires credit card after trial
- ❌ Small monthly cost ($1-5/month)

**Setup:**
1. Sign up: https://www.twilio.com/try-twilio
2. Get trial number
3. Copy Account SID & Auth Token
4. Edit `auto_sms_monitor.py` with your credentials

---

### **Option 2: Email-to-SMS (FREE)** 📧
**Best for:** $0 cost, no signup

**How it works:** Send email to your carrier's SMS gateway

**Carriers:**
- AT&T: 9732948219@txt.att.net
- Verizon: 9732948219@vtext.com
- T-Mobile: 9732948219@tmomail.net
- Sprint: 9732948219@messaging.sprintpcs.com

**Setup:**
```python
# In auto_sms_monitor.py, replace Twilio code with:
import smtplib
from email.mime.text import MIMEText

def send_email_sms(message, phone_gateway):
    msg = MIMEText(message)
    msg['From'] = 'youremail@gmail.com'
    msg['To'] = phone_gateway
    msg['Subject'] = 'Trade Alert'
    
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login('youremail@gmail.com', 'your_app_password')
        server.send_message(msg)

# Usage:
send_email_sms(alert_text, '9732948219@txt.att.net')
```

**Pros:**
- ✅ Completely FREE
- ✅ No signup required
- ✅ Works forever

**Cons:**
- ❌ Slower delivery (30-60 seconds)
- ❌ May be blocked by carriers
- ❌ Requires email app password setup

---

### **Option 3: Pushbullet (FREE)** 📱
**Best for:** Push notifications to your phone app

**Pros:**
- ✅ FREE
- ✅ Fast delivery
- ✅ Desktop notifications too

**Cons:**
- ❌ Requires app install
- ❌ Not true SMS (app notification)

**Setup:**
1. Install Pushbullet app on phone
2. Get API key from pushbullet.com
3. Use Python API:
```python
from pushbullet import Pushbullet
pb = Pushbullet('YOUR_API_KEY')
pb.push_note("Trade Alert", message)
```

---

### **Option 4: Discord Webhook (FREE)** 💬
**Best for:** Alerts to your phone via Discord

**Pros:**
- ✅ Completely FREE
- ✅ Instant delivery
- ✅ Can share with friends

**Cons:**
- ❌ Requires Discord app
- ❌ Not true SMS

**Setup:**
1. Create Discord server
2. Create webhook in channel settings
3. Send alerts:
```python
import requests
webhook_url = 'YOUR_WEBHOOK_URL'
requests.post(webhook_url, json={'content': message})
```

---

## ⏰ **AUTO-START WITH TASK SCHEDULER**

### **Quick Setup (Windows)**

**1. Run the setup script:**
```bash
Right-click setup_auto_monitor.bat -> "Run as administrator"
```

This creates a scheduled task that:
- ✅ Runs every day at 5:00 PM ET
- ✅ Automatically starts monitoring games
- ✅ Sends alerts when high-confidence runs appear

**2. That's it!** The script will now run automatically.

---

### **Manual Setup (Windows)**

If the `.bat` file doesn't work:

**1. Open Task Scheduler:**
- Press `Win + R`
- Type `taskschd.msc`
- Press Enter

**2. Create New Task:**
- Click "Create Task" (right panel)
- Name: `IgnitionAI_AutoMonitor`

**3. Triggers Tab:**
- Click "New"
- Begin: "On a schedule"
- Settings: Daily, Start at **5:00 PM**
- ✅ Check "Enabled"

**4. Actions Tab:**
- Click "New"
- Action: "Start a program"
- Program: `python`
- Arguments: `C:\Users\123se\Kalshi\Kalshi-method\auto_sms_monitor.py`
- Start in: `C:\Users\123se\Kalshi\Kalshi-method\`

**5. Conditions Tab:**
- ✅ Check "Start only if computer is on AC power" (uncheck if laptop)
- ✅ Check "Wake computer to run this task"

**6. Click OK to save**

---

### **Verify It's Working**

**Test the scheduled task:**
```bash
schtasks /run /tn "IgnitionAI_AutoMonitor"
```

**View task status:**
```bash
schtasks /query /tn "IgnitionAI_AutoMonitor"
```

**View task logs:**
- Task Scheduler -> Task History (bottom panel)

---

## 🎯 **RECOMMENDED SETUP**

### **For Most Users:**

**SMS Method:** Twilio (professional, reliable)
**Auto-Start:** Task Scheduler at 5:00 PM

**Why:** 
- Twilio is reliable and only costs ~$3-5/month
- Task Scheduler ensures you never miss games
- Professional setup for serious trading

### **For Free/Hobby Users:**

**SMS Method:** Email-to-SMS (your carrier)
**Auto-Start:** Task Scheduler at 5:00 PM

**Why:**
- Completely FREE
- Works forever
- Good enough for testing

---

## 📋 **COMPLETE SETUP CHECKLIST**

### **First Time Setup:**
- [ ] Downloaded 2025-26 season data
- [ ] Trained model (python train_for_live_alerts.py)
- [ ] Chose SMS method (Twilio, Email, Pushbullet, or Discord)
- [ ] Edited auto_sms_monitor.py with credentials
- [ ] Tested manual run: python auto_sms_monitor.py
- [ ] Setup Task Scheduler (run setup_auto_monitor.bat)
- [ ] Verified task runs: schtasks /run /tn "IgnitionAI_AutoMonitor"

### **Daily Checklist:**
- [ ] Computer is on and connected to internet by 5 PM
- [ ] Phone is nearby with volume on
- [ ] Betting account is logged in
- [ ] Ready to trade!

---

## 🔧 **TROUBLESHOOTING**

### **Task Scheduler not running?**
- Check Task History for errors
- Verify Python path is correct
- Try running as Administrator
- Check if computer is on AC power

### **SMS not arriving?**
- Check credentials in auto_sms_monitor.py
- Verify phone number format (+19732948219)
- Check Twilio console for errors
- Test with a manual SMS send

### **Script crashes?**
- Check logs in terminal
- Verify internet connection
- Check NBA API rate limits
- Restart Task Scheduler

---

## 💡 **PRO TIPS**

### **Multiple Alert Methods:**
Set up BOTH Twilio AND Discord:
- Primary: Twilio SMS (reliable)
- Backup: Discord (free, instant)

### **Battery Backup:**
If on laptop, keep plugged in during game times

### **Internet Backup:**
Have mobile hotspot ready if WiFi fails

### **Quick Test:**
Before first game, run:
```bash
python check_tonight_games.py
```
Then start monitor manually to test SMS

---

## 🎮 **GAME DAY WORKFLOW**

### **Automatic (Task Scheduler):**
1. ⏰ 5:00 PM - Script auto-starts
2. 📱 Wait for SMS alerts
3. 💰 Execute trades
4. 📊 Track results

### **Manual:**
1. Check games: `python check_tonight_games.py`
2. Start monitor: `python auto_sms_monitor.py`
3. Wait for SMS alerts
4. Execute trades
5. Track results

---

**Need help?** See `SETUP_LIVE_SMS_ALERTS.md` for full documentation.

**Ready to start?** Run `setup_auto_monitor.bat` now!

