"""
Simplified Kalshi API Investigation
Uses only the requests library - no SDK required
"""

import requests
import json
import time
from datetime import datetime, timedelta

# API Configuration
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"

def test_connection():
    """Test if we can reach the Kalshi API."""
    print("\n" + "="*70)
    print("Testing API Connection...")
    print("="*70)
    
    for name, url in [("Production", BASE_URL), ("Demo", DEMO_URL)]:
        try:
            response = requests.get(f"{url}/exchange/status", timeout=10)
            if response.status_code == 200:
                print(f"[OK] {name} API is reachable")
                data = response.json()
                print(f"  Status: {data.get('exchange_status', 'Unknown')}")
                return url
        except Exception as e:
            print(f"[X] {name} API failed: {e}")
    
    return None

def search_by_category(base_url, category_name, subcategory=None):
    """Search for events in a specific category."""
    print("\n" + "="*70)
    print(f"Searching in category: {category_name}" + (f" > {subcategory}" if subcategory else ""))
    print("="*70)
    
    try:
        response = requests.get(
            f"{base_url}/events",
            params={"status": "open", "limit": 200},  # API limit
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"[X] Failed to fetch events: {response.status_code}")
            return None
        
        events = response.json().get("events", [])
        print(f"[OK] Found {len(events)} total open events")
        
        # Filter by category
        category_lower = category_name.lower()
        subcategory_lower = subcategory.lower() if subcategory else None
        filtered_events = []
        
        for event in events:
            event_category = event.get("category", "").lower()
            event_subcategory = event.get("subcategory", "").lower()
            # Also check if subcategory info is in category field
            full_category = f"{event_category} {event_subcategory}".lower()
            
            # Check if category matches
            if category_lower in event_category or category_lower in full_category:
                # If subcategory specified, check that too
                if subcategory:
                    # Check various ways subcategory might appear
                    if (subcategory_lower in event_subcategory or 
                        subcategory_lower in event_category or 
                        subcategory_lower in full_category or
                        "basketball" in event_category and "m" in event_category):
                        filtered_events.append(event)
                else:
                    filtered_events.append(event)
        
        print(f"[OK] Found {len(filtered_events)} events in {category_name}")
        if subcategory:
            print(f"      (filtered by subcategory: {subcategory})")
        
        # Show all events in this category
        if filtered_events:
            print("\nEvents found:")
            for i, event in enumerate(filtered_events[:20]):  # Show first 20
                print(f"  {i+1}. {event.get('title')}")
                print(f"      Category: {event.get('category')}")
                if event.get('subcategory'):
                    print(f"      Subcategory: {event.get('subcategory')}")
                print(f"      Ticker: {event.get('event_ticker')}")
            if len(filtered_events) > 20:
                print(f"  ... and {len(filtered_events) - 20} more")
        
        return filtered_events
        
    except Exception as e:
        print(f"[X] Error: {e}")
        return None

def search_for_game(base_url, team1="Lakers", team2="Hawks"):
    """Search for a specific game market."""
    search_desc = f"{team1} vs {team2}" if team2 else f"{team1}"
    print("\n" + "="*70)
    print(f"Searching for {search_desc}...")
    print("="*70)
    
    try:
        response = requests.get(
            f"{base_url}/events",
            params={"status": "open", "limit": 200},  # API limit
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"[X] Failed to fetch events: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return None
        
        events = response.json().get("events", [])
        print(f"[OK] Found {len(events)} total open events")
        
        # Search for our game
        team1_lower = team1.lower() if team1 else ""
        team2_lower = team2.lower() if team2 else ""
        
        for event in events:
            title = event.get("title", "").lower()
            category = event.get("category", "").lower()
            subcategory = event.get("subcategory", "").lower()
            
            # If both terms provided, both must match
            if team2:
                # Check if both teams are in title (handles "vs", "at", "versus" formats)
                team1_match = team1_lower in title
                team2_match = team2_lower in title
                # Also check if "at" or "vs" is in the search terms
                has_at = "at" in team1_lower or "at" in team2_lower
                has_vs = "vs" in team1_lower or "vs" in team2_lower
                
                if team1_match and team2_match:
                    # Also prefer titles that have "at" or "vs" if we're searching for that format
                    title_has_at = " at " in title or " @ " in title
                    title_has_vs = " vs " in title or " versus " in title
                    
                    # Match if both teams present, and if format specified, prefer matching format
                    if (not has_at and not has_vs) or (has_at and title_has_at) or (has_vs and title_has_vs) or (not title_has_at and not title_has_vs):
                        print(f"\n[OK] FOUND: {event.get('title')}")
                        print(f"  Ticker: {event.get('event_ticker')}")
                        print(f"  Category: {event.get('category')}")
                        if subcategory:
                            print(f"  Subcategory: {event.get('subcategory')}")
                        return event
            # If only one term, just check that one
            elif team1:
                if team1_lower in title or team1_lower in category:
                    print(f"\n[OK] FOUND: {event.get('title')}")
                    print(f"  Ticker: {event.get('event_ticker')}")
                    print(f"  Category: {event.get('category')}")
                    if subcategory:
                        print(f"  Subcategory: {event.get('subcategory')}")
                    return event
        
        print(f"\n[X] No market found for {search_desc}")
        print("\n  Sample events found:")
        for i, event in enumerate(events[:5]):
            print(f"    {i+1}. {event.get('title')}")
        
        return None
        
    except Exception as e:
        print(f"[X] Error: {e}")
        return None

def explore_api_structure(base_url):
    """Step-by-step exploration: Find all categories, then Sports, then Pro Basketball (M)."""
    print("\n" + "="*70)
    print("STEP-BY-STEP API EXPLORATION")
    print("="*70)
    
    try:
        # Step 1: Get all markets to see structure
        print("\nStep 1: Fetching all markets to understand structure...")
        response = requests.get(
            f"{base_url}/markets",
            params={"status": "open", "limit": 200},
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"[X] Failed to fetch markets: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return None
        
        markets = response.json().get("markets", [])
        print(f"[OK] Retrieved {len(markets)} markets")
        
        # Show structure of first market to understand fields
        if markets:
            print("\n[DEBUG] Structure of first market:")
            first_market = markets[0]
            print(f"  Keys: {list(first_market.keys())}")
            print(f"  Sample market: {json.dumps(first_market, indent=2)[:500]}...")
        
        # Step 2: Find all unique categories
        print("\nStep 2: Analyzing categories...")
        categories = {}
        for m in markets:
            # Try different possible field names
            cat = m.get("category") or m.get("Category") or m.get("cat") or ""
            subcat = m.get("subcategory") or m.get("Subcategory") or m.get("subcat") or ""
            # Also check if category info is in other fields
            series = str(m.get("series_ticker", "")).lower()
            title = str(m.get("title", "")).lower()
            
            if cat:
                if cat not in categories:
                    categories[cat] = set()
                if subcat:
                    categories[cat].add(subcat)
        
        print(f"[OK] Found {len(categories)} unique categories:")
        for cat in sorted(categories.keys()):
            subcats = categories[cat]
            print(f"  - {cat}")
            if subcats:
                for subcat in sorted(subcats):
                    print(f"    └─ {subcat}")
        
        # If no categories found, show all unique values from all fields
        if not categories and markets:
            print("\n[DEBUG] No categories found. Showing all unique values from key fields:")
            all_titles = set()
            all_series = set()
            for m in markets[:50]:  # Check first 50
                if m.get("title"):
                    all_titles.add(m.get("title")[:50])
                if m.get("series_ticker"):
                    all_series.add(m.get("series_ticker"))
            print(f"  Sample titles: {list(all_titles)[:10]}")
            print(f"  Sample series tickers: {list(all_series)[:10]}")
        
        # Step 3: Find Sports category
        print("\nStep 3: Looking for 'Sports' category...")
        sports_markets = [m for m in markets if "sport" in str(m.get("category", "")).lower()]
        print(f"[OK] Found {len(sports_markets)} markets in Sports category")
        
        if sports_markets:
            # Step 4: Find Pro Basketball (M) subcategory
            print("\nStep 4: Looking for 'Pro Basketball (M)' subcategory...")
            basketball_markets = []
            for m in sports_markets:
                subcat = str(m.get("subcategory", "")).lower()
                cat = str(m.get("category", "")).lower()
                title = str(m.get("title", "")).lower()
                
                # Check various ways Pro Basketball (M) might appear
                if ("basketball" in subcat or "basketball" in cat or 
                    ("basketball" in title and "pro" in title)):
                    basketball_markets.append(m)
            
            print(f"[OK] Found {len(basketball_markets)} basketball-related markets")
            
            if basketball_markets:
                print("\nBasketball markets found:")
                for i, m in enumerate(basketball_markets[:30]):
                    print(f"  {i+1}. {m.get('title')}")
                    print(f"      Category: {m.get('category')}")
                    print(f"      Subcategory: {m.get('subcategory', 'N/A')}")
                    print(f"      Ticker: {m.get('ticker')}")
                    print()
                if len(basketball_markets) > 30:
                    print(f"  ... and {len(basketball_markets) - 30} more")
            
            return basketball_markets
        else:
            print("[X] No Sports markets found")
            return None
            
    except Exception as e:
        print(f"[X] Error exploring API: {e}")
        return None

def search_markets_directly(base_url, category=None, subcategory=None):
    """Search markets directly (may have better category filtering)."""
    print("\n" + "="*70)
    print("Searching markets directly...")
    print("="*70)
    
    try:
        params = {"status": "open", "limit": 200}
        if category:
            params["category"] = category
        
        response = requests.get(
            f"{base_url}/markets",
            params=params,
            timeout=10
        )
        
        if response.status_code == 200:
            markets = response.json().get("markets", [])
            print(f"[OK] Found {len(markets)} markets")
            
            # First, show unique categories/subcategories to understand structure
            categories = set()
            subcategories = set()
            for m in markets:
                cat = m.get("category", "")
                subcat = m.get("subcategory", "")
                if cat:
                    categories.add(cat)
                if subcat:
                    subcategories.add(subcat)
            
            if categories:
                print(f"\nUnique categories found: {sorted(categories)}")
            if subcategories:
                print(f"Unique subcategories found: {sorted(subcategories)}")
            
            # Filter by subcategory if provided
            if subcategory:
                filtered = []
                for m in markets:
                    m_subcat = str(m.get("subcategory", "")).lower()
                    m_category = str(m.get("category", "")).lower()
                    # Try various matching strategies
                    subcat_lower = subcategory.lower()
                    if (subcat_lower in m_subcat or subcat_lower in m_category or
                        "basketball" in m_subcat or "basketball" in m_category):
                        filtered.append(m)
                markets = filtered
                print(f"[OK] Filtered to {len(markets)} markets matching '{subcategory}'")
            
            # Show markets
            if markets:
                print("\nMarkets found:")
                for i, m in enumerate(markets[:20]):
                    print(f"  {i+1}. {m.get('title')}")
                    print(f"      Category: {m.get('category')}")
                    print(f"      Subcategory: {m.get('subcategory', 'N/A')}")
                    print(f"      Ticker: {m.get('ticker')}")
                    # Show all fields for first market to debug
                    if i == 0:
                        print(f"      All fields: {list(m.keys())}")
                    print()
                if len(markets) > 20:
                    print(f"  ... and {len(markets) - 20} more markets")
            else:
                print("[X] No markets found")
            
            return markets
        else:
            print(f"[X] Failed to fetch markets: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return []
            
    except Exception as e:
        print(f"[X] Error: {e}")
        return []

def get_market_ticker(base_url, event_ticker):
    """Get the market ticker for an event."""
    try:
        response = requests.get(
            f"{base_url}/events/{event_ticker}/markets",
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"[X] Failed to get market: {response.status_code}")
            return None
        
        markets = response.json().get("markets", [])
        if markets:
            return markets[0].get("ticker")
        
        return None
        
    except Exception as e:
        print(f"[X] Error: {e}")
        return None

def get_current_prices(base_url, market_ticker):
    """Get current orderbook and prices."""
    print("\n" + "="*70)
    print("Current Market Prices")
    print("="*70)
    
    try:
        # Get orderbook
        response = requests.get(
            f"{base_url}/markets/{market_ticker}/orderbook",
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"[X] Failed: {response.status_code}")
            return None
        
        orderbook = response.json()
        
        # Get market info
        response = requests.get(
            f"{base_url}/markets/{market_ticker}",
            timeout=10
        )
        
        market_data = response.json().get("market", {}) if response.status_code == 200 else {}
        
        # Parse orderbook
        yes_bids = orderbook.get("yes", [])
        no_asks = orderbook.get("no", [])
        
        yes_bid = yes_bids[0][0] if yes_bids else None
        yes_ask = (100 - no_asks[0][0]) if no_asks else None
        last_price = market_data.get("last_price")
        
        print(f"YES Bid:      {yes_bid}cents" if yes_bid else "YES Bid:      N/A")
        print(f"YES Ask:      {yes_ask}cents" if yes_ask else "YES Ask:      N/A")
        print(f"Last Price:   {last_price}cents" if last_price else "Last Price:   N/A")
        print(f"Spread:       {yes_ask - yes_bid}cents" if (yes_bid and yes_ask) else "Spread:       N/A")
        print(f"Volume:       {market_data.get('volume', 'N/A')}")
        print(f"Open Interest: {market_data.get('open_interest', 'N/A')}")
        
        return {
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "last_price": last_price,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"[X] Error: {e}")
        return None

def monitor_market(base_url, market_ticker, duration_minutes=5):
    """Monitor price changes - THE LITMUS TEST."""
    print("\n" + "="*70)
    print("THE LITMUS TEST - Monitoring Price Changes")
    print("="*70)
    print(f"Duration: {duration_minutes} minutes")
    print("Sampling every 2 seconds...")
    print("-"*70)
    
    snapshots = []
    end_time = datetime.now() + timedelta(minutes=duration_minutes)
    sample_count = 0
    
    try:
        while datetime.now() < end_time:
            sample_count += 1
            
            try:
                response = requests.get(
                    f"{base_url}/markets/{market_ticker}/orderbook",
                    timeout=5
                )
                
                if response.status_code == 200:
                    orderbook = response.json()
                    yes_bids = orderbook.get("yes", [])
                    no_asks = orderbook.get("no", [])
                    
                    yes_bid = yes_bids[0][0] if yes_bids else None
                    yes_ask = (100 - no_asks[0][0]) if no_asks else None
                    
                    snapshot = {
                        "timestamp": datetime.now().isoformat(),
                        "yes_bid": yes_bid,
                        "yes_ask": yes_ask,
                        "spread": (yes_ask - yes_bid) if (yes_bid and yes_ask) else None
                    }
                    snapshots.append(snapshot)
                    
                    # Print every 30 seconds
                    if sample_count % 15 == 0:
                        print(f"[{snapshot['timestamp'][:19]}] Bid: {yes_bid}cents, Ask: {yes_ask}cents")
                
            except Exception as e:
                print(f"Sample {sample_count} failed: {e}")
            
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\n[!] Stopped by user")
    
    # Analyze
    analyze_results(snapshots)
    
    return snapshots

def analyze_results(snapshots):
    """Analyze if market is fast or slow."""
    print("\n" + "="*70)
    print("ANALYSIS: Is This Market Fast or Slow?")
    print("="*70)
    
    if len(snapshots) < 2:
        print("[!] Not enough data")
        return
    
    # Count price changes
    changes = 0
    for i in range(1, len(snapshots)):
        if (snapshots[i]["yes_bid"] != snapshots[i-1]["yes_bid"] or
            snapshots[i]["yes_ask"] != snapshots[i-1]["yes_ask"]):
            changes += 1
    
    change_rate = (changes / len(snapshots)) * 100
    
    print(f"Samples Collected: {len(snapshots)}")
    print(f"Price Changes:     {changes}")
    print(f"Change Rate:       {change_rate:.1f}%")
    
    # Calculate spread
    spreads = [s["spread"] for s in snapshots if s["spread"]]
    if spreads:
        print(f"Average Spread:    {sum(spreads)/len(spreads):.2f}cents")
    
    # Verdict
    print("\n" + "-"*70)
    if change_rate > 10:
        print("[OK] VERDICT: FAST MARKET")
        print("  Prices update frequently. Could be viable for real-time trading.")
    elif change_rate > 1:
        print("[!] VERDICT: MODERATE MARKET")
        print("  Some price movement, but not highly active.")
    else:
        print("[X] VERDICT: SLOW/STATIC MARKET")
        print("  Minimal price movement. NOT suitable for momentum trading.")
        print("  This market is too human-driven and slow.")
    print("-"*70)

def main():
    """Main execution."""
    print("\n" + "="*70)
    print("KALSHI API INVESTIGATION - SIMPLIFIED VERSION")
    print("Target: Lakers vs Hawks - November 8, 2025")
    print("="*70)
    
    # Step 1: Connect
    base_url = test_connection()
    if not base_url:
        print("\n[X] Cannot connect to Kalshi API")
        return
    
    # Step 2: Find game - try multiple search strategies
    print("\nTrying different search terms...")
    
    # FIRST: Step-by-step exploration to understand API structure
    print("\n" + "="*70)
    print("STEP 2A: STEP-BY-STEP EXPLORATION")
    print("Finding: Sports > Pro Basketball (M)")
    print("="*70)
    basketball_markets = explore_api_structure(base_url)
    
    # Search for our game in the basketball markets found
    event = None
    if basketball_markets:
        print("\n" + "="*70)
        print("Searching for Lakers vs/at Hawks within basketball markets...")
        print("="*70)
        team1_keywords = ["los angeles l", "lakers", "lal", "los angeles"]
        team2_keywords = ["atlanta", "hawks", "atl"]
        
        for m in basketball_markets:
            title_lower = m.get("title", "").lower()
            # Check if both teams are mentioned (with "vs", "at", or "versus")
            has_team1 = any(kw in title_lower for kw in team1_keywords)
            has_team2 = any(kw in title_lower for kw in team2_keywords)
            
            # Check for game format indicators
            has_at = " at " in title_lower or " @ " in title_lower
            has_vs = " vs " in title_lower or " versus " in title_lower
            
            if has_team1 and has_team2 and (has_at or has_vs):
                # Found it! Create a pseudo-event structure
                event = {
                    "title": m.get("title"),
                    "event_ticker": m.get("event_ticker", m.get("ticker")),
                    "category": m.get("category"),
                    "subcategory": m.get("subcategory"),
                    "market_ticker": m.get("ticker")
                }
                print(f"\n[OK] FOUND: {m.get('title')}")
                print(f"  Market Ticker: {m.get('ticker')}")
                print(f"  Category: {m.get('category')}")
                if m.get('subcategory'):
                    print(f"  Subcategory: {m.get('subcategory')}")
                break
    
    # SECOND: Try searching events with "at" instead of "vs"
    if not event:
        print("\n" + "="*70)
        print("STEP 2B: Searching for 'Los Angeles L at Atlanta'")
        print("="*70)
        # Try "at" format
        event = search_for_game(base_url, "Los Angeles L", "Atlanta")
        if not event:
            event = search_for_game(base_url, "Los Angeles L", "ATL")
    
    # THIRD: Try other search methods if still not found
    if not event:
        print("\n" + "="*70)
        print("STEP 2C: Trying other search methods...")
        print("="*70)
    
    # First, let's get all events and see their structure
    if not event:
        try:
            response = requests.get(
                f"{base_url}/events",
                params={"status": "open", "limit": 200},
                timeout=10
            )
            
            if response.status_code == 200:
                all_events = response.json().get("events", [])
                print(f"[OK] Retrieved {len(all_events)} events")
                
                # Look for Pro Basketball events - check all possible field names
                basketball_events = []
                for e in all_events:
                    category = str(e.get("category", "")).lower()
                    subcategory = str(e.get("subcategory", "")).lower()
                    series_ticker = str(e.get("series_ticker", "")).lower()
                    title = str(e.get("title", "")).lower()
                    
                    # Check if it's basketball related
                    if ("basketball" in category or "basketball" in subcategory or 
                        "basketball" in series_ticker or "basketball" in title):
                        basketball_events.append(e)
                
                print(f"[OK] Found {len(basketball_events)} basketball-related events")
                
                if basketball_events:
                    print("\nBasketball events found:")
                    for i, e in enumerate(basketball_events[:20]):
                        print(f"  {i+1}. {e.get('title')}")
                        print(f"      Category: {e.get('category')}")
                        print(f"      Subcategory: {e.get('subcategory', 'N/A')}")
                        print(f"      Series Ticker: {e.get('series_ticker', 'N/A')}")
                        print(f"      Event Ticker: {e.get('event_ticker')}")
                        print()
                    
                    # Now search for our specific game
                    if not event:
                        print("\n" + "="*70)
                        print("Searching for Lakers vs/at Hawks within basketball events...")
                        print("="*70)
                        team1_keywords = ["los angeles l", "lakers", "lal", "los angeles"]
                        team2_keywords = ["atlanta", "hawks", "atl"]
                        
                        for e in basketball_events:
                            title_lower = e.get("title", "").lower()
                            # Check if both teams are mentioned (with "vs", "at", or "versus")
                            has_team1 = any(kw in title_lower for kw in team1_keywords)
                            has_team2 = any(kw in title_lower for kw in team2_keywords)
                            
                            if has_team1 and has_team2:
                                event = e
                                print(f"\n[OK] FOUND: {e.get('title')}")
                                print(f"  Ticker: {e.get('event_ticker')}")
                                print(f"  Category: {e.get('category')}")
                                if e.get('subcategory'):
                                    print(f"  Subcategory: {e.get('subcategory')}")
                                if e.get('series_ticker'):
                                    print(f"  Series Ticker: {e.get('series_ticker')}")
                                break
            else:
                print(f"[X] Failed to fetch events: {response.status_code}")
        except Exception as e:
            print(f"[X] Error fetching events: {e}")
    
    # Also try searching markets directly (might have better category support)
    if not event:
        print("\n" + "="*70)
        print("STEP 2B: Searching markets directly for Pro Basketball (M)")
        print("="*70)
        markets = search_markets_directly(base_url, "Sports", "Pro Basketball (M)")
        
        if markets:
            # Search for our game in these markets
            team1_keywords = ["los angeles l", "lakers", "lal", "los angeles"]
            team2_keywords = ["atlanta", "hawks", "atl"]
            
            for m in markets:
                title_lower = m.get("title", "").lower()
                has_team1 = any(kw in title_lower for kw in team1_keywords)
                has_team2 = any(kw in title_lower for kw in team2_keywords)
                
                # Check for "at" or "vs" format
                has_at_format = ("at" in title_lower and has_team1 and has_team2)
                has_vs_format = ("vs" in title_lower or "versus" in title_lower) and has_team1 and has_team2
                
                if has_team1 and has_team2 and (has_at_format or has_vs_format or "at" in title_lower or "vs" in title_lower):
                    # Found it! Create a pseudo-event structure
                    event = {
                        "title": m.get("title"),
                        "event_ticker": m.get("event_ticker", m.get("ticker")),
                        "category": m.get("category"),
                        "subcategory": m.get("subcategory"),
                        "market_ticker": m.get("ticker")
                    }
                    print(f"\n[OK] FOUND MARKET: {m.get('title')}")
                    print(f"  Market Ticker: {m.get('ticker')}")
                    print(f"  Category: {m.get('category')}")
                    if m.get('subcategory'):
                        print(f"  Subcategory: {m.get('subcategory')}")
                    break
    
    # If not found in Pro Basketball, try other search methods
    if not event:
        print("\n[!] Not found in Pro Basketball (M), trying other search methods...")
        # Try "at" format first (Los Angeles L at Atlanta)
        print("\nTrying 'at' format: Los Angeles L at Atlanta")
        event = search_for_game(base_url, "Los Angeles L at", "Atlanta")
        if not event:
            event = search_for_game(base_url, "Los Angeles L", "at Atlanta")
        # Try the exact format the user mentioned
        if not event:
            event = search_for_game(base_url, "Los Angeles L", "Atlanta")
        if not event:
            event = search_for_game(base_url, "Los Angeles L", "ATL")
    # Try city names
    if not event:
        event = search_for_game(base_url, "Atlanta", "Los Angeles")
    if not event:
        event = search_for_game(base_url, "Atlanta", "LA")
    if not event:
        event = search_for_game(base_url, "ATL", "LA")
    # Try team names
    if not event:
        event = search_for_game(base_url, "Lakers", "Hawks")
    if not event:
        event = search_for_game(base_url, "LAL", "ATL")
    # Try just one city to see what NBA markets exist
    if not event:
        print("\n" + "="*70)
        print("Searching for any NBA/Sports related markets...")
        print("="*70)
        event = search_for_game(base_url, "NBA", "")
        if not event:
            event = search_for_game(base_url, "basketball", "")
        if not event:
            event = search_for_game(base_url, "sports", "")
    
    # If still nothing, show all sports events
    if not event:
        print("\n" + "="*70)
        print("Listing all Sports category events...")
        print("="*70)
        try:
            response = requests.get(
                f"{base_url}/events",
                params={"status": "open", "limit": 200},
                timeout=10
            )
            if response.status_code == 200:
                events = response.json().get("events", [])
                sports_events = [e for e in events if "sport" in e.get("category", "").lower()]
                print(f"[OK] Found {len(sports_events)} sports-related events:")
                for i, e in enumerate(sports_events[:10]):
                    print(f"  {i+1}. {e.get('title')} ({e.get('category')})")
                if len(sports_events) > 10:
                    print(f"  ... and {len(sports_events) - 10} more")
        except Exception as e:
            print(f"[X] Error listing sports events: {e}")
    
    if not event:
        print("\n[X] Cannot find the game. Check Kalshi.com to verify it exists.")
        return
    
    # Step 3: Get market ticker
    market_ticker = None
    
    # If we found it via markets endpoint, we already have the ticker
    if event and event.get("market_ticker"):
        market_ticker = event.get("market_ticker")
        print(f"\n[OK] Using market ticker from markets search: {market_ticker}")
    # Otherwise, get it from the event
    elif event:
        event_title = event.get("title", "").lower()
        is_game = any(word in event_title for word in ["vs", "versus", "game", "win", "beat", "defeat"])
        
        if not is_game:
            print(f"\n[!] Found NBA event but it's not a game market: {event.get('title')}")
            print("    This appears to be a question/prediction market, not a game winner market.")
            print("    Continuing search for actual game markets...")
            event = None  # Reset to continue searching
        else:
            market_ticker = get_market_ticker(base_url, event["event_ticker"])
            if not market_ticker:
                print(f"\n[!] Could not get market ticker for: {event.get('title')}")
                print("    This might not be a tradeable market yet, or the API structure is different.")
                print("    Showing all sports events instead...")
                event = None  # Reset to show all sports
            else:
                print(f"\n[OK] Market Ticker: {market_ticker}")
    
    if not event:
        # Show all relevant events
        print("\n" + "="*70)
        print("Searching for ANY events containing Los Angeles, Atlanta, Lakers, or Hawks...")
        print("="*70)
        try:
            response = requests.get(
                f"{base_url}/events",
                params={"status": "open", "limit": 200},
                timeout=10
            )
            if response.status_code == 200:
                events = response.json().get("events", [])
                
                # Search for any event with our keywords
                keywords = ["los angeles", "atlanta", "lakers", "hawks", "lal", "atl"]
                relevant_events = []
                for e in events:
                    title_lower = e.get("title", "").lower()
                    if any(keyword in title_lower for keyword in keywords):
                        relevant_events.append(e)
                
                if relevant_events:
                    print(f"[OK] Found {len(relevant_events)} events with relevant keywords:")
                    for i, e in enumerate(relevant_events):
                        print(f"  {i+1}. {e.get('title')} ({e.get('category')})")
                        print(f"      Ticker: {e.get('event_ticker')}")
                        print(f"      Status: {e.get('status', 'Unknown')}")
                else:
                    print("[X] No events found with keywords: Los Angeles, Atlanta, Lakers, Hawks")
                    print("\n[!] Trying to search ALL event statuses (not just 'open')...")
                    # Try searching without status filter
                    try:
                        response_all = requests.get(
                            f"{base_url}/events",
                            params={"limit": 200},  # No status filter
                            timeout=10
                        )
                        if response_all.status_code == 200:
                            all_events = response_all.json().get("events", [])
                            all_relevant = []
                            for e in all_events:
                                title_lower = e.get("title", "").lower()
                                if any(keyword in title_lower for keyword in keywords):
                                    all_relevant.append(e)
                            if all_relevant:
                                print(f"[OK] Found {len(all_relevant)} events (all statuses) with keywords:")
                                for i, e in enumerate(all_relevant[:10]):
                                    print(f"  {i+1}. {e.get('title')} - Status: {e.get('status', 'Unknown')}")
                            else:
                                print("[X] Still no events found in any status")
                    except Exception as e2:
                        print(f"[X] Error searching all statuses: {e2}")
                
                # Also show sports events
                print("\n" + "="*70)
                print("Listing all Sports category events...")
                print("="*70)
                sports_events = [e for e in events if "sport" in e.get("category", "").lower()]
                print(f"[OK] Found {len(sports_events)} sports-related events:")
                for i, e in enumerate(sports_events[:15]):
                    title = e.get('title', '')
                    print(f"  {i+1}. {title} ({e.get('category')})")
                if len(sports_events) > 15:
                    print(f"  ... and {len(sports_events) - 15} more")
                
                # Check if any look like game markets
                game_like = [e for e in sports_events if any(word in e.get("title", "").lower() 
                            for word in ["vs", "versus", "game", "win", "beat"])]
                if game_like:
                    print(f"\n[OK] Found {len(game_like)} events that might be game markets:")
                    for e in game_like[:5]:
                        print(f"  - {e.get('title')} ({e.get('category')})")
                
                if not sports_events:
                    print("\n[!] No sports events found in the first 200 events.")
                    print("    Kalshi may not have NBA game markets, or they're listed differently.")
                    return
        except Exception as e:
            print(f"[X] Error listing sports events: {e}")
            return
    
    # Step 4: Get prices (only if we have a valid market)
    if not event or not market_ticker:
        print("\n" + "="*70)
        print("CONCLUSION")
        print("="*70)
        print("[X] No game market found for Lakers vs Hawks")
        print("\nPossible reasons:")
        print("  1. Kalshi doesn't offer NBA game markets")
        print("  2. The market hasn't been created yet (check closer to game time)")
        print("  3. Markets are named differently than expected")
        print("\nRecommendation: Proceed with Phase 2 (historical data approach)")
        return
    
    prices = get_current_prices(base_url, market_ticker)
    if not prices:
        print("\n[X] Cannot get prices")
        return
    
    # Step 5: THE LITMUS TEST
    print("\n" + "="*70)
    print("Ready to run the LITMUS TEST")
    print("This will monitor for 5 minutes. Press Ctrl+C to stop early.")
    print("="*70)
    
    response = input("\nProceed? (y/n): ")
    if response.lower() == 'y':
        snapshots = monitor_market(base_url, market_ticker, duration_minutes=5)
        
        # Save results
        results = {
            "event": event,
            "market_ticker": market_ticker,
            "initial_prices": prices,
            "snapshots": snapshots
        }
        
        filename = f"kalshi_simple_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[OK] Results saved to: {filename}")
    
    print("\n" + "="*70)
    print("INVESTIGATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()

