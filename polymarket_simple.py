"""
Simplified Polymarket API Investigation
Uses GraphQL to search for NBA game markets
"""

import requests
import json
import time
from datetime import datetime, timedelta

# Polymarket API endpoints (trying multiple)
API_BASE = "https://clob.polymarket.com"
API_V2 = "https://clob.polymarket.com/v2"
POLYMARKET_WEB = "https://polymarket.com"

def test_connection():
    """Test if we can reach Polymarket APIs."""
    print("\n" + "="*70)
    print("Testing Polymarket API Connection...")
    print("="*70)
    
    # Try REST API endpoints
    endpoints = [
        (f"{API_BASE}/markets", "Base markets endpoint"),
        (f"{API_V2}/markets", "V2 markets endpoint"),
        (f"{API_BASE}/events", "Events endpoint"),
    ]
    
    for url, desc in endpoints:
        try:
            response = requests.get(url, params={"limit": 10}, timeout=10)
            if response.status_code == 200:
                print(f"[OK] {desc} is reachable")
                return url
            elif response.status_code == 404:
                print(f"[!] {desc} returned 404 (not found)")
            else:
                print(f"[X] {desc} returned {response.status_code}")
        except Exception as e:
            print(f"[X] {desc} failed: {e}")
    
    print("\n[!] Could not find working API endpoint")
    print("  Polymarket may require authentication or use a different API structure")
    return None

def search_markets(base_url, search_terms):
    """Search for markets matching given terms."""
    print("\n" + "="*70)
    print(f"Searching for markets with terms: {', '.join(search_terms)}")
    print("="*70)
    
    if not base_url:
        print("[X] No working API endpoint found")
        return []
    
    try:
        # Try to get markets
        response = requests.get(
            f"{base_url}/markets",
            params={"limit": 500, "active": "true"},
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"[X] Failed to fetch markets: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            # Try alternative endpoint
            response = requests.get(base_url, timeout=10)
            if response.status_code != 200:
                return []
        
        data = response.json()
        
        # Handle different response formats
        markets = data.get("markets", []) or data.get("data", []) or (data if isinstance(data, list) else [])
        
        if not markets:
            print("[X] No markets in response")
            print(f"  Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            print(f"  Response structure: {str(data)[:300]}")
            # Try to extract from nested structure
            if isinstance(data, dict) and "data" in data:
                if isinstance(data["data"], list):
                    markets = data["data"]
                elif isinstance(data["data"], dict) and "markets" in data["data"]:
                    markets = data["data"]["markets"]
            
            if not markets:
                return []
        
        print(f"[OK] Found {len(markets)} total markets")
        
        # Filter for our game
        matching = []
        for market in markets:
            # Try different field names for question/title
            question = (
                market.get("question", "") or 
                market.get("title", "") or 
                market.get("name", "") or
                str(market)
            ).lower()
            
            # Check if any search term matches
            if any(term.lower() in question for term in search_terms):
                matching.append(market)
                title = market.get("question") or market.get("title") or market.get("name", "Unknown")
                active_status = "ACTIVE" if market.get("active") and not market.get("archived") else "ARCHIVED"
                print(f"\n[OK] FOUND ({active_status}): {title}")
                print(f"    Condition ID: {market.get('condition_id', 'N/A')}")
                print(f"    Question ID: {market.get('question_id', 'N/A')}")
                print(f"    Active: {market.get('active')}, Archived: {market.get('archived')}, Closed: {market.get('closed')}")
        
        if not matching:
            print(f"\n[X] No markets found matching: {search_terms}")
            print("\n  Sample markets found:")
            for i, market in enumerate(markets[:10]):
                title = market.get("question") or market.get("title") or market.get("name", str(market)[:50])
                print(f"    {i+1}. {title}")
        
        return matching
        
    except Exception as e:
        print(f"[X] Error: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_market_prices(base_url, market_id):
    """Get current prices for a market."""
    print("\n" + "="*70)
    print("Getting Market Prices")
    print("="*70)
    
    try:
        # Try different endpoints for getting market prices
        endpoints = [
            f"{base_url}/markets/{market_id}",
            f"{base_url}/markets?condition_id={market_id}",
            f"{base_url}/orderbook?token_id={market_id}",
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Handle different response formats
                    market = data if isinstance(data, dict) else data.get("market", {}) or data.get("data", {})
                    
                    # Extract prices (format may vary)
                    yes_price = (
                        market.get("yes_price") or 
                        market.get("yesPrice") or 
                        market.get("prices", {}).get("yes") or
                        market.get("outcomes", [{}])[0].get("price") if market.get("outcomes") else None
                    )
                    no_price = (
                        market.get("no_price") or 
                        market.get("noPrice") or 
                        market.get("prices", {}).get("no") or
                        market.get("outcomes", [{}])[1].get("price") if market.get("outcomes") and len(market.get("outcomes", [])) > 1 else None
                    )
                    
                    if yes_price or no_price:
                        print(f"[OK] Current Prices (from {endpoint}):")
                        print(f"  YES: {yes_price}" if yes_price else "  YES: N/A")
                        print(f"  NO:  {no_price}" if no_price else "  NO:  N/A")
                        
                        return {
                            "yes_price": yes_price,
                            "no_price": no_price,
                            "timestamp": datetime.now().isoformat()
                        }
            except:
                continue
        
        print(f"[X] Could not get prices from any endpoint")
        print(f"    Tried: {endpoints}")
        return None
            
    except Exception as e:
        print(f"[X] Error: {e}")
        return None

def monitor_market(base_url, market_id, duration_minutes=5):
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
                    f"{base_url}/markets/{market_id}",
                    timeout=5
                )
                
                if response.status_code == 200:
                    market = response.json()
                    yes_price = market.get("yes_price") or market.get("yesPrice") or market.get("prices", {}).get("yes")
                    
                    snapshot = {
                        "timestamp": datetime.now().isoformat(),
                        "yes_price": yes_price
                    }
                    snapshots.append(snapshot)
                    
                    # Print every 30 seconds
                    if sample_count % 15 == 0:
                        print(f"[{snapshot['timestamp'][:19]}] YES Price: {yes_price}")
                
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
        if snapshots[i]["yes_price"] != snapshots[i-1]["yes_price"]:
            changes += 1
    
    change_rate = (changes / len(snapshots)) * 100
    
    print(f"Samples Collected: {len(snapshots)}")
    print(f"Price Changes:     {changes}")
    print(f"Change Rate:       {change_rate:.1f}%")
    
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
    print("POLYMARKET API INVESTIGATION - SIMPLIFIED VERSION")
    print("Target: Lakers vs Hawks - November 8, 2025")
    print("="*70)
    
    # Step 1: Connect
    base_url = test_connection()
    if not base_url:
        print("\n" + "="*70)
        print("CONCLUSION")
        print("="*70)
        print("[X] Cannot connect to Polymarket API")
        print("\nPossible reasons:")
        print("  1. Polymarket API requires authentication")
        print("  2. API structure has changed")
        print("  3. Different endpoint URL needed")
        print("\nRecommendation:")
        print("  - Check Polymarket.com directly for NBA markets")
        print("  - Consider using their official SDK if available")
        print("  - Or proceed with Phase 2 (historical data approach)")
        return
    
    # Step 2: Find game
    search_terms = [
        "Los Angeles L", "Atlanta", "Lakers", "Hawks", 
        "LAL", "ATL", "Los Angeles", "NBA"
    ]
    
    markets = search_markets(base_url, search_terms)
    
    if not markets:
        print("\n[X] Cannot find the game. Check Polymarket.com to verify it exists.")
        return
    
    # Filter for active markets (not archived/closed)
    active_markets = [m for m in markets if m.get("active") and not m.get("archived") and not m.get("closed")]
    
    if not active_markets:
        print("\n[!] All matching markets are archived/closed")
        print("    Showing archived markets for reference:")
        for m in markets[:5]:
            print(f"      - {m.get('question') or m.get('title', 'Unknown')} (Active: {m.get('active')}, Archived: {m.get('archived')})")
        print("\n[X] No active markets found for Lakers vs Hawks")
        print("    The game market may not exist yet or has already closed")
        return
    
    # Use first active match
    market = active_markets[0]
    # Use condition_id or question_id as market identifier
    market_id = market.get("condition_id") or market.get("question_id") or market.get("id")
    
    print(f"\n[OK] Using market: {market.get('question') or market.get('title', 'Unknown')}")
    print(f"    Market ID: {market_id}")
    print(f"    Active: {market.get('active')}")
    print(f"    Archived: {market.get('archived')}")
    print(f"    Closed: {market.get('closed')}")
    
    # Step 3: Get prices
    prices = get_market_prices(base_url, market_id)
    if not prices:
        print("\n[X] Cannot get prices")
        return
    
    # Step 4: THE LITMUS TEST
    print("\n" + "="*70)
    print("Ready to run the LITMUS TEST")
    print("This will monitor for 5 minutes. Press Ctrl+C to stop early.")
    print("="*70)
    
    response = input("\nProceed? (y/n): ")
    if response.lower() == 'y':
        snapshots = monitor_market(base_url, market_id, duration_minutes=5)
        
        # Save results
        results = {
            "market": market,
            "initial_prices": prices,
            "snapshots": snapshots
        }
        
        filename = f"polymarket_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[OK] Results saved to: {filename}")
    
    print("\n" + "="*70)
    print("INVESTIGATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
