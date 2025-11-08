"""
Polymarket API Investigation Script
Purpose: Test if Polymarket API is viable for real-time NBA game data
Target: Lakers vs Hawks game on November 8, 2025
"""

import requests
import json
import time
from datetime import datetime, timedelta

# Polymarket API Configuration
# Polymarket uses multiple endpoints
GRAPHQL_URL = "https://data-api.polymarket.com/graphql"  # Try alternative
CLOB_URL = "https://clob.polymarket.com"
MARKETS_URL = "https://clob.polymarket.com/markets"  # Use CLOB for markets

def test_connection():
    """Test if we can reach the Polymarket API."""
    print("\n" + "="*70)
    print("Testing Polymarket API Connection...")
    print("="*70)
    
    # Try GraphQL query for markets
    graphql_query = {
        "query": """
        query {
            markets(limit: 10) {
                id
                question
                slug
                active
            }
        }
        """
    }
    
    try:
        response = requests.post(GRAPHQL_URL, json=graphql_query, timeout=10)
        if response.status_code == 200:
            print(f"[OK] GraphQL API is reachable")
            data = response.json()
            if "data" in data and "markets" in data["data"]:
                markets = data["data"]["markets"]
                print(f"  Found {len(markets)} sample markets")
                return GRAPHQL_URL
            else:
                print(f"  Response structure: {list(data.keys())}")
        else:
            print(f"[X] GraphQL API returned status {response.status_code}")
    except Exception as e:
        print(f"[X] GraphQL API failed: {e}")
    
    # Try REST endpoints
    endpoints = [
        ("CLOB Markets API", MARKETS_URL),
        ("CLOB API", CLOB_URL),
    ]
    
    for name, url in endpoints:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"[OK] {name} is reachable")
                return url
            else:
                print(f"[X] {name} returned status {response.status_code}")
        except Exception as e:
            print(f"[X] {name} failed: {e}")
    
    return GRAPHQL_URL  # Default to GraphQL

def search_markets(base_url, search_terms):
    """Search for markets matching search terms."""
    print("\n" + "="*70)
    print(f"Searching for markets: {', '.join(search_terms)}")
    print("="*70)
    
    matching_markets = []
    
    # Try GraphQL query
    try:
        graphql_query = {
            "query": """
            query {
                markets(limit: 200, active: true) {
                    id
                    question
                    slug
                    active
                    description
                    conditionId
                    outcomes
                    volume
                    liquidity
                }
            }
            """
        }
        
        response = requests.post(GRAPHQL_URL, json=graphql_query, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "data" in data and "markets" in data["data"]:
                markets = data["data"]["markets"]
                print(f"[OK] Found {len(markets)} markets via GraphQL")
                
                # Search for our game
                for market in markets:
                    title = str(market.get("question", "")).lower()
                    description = str(market.get("description", "")).lower()
                    
                    if any(term.lower() in title or term.lower() in description for term in search_terms):
                        matching_markets.append(market)
    except Exception as e:
        print(f"[X] GraphQL search failed: {e}")
    
    # Try CLOB REST endpoint
    try:
        response = requests.get(
            MARKETS_URL,
            timeout=10
        )
        
        if response.status_code == 200:
            markets = response.json()
            if isinstance(markets, list):
                print(f"[OK] Found {len(markets)} markets via CLOB REST")
            else:
                markets = markets.get("data", markets.get("markets", [])) if isinstance(markets, dict) else []
                print(f"[OK] Found {len(markets)} markets via CLOB REST")
            
            # Search CLOB results - prioritize active, non-closed markets
            for market in markets:
                title = str(market.get("question", market.get("title", market.get("name", "")))).lower()
                description = str(market.get("description", "")).lower()
                
                # Check if matches search terms
                matches_search = any(term.lower() in title or term.lower() in description for term in search_terms)
                
                if matches_search:
                    # Check if market is actually active (not closed)
                    is_closed = market.get("closed", False)
                    is_active = market.get("active", True) and not is_closed
                    accepting_orders = market.get("accepting_orders", False)
                    
                    # Prioritize active markets
                    if is_active and accepting_orders:
                        matching_markets.insert(0, market)  # Put active markets first
                    else:
                        matching_markets.append(market)
            
            # Also show sample of all markets to understand structure
            if markets and len(markets) > 0:
                print(f"\n[DEBUG] Sample market structure:")
                sample = markets[0]
                print(f"  Keys: {list(sample.keys())[:10]}")
                print(f"  Sample: {json.dumps(sample, indent=2)[:300]}")
        else:
            print(f"[X] CLOB REST returned status {response.status_code}")
    except Exception as e:
        print(f"[X] CLOB REST search failed: {e}")
    
    if matching_markets:
        # Separate active vs closed markets
        truly_active = []
        closed_markets = []
        
        for m in matching_markets:
            is_closed = m.get("closed", False)
            accepting_orders = m.get("accepting_orders", False)
            is_active = m.get("active", True) and not is_closed
            
            if is_active and accepting_orders:
                truly_active.append(m)
            else:
                closed_markets.append(m)
        
        # Show active markets first, then closed ones
        display_markets = truly_active + closed_markets[:5]  # Show some closed for reference
        
        print(f"\n[OK] Found {len(matching_markets)} total matching markets")
        print(f"      {len(truly_active)} are ACTIVE (accepting orders, not closed)")
        print(f"      {len(closed_markets)} are CLOSED (historical/resolved)")
        
        if truly_active:
            print(f"\n[OK] ACTIVE MARKETS (these can be monitored):")
        else:
            print(f"\n[!] NO ACTIVE MARKETS FOUND - all are historical/resolved")
        
        print(f"\nShowing markets:")
        for i, m in enumerate(display_markets[:10]):
            title = m.get("question") or m.get("title") or m.get("name", "Unknown")
            is_closed = m.get("closed", False)
            accepting = m.get("accepting_orders", False)
            status = "ACTIVE" if (not is_closed and accepting) else "CLOSED"
            
            print(f"  {i+1}. {title}")
            print(f"      Status: {status}")
            print(f"      Condition ID: {m.get('conditionId') or m.get('condition_id', 'N/A')}")
            print(f"      Closed: {is_closed}, Accepting Orders: {accepting}")
            if m.get("end_date_iso"):
                print(f"      End Date: {m.get('end_date_iso')}")
            print()
        
        return truly_active if truly_active else display_markets
    else:
        print(f"\n[X] No markets found matching: {search_terms}")
        return []

def search_events(base_url, search_terms):
    """Search for events matching search terms (Polymarket may not have separate events)."""
    # Polymarket markets are the events, so this is a placeholder
    return []

def get_market_prices(market_id):
    """Get current prices for a market."""
    print("\n" + "="*70)
    print("Current Market Prices")
    print("="*70)
    
    try:
        # Try different possible endpoints
        endpoints = [
            f"{MARKETS_URL}/{market_id}",
            f"{CLOB_URL}/markets/{market_id}",
            f"{CLOB_URL}/market/{market_id}",
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    print(f"[OK] Market data retrieved from {endpoint}")
                    
                    # Try to extract price information
                    prices = data.get("prices", {})
                    if not prices:
                        prices = data.get("outcomes", {})
                    if not prices:
                        prices = data.get("yes", {})
                    
                    if prices:
                        print(f"  Prices: {json.dumps(prices, indent=2)}")
                    else:
                        print(f"  Full response structure: {list(data.keys())}")
                        print(f"  Sample data: {json.dumps(data, indent=2)[:500]}")
                    
                    return data
            except:
                continue
        
        print("[X] Could not retrieve market prices from any endpoint")
        return None
        
    except Exception as e:
        print(f"[X] Error: {e}")
        return None

def explore_categories():
    """Explore available categories on Polymarket."""
    print("\n" + "="*70)
    print("Exploring Available Markets")
    print("="*70)
    
    try:
        # Get sample markets to see what categories exist
        graphql_query = {
            "query": """
            query {
                markets(limit: 100, active: true) {
                    question
                    slug
                    category
                    tags
                }
            }
            """
        }
        
        response = requests.post(GRAPHQL_URL, json=graphql_query, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "data" in data and "markets" in data["data"]:
                markets = data["data"]["markets"]
                print(f"[OK] Retrieved {len(markets)} sample markets")
                
                # Extract categories
                categories = set()
                tags = set()
                for m in markets:
                    if m.get("category"):
                        categories.add(m.get("category"))
                    if m.get("tags"):
                        if isinstance(m.get("tags"), list):
                            tags.update(m.get("tags"))
                        else:
                            tags.add(m.get("tags"))
                
                if categories:
                    print(f"\n  Categories found: {sorted(categories)}")
                if tags:
                    print(f"  Tags found: {sorted(list(tags))[:20]}")
                
                # Show sample markets
                print(f"\n  Sample markets:")
                for i, m in enumerate(markets[:10]):
                    print(f"    {i+1}. {m.get('question', 'Unknown')}")
                    if m.get("category"):
                        print(f"       Category: {m.get('category')}")
                
    except Exception as e:
        print(f"[X] Error exploring: {e}")

def main():
    """Main execution."""
    print("\n" + "="*70)
    print("POLYMARKET API INVESTIGATION")
    print("Target: Lakers vs Hawks - November 8, 2025")
    print("="*70)
    
    # Step 1: Test connection
    base_url = test_connection()
    if not base_url:
        print("\n[X] Cannot connect to Polymarket API")
        print("\nTrying alternative endpoints...")
        explore_categories()
        return
    
    # Step 2: Search for the game
    search_terms = [
        "Los Angeles L", "Lakers", "LAL", "Los Angeles",
        "Atlanta", "Hawks", "ATL",
        "NBA", "basketball"
    ]
    
    markets = search_markets(base_url, search_terms)
    # Polymarket doesn't have separate events endpoint
    events = []
    
    # Step 3: If found, get prices
    if markets:
        print("\n" + "="*70)
        print("Market Found! Getting price data...")
        print("="*70)
        market = markets[0]
        market_id = market.get("id") or market.get("market_id")
        condition_id = market.get("conditionId") or market.get("condition_id")
        
        if market_id or condition_id:
            prices = get_market_prices(market_id or condition_id)
            
            if prices:
                # Check if market is actually active
                is_closed = prices.get("closed", False)
                accepting_orders = prices.get("accepting_orders", False)
                
                if is_closed or not accepting_orders:
                    print(f"\n[!] WARNING: This market is CLOSED or not accepting orders")
                    print(f"    Closed: {is_closed}")
                    print(f"    Accepting Orders: {accepting_orders}")
                    print(f"    This is a historical/resolved market - prices won't change")
                    print(f"\n[X] Cannot monitor prices for inactive market")
                else:
                    print("\n" + "="*70)
                    print("Ready to monitor prices (THE LITMUS TEST)")
                    print("="*70)
                    print("Market is ACTIVE - prices should update in real-time")
                    print("\n[!] Price monitoring not yet implemented")
                    print("    This would sample prices every 2 seconds for 5 minutes")
                    print("    To implement: Query market prices repeatedly and track changes")
    
    elif events:
        print("\n[OK] Found events but need to get associated markets")
        print("     Events found may not have active markets yet")
    
    else:
        print("\n" + "="*70)
        print("CONCLUSION")
        print("="*70)
        print("[X] No game market found for Lakers vs Hawks")
        print("\nPossible reasons:")
        print("  1. Polymarket doesn't offer NBA game markets")
        print("  2. The market hasn't been created yet")
        print("  3. Markets are named differently")
        print("  4. API structure is different than expected")
        
        # Explore what's available
        print("\n" + "="*70)
        print("Exploring available markets...")
        print("="*70)
        explore_categories()
    
    print("\n" + "="*70)
    print("INVESTIGATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
