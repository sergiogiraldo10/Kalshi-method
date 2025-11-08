"""
Kalshi API Investigation Script
Purpose: Test if Kalshi API is viable for real-time NBA game data
Target: Lakers vs Hawks game on November 8, 2025
"""

import requests
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

class KalshiInvestigator:
    """
    A class to investigate Kalshi API for NBA game markets.
    Supports both authenticated and unauthenticated access.
    """
    
    def __init__(self, api_key: Optional[str] = None, private_key_path: Optional[str] = None):
        """
        Initialize the Kalshi investigator.
        
        Args:
            api_key: Optional API key for authenticated requests
            private_key_path: Optional path to private key file
        """
        self.base_url = "https://api.elections.kalshi.com/trade-api/v2"
        self.demo_url = "https://demo-api.kalshi.co/trade-api/v2"
        self.api_key = api_key
        self.private_key_path = private_key_path
        self.session = requests.Session()
        
        # Try both production and demo endpoints
        self.use_demo = False
        
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _get_base_url(self) -> str:
        """Get the appropriate base URL."""
        return self.demo_url if self.use_demo else self.base_url
    
    def test_connection(self) -> bool:
        """Test connection to Kalshi API."""
        print("\n" + "="*70)
        print("STEP 1: Testing API Connection")
        print("="*70)
        
        # Try production first
        try:
            url = f"{self.base_url}/exchange/status"
            response = self.session.get(url, headers=self._get_headers(), timeout=10)
            if response.status_code == 200:
                print(f"✓ Production API connection successful")
                print(f"  Status: {response.json().get('exchange_status', 'Unknown')}")
                return True
        except Exception as e:
            print(f"✗ Production API connection failed: {e}")
        
        # Try demo
        try:
            self.use_demo = True
            url = f"{self.demo_url}/exchange/status"
            response = self.session.get(url, headers=self._get_headers(), timeout=10)
            if response.status_code == 200:
                print(f"✓ Demo API connection successful")
                print(f"  Status: {response.json().get('exchange_status', 'Unknown')}")
                return True
        except Exception as e:
            print(f"✗ Demo API connection failed: {e}")
        
        print("\n⚠ Could not connect to Kalshi API")
        print("  This may require authentication. See setup instructions below.")
        return False
    
    def search_markets(self, search_terms: List[str]) -> List[Dict]:
        """
        Search for markets matching given terms.
        
        Args:
            search_terms: List of search terms (e.g., ["Lakers", "Hawks"])
            
        Returns:
            List of matching markets
        """
        print("\n" + "="*70)
        print("STEP 2: Searching for Markets")
        print("="*70)
        print(f"Search terms: {', '.join(search_terms)}")
        
        try:
            # Get all events
            url = f"{self._get_base_url()}/events"
            params = {
                "status": "open",
                "limit": 200
            }
            
            response = self.session.get(url, headers=self._get_headers(), params=params, timeout=10)
            
            if response.status_code != 200:
                print(f"✗ Failed to fetch events: {response.status_code}")
                print(f"  Response: {response.text}")
                return []
            
            data = response.json()
            events = data.get("events", [])
            print(f"✓ Found {len(events)} total open events")
            
            # Filter for our game
            matching_events = []
            for event in events:
                title = event.get("title", "").lower()
                category = event.get("category", "").lower()
                
                # Look for NBA games with Lakers and Hawks
                if any(term.lower() in title for term in search_terms):
                    matching_events.append(event)
                    print(f"\n  Match found: {event.get('title')}")
                    print(f"    Event Ticker: {event.get('event_ticker')}")
                    print(f"    Category: {event.get('category')}")
                    print(f"    Close Time: {event.get('close_time', 'N/A')}")
            
            if not matching_events:
                print(f"\n✗ No markets found matching: {search_terms}")
                print("  This could mean:")
                print("    1. The game hasn't been listed yet")
                print("    2. Kalshi doesn't have markets for this game")
                print("    3. The market naming is different than expected")
                
                # Show some sample events for reference
                print("\n  Sample of available events:")
                for i, event in enumerate(events[:5]):
                    print(f"    {i+1}. {event.get('title')} ({event.get('category')})")
            
            return matching_events
            
        except Exception as e:
            print(f"✗ Error searching markets: {e}")
            return []
    
    def get_market_details(self, event_ticker: str) -> Optional[Dict]:
        """
        Get detailed market information for an event.
        
        Args:
            event_ticker: The event ticker
            
        Returns:
            Market details or None
        """
        print("\n" + "="*70)
        print("STEP 3: Getting Market Details")
        print("="*70)
        
        try:
            url = f"{self._get_base_url()}/events/{event_ticker}/markets"
            response = self.session.get(url, headers=self._get_headers(), timeout=10)
            
            if response.status_code != 200:
                print(f"✗ Failed to fetch market details: {response.status_code}")
                return None
            
            data = response.json()
            markets = data.get("markets", [])
            
            if not markets:
                print("✗ No markets found for this event")
                return None
            
            # Typically the first market is the main one (winner)
            market = markets[0]
            print(f"✓ Market found: {market.get('title')}")
            print(f"  Ticker: {market.get('ticker')}")
            print(f"  Status: {market.get('status')}")
            
            return market
            
        except Exception as e:
            print(f"✗ Error getting market details: {e}")
            return None
    
    def get_current_prices(self, market_ticker: str) -> Optional[Dict]:
        """
        Get current price data for a market.
        
        Args:
            market_ticker: The market ticker
            
        Returns:
            Price data dictionary or None
        """
        print("\n" + "="*70)
        print("STEP 4: Current Price Data")
        print("="*70)
        
        try:
            # Get orderbook
            url = f"{self._get_base_url()}/markets/{market_ticker}/orderbook"
            response = self.session.get(url, headers=self._get_headers(), timeout=10)
            
            if response.status_code != 200:
                print(f"✗ Failed to fetch orderbook: {response.status_code}")
                return None
            
            orderbook = response.json()
            
            # Get market info for last price
            url = f"{self._get_base_url()}/markets/{market_ticker}"
            response = self.session.get(url, headers=self._get_headers(), timeout=10)
            
            if response.status_code != 200:
                print(f"✗ Failed to fetch market info: {response.status_code}")
                return None
            
            market_info = response.json().get("market", {})
            
            # Extract price data
            yes_bids = orderbook.get("yes", [])
            no_asks = orderbook.get("no", [])
            
            yes_bid = yes_bids[0][0] if yes_bids else None  # Best bid price
            yes_ask = (100 - no_asks[0][0]) if no_asks else None  # Implied from no ask
            last_price = market_info.get("last_price")
            
            price_data = {
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "last_price": last_price,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"✓ Current Prices (in cents):")
            print(f"  YES Bid:  {yes_bid}¢" if yes_bid else "  YES Bid:  N/A")
            print(f"  YES Ask:  {yes_ask}¢" if yes_ask else "  YES Ask:  N/A")
            print(f"  Last Price: {last_price}¢" if last_price else "  Last Price: N/A")
            print(f"  Volume:   {market_info.get('volume', 'N/A')}")
            print(f"  Open Interest: {market_info.get('open_interest', 'N/A')}")
            
            return price_data
            
        except Exception as e:
            print(f"✗ Error getting prices: {e}")
            return None
    
    def monitor_price_changes(self, market_ticker: str, duration_minutes: int = 5) -> List[Dict]:
        """
        Monitor price changes over a period of time.
        This is the "LITMUS TEST" to see if the market is fast or slow.
        
        Args:
            market_ticker: The market ticker
            duration_minutes: How many minutes to monitor (default 5)
            
        Returns:
            List of price snapshots
        """
        print("\n" + "="*70)
        print("STEP 5: THE LITMUS TEST - Price Movement Analysis")
        print("="*70)
        print(f"Monitoring market for {duration_minutes} minutes...")
        print("Sampling every 2 seconds to detect price changes")
        print("-" * 70)
        
        snapshots = []
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        sample_count = 0
        
        try:
            while datetime.now() < end_time:
                sample_count += 1
                
                # Get current prices
                url = f"{self._get_base_url()}/markets/{market_ticker}/orderbook"
                response = self.session.get(url, headers=self._get_headers(), timeout=10)
                
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
                        "spread": (yes_ask - yes_bid) if (yes_ask and yes_bid) else None
                    }
                    snapshots.append(snapshot)
                    
                    # Print every 30 seconds
                    if sample_count % 15 == 0:
                        print(f"[{snapshot['timestamp'][:19]}] Bid: {yes_bid}¢, Ask: {yes_ask}¢, Spread: {snapshot['spread']}¢")
                
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n\n⚠ Monitoring stopped by user")
        except Exception as e:
            print(f"\n✗ Error during monitoring: {e}")
        
        # Analyze results
        self._analyze_market_speed(snapshots)
        
        return snapshots
    
    def _analyze_market_speed(self, snapshots: List[Dict]):
        """Analyze if the market is 'fast' or 'slow'."""
        print("\n" + "="*70)
        print("LITMUS TEST RESULTS: Market Speed Analysis")
        print("="*70)
        
        if len(snapshots) < 2:
            print("⚠ Not enough data to analyze market speed")
            return
        
        # Count price changes
        price_changes = 0
        for i in range(1, len(snapshots)):
            prev = snapshots[i-1]
            curr = snapshots[i]
            
            if (prev["yes_bid"] != curr["yes_bid"] or 
                prev["yes_ask"] != curr["yes_ask"]):
                price_changes += 1
        
        change_rate = (price_changes / len(snapshots)) * 100
        
        print(f"Total Samples: {len(snapshots)}")
        print(f"Price Changes Detected: {price_changes}")
        print(f"Change Rate: {change_rate:.1f}%")
        print(f"\nDuration: ~{len(snapshots) * 2} seconds")
        
        # Calculate average spread
        spreads = [s["spread"] for s in snapshots if s["spread"] is not None]
        if spreads:
            avg_spread = sum(spreads) / len(spreads)
            print(f"Average Spread: {avg_spread:.2f}¢")
        
        # Verdict
        print("\n" + "-"*70)
        if change_rate > 10:
            print("✓ VERDICT: FAST MARKET")
            print("  The market is actively trading with frequent price updates.")
            print("  This could be viable for real-time momentum trading.")
        elif change_rate > 1:
            print("⚠ VERDICT: MODERATE MARKET")
            print("  The market has occasional price changes but isn't highly active.")
            print("  May be viable but not ideal for rapid momentum plays.")
        else:
            print("✗ VERDICT: SLOW/STATIC MARKET")
            print("  The market shows minimal price movement.")
            print("  NOT viable for real-time momentum trading.")
            print("  This is likely too human-driven and slow for your use case.")
        print("-"*70)


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("KALSHI API INVESTIGATION")
    print("Target: Lakers vs Hawks - November 8, 2025")
    print("="*70)
    
    # Initialize investigator
    # Note: For read-only access, try without credentials first
    investigator = KalshiInvestigator()
    
    # Step 1: Test connection
    if not investigator.test_connection():
        print("\n" + "="*70)
        print("SETUP INSTRUCTIONS")
        print("="*70)
        print("To use the Kalshi API with authentication:")
        print("1. Sign up at https://kalshi.com")
        print("2. Generate API credentials in your account settings")
        print("3. Update this script with your credentials:")
        print("   investigator = KalshiInvestigator(")
        print("       api_key='your_api_key',")
        print("       private_key_path='path/to/private_key.pem'")
        print("   )")
        print("="*70)
        
        # Try to continue anyway
        print("\nAttempting to continue with public endpoints...")
    
    # Step 2: Search for the game
    markets = investigator.search_markets(["Lakers", "Hawks", "ATL", "LAL"])
    
    if not markets:
        print("\n" + "="*70)
        print("CONCLUSION: Cannot proceed without finding the market")
        print("="*70)
        print("\nPossible reasons:")
        print("1. The game market hasn't been created yet")
        print("2. Kalshi doesn't offer markets for this specific game")
        print("3. Authentication is required to see these markets")
        print("4. The Lakers may not be playing the Hawks tonight")
        print("\nRecommendation: Check Kalshi.com directly to see if this market exists.")
        return
    
    # Use the first matching event
    event = markets[0]
    event_ticker = event.get("event_ticker")
    
    # Step 3: Get market details
    market = investigator.get_market_details(event_ticker)
    
    if not market:
        print("\nCannot proceed without market data")
        return
    
    market_ticker = market.get("ticker")
    
    # Step 4: Get current prices
    prices = investigator.get_current_prices(market_ticker)
    
    if not prices:
        print("\nCannot get price data")
        return
    
    # Step 5: Run the litmus test
    print("\n" + "="*70)
    print("Starting the LITMUS TEST...")
    print("This will monitor the market for 5 minutes.")
    print("Press Ctrl+C to stop early if you've seen enough.")
    print("="*70)
    
    response = input("\nProceed with 5-minute monitoring? (y/n): ")
    if response.lower() == 'y':
        snapshots = investigator.monitor_price_changes(market_ticker, duration_minutes=5)
        
        # Save results to file
        results = {
            "event": event,
            "market": market,
            "initial_prices": prices,
            "snapshots": snapshots
        }
        
        filename = f"kalshi_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {filename}")
    else:
        print("\nSkipping extended monitoring.")
    
    print("\n" + "="*70)
    print("INVESTIGATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

