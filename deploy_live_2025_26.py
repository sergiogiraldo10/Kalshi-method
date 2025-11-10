"""
LIVE DEPLOYMENT SCRIPT FOR 2025-26 SEASON
==========================================

This script will:
1. Load the trained model (from ultimate_honest_test_2024_25.py)
2. Monitor live games
3. Identify trading opportunities
4. Alert you when to enter/exit trades

NOTE: You need to integrate this with a real-time NBA data feed
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import math

sys.path.append('src')

print("\n" + "="*70)
print("IGNITION AI - LIVE DEPLOYMENT FOR 2025-26 SEASON")
print("="*70)

# Configuration
CONFIG = {
    'min_confidence_percentile': 80,  # Top 20%
    'min_quality_score': 60,
    'position_size_pct': 0.05,
    'take_profit_pct': 0.25,
    'stop_loss_pct': -0.05,
    'max_trades_per_day': 5
}

print("\n[!] TO USE THIS SCRIPT:")
print("    1. You need to train and save the model first")
print("    2. Run ultimate_honest_test_2024_25.py and save the model")
print("    3. Integrate with a real-time NBA API")
print("    4. This is a TEMPLATE for live deployment")

print("\n" + "="*70)
print("TEMPLATE CODE FOR LIVE TRADING")
print("="*70)

# Example: How to use the model for live prediction
def predict_run_extension(features, model, scaler):
    """
    Predict if a 6-0 run will extend to 10+
    
    Parameters:
    - features: dict of current game state
    - model: trained XGBoost model
    - scaler: fitted StandardScaler
    
    Returns:
    - probability: float (0-1)
    - confidence: str ('LOW', 'MEDIUM', 'HIGH')
    """
    # Convert features to array
    feature_array = np.array([[
        features.get('run_score', 0),
        features.get('opp_score', 0),
        features.get('team_steals_2min', 0),
        features.get('team_blocks_2min', 0),
        # ... all 43 features
    ]])
    
    # Scale
    feature_scaled = scaler.transform(feature_array)
    
    # Predict
    probability = model.predict_proba(feature_scaled)[0, 1]
    
    # Confidence level
    if probability >= 0.35:
        confidence = 'HIGH'
    elif probability >= 0.30:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'
    
    return probability, confidence

def calculate_quality_score(game_state):
    """
    Calculate quality score for a run
    
    Parameters:
    - game_state: dict with current game info
    
    Returns:
    - quality_score: int (0-100)
    """
    score = 0
    
    # Pure run bonus
    if game_state.get('opp_score', 1) == 0:
        score += 20
    
    # Run size
    run_score = game_state.get('run_score', 0)
    if run_score >= 7:
        score += 15
    elif run_score >= 6:
        score += 10
    
    # Defensive pressure
    defensive_actions = (
        game_state.get('team_steals_2min', 0) +
        game_state.get('team_blocks_2min', 0) +
        game_state.get('opponent_turnovers_2min', 0)
    )
    if defensive_actions >= 3:
        score += 15
    elif defensive_actions >= 2:
        score += 10
    elif defensive_actions >= 1:
        score += 5
    
    # Three-pointers
    threes = game_state.get('team_threes_2min', 0)
    if threes >= 2:
        score += 10
    elif threes >= 1:
        score += 5
    
    # Team quality
    quality_diff = game_state.get('team_quality_diff', 0)
    if quality_diff > 0.10:
        score += 10
    elif quality_diff > 0:
        score += 5
    
    # Form
    form_diff = game_state.get('team_form_advantage', 0)
    if form_diff > 0.2:
        score += 10
    elif form_diff > 0:
        score += 5
    
    # Early game
    period = game_state.get('period', 1)
    if period <= 2:
        score += 10
    
    return min(score, 100)

def should_enter_trade(game_state, model, scaler, config):
    """
    Decide if we should enter a trade
    
    Parameters:
    - game_state: dict with current game info
    - model: trained model
    - scaler: fitted scaler
    - config: trading config
    
    Returns:
    - should_trade: bool
    - reason: str
    - confidence: float
    - quality: int
    """
    # Check basic criteria
    if game_state.get('run_score', 0) < 6:
        return False, "Run score < 6", 0, 0
    
    if game_state.get('opp_score', 0) > 0:
        return False, "Not a pure 6-0 run", 0, 0
    
    if game_state.get('period', 1) > 3:
        return False, "Q4 - too volatile", 0, 0
    
    # Calculate quality
    quality = calculate_quality_score(game_state)
    if quality < config['min_quality_score']:
        return False, f"Quality {quality} < {config['min_quality_score']}", 0, quality
    
    # Get model prediction
    probability, confidence_level = predict_run_extension(game_state, model, scaler)
    
    # Check confidence threshold (you'd calculate this from historical data)
    if probability < 0.345:  # Top 20% threshold from validation
        return False, f"Confidence {probability:.1%} < 34.5%", probability, quality
    
    # All checks passed
    return True, "TRADE SIGNAL", probability, quality

def calculate_position_size(bankroll, config):
    """Calculate position size"""
    return bankroll * config['position_size_pct']

def calculate_fee(position_size, probability):
    """Calculate trading fee"""
    fee = 0.07 * position_size * probability * (1 - probability)
    return math.ceil(fee * 100) / 100

# Example usage
print("\nEXAMPLE: How to use this in real-time")
print("-" * 70)

example_game_state = {
    'run_score': 6,
    'opp_score': 0,
    'period': 2,
    'team_steals_2min': 2,
    'team_blocks_2min': 1,
    'team_threes_2min': 3,
    'opponent_turnovers_2min': 2,
    'team_quality_diff': 0.15,
    'team_form_advantage': 0.10,
    # ... (43 features total)
}

print(f"\nGame State:")
print(f"  Run: {example_game_state['run_score']}-{example_game_state['opp_score']}")
print(f"  Quarter: {example_game_state['period']}")
print(f"  Defensive pressure: {example_game_state['team_steals_2min']} steals, {example_game_state['team_blocks_2min']} blocks")
print(f"  3-pointers: {example_game_state['team_threes_2min']}")

quality = calculate_quality_score(example_game_state)
print(f"\nQuality Score: {quality}/100")

print("\n[!] To get live predictions, you need to:")
print("    1. Load the trained model: model = joblib.load('models/ignition_ai_2025.pkl')")
print("    2. Connect to live NBA data feed")
print("    3. Extract features in real-time")
print("    4. Call should_enter_trade() for each game")
print("    5. Execute trades on betting platform")

print("\n" + "="*70)
print("RISK WARNING")
print("="*70)
print("\n  This is a trading algorithm. You can lose money.")
print("  Past performance does not guarantee future results.")
print("  Start with small amounts and paper trade first.")
print("  Never trade with money you can't afford to lose.")
print("  The model has a 35.8% win rate, not 100%.")
print("\n" + "="*70 + "\n")

