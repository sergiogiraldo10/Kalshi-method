"""
Backtesting Framework
Runs complete backtest simulation combining all components
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Import our modules
from win_probability import WinProbabilityModel
from feature_engineering import MomentumFeatureExtractor
from train_model import MomentumModel
from trading_simulator import TradingSimulator

class Backtester:
    def __init__(self, 
                 momentum_model_path='models/momentum_model.pkl',
                 win_prob_model_path='models/win_probability_model.pkl'):
        
        print("\n" + "="*60)
        print("Initializing Backtester")
        print("="*60 + "\n")
        
        # Load models
        print("Loading momentum model...")
        self.momentum_model = MomentumModel()
        if os.path.exists(momentum_model_path):
            self.momentum_model.load(momentum_model_path)
        else:
            print(f"[WARNING] Momentum model not found at {momentum_model_path}")
            print("Please run train_model.py first")
            sys.exit(1)
        
        print("Loading win probability model...")
        self.win_prob_model = WinProbabilityModel()
        if os.path.exists(win_prob_model_path):
            self.win_prob_model.load(win_prob_model_path)
        else:
            print(f"[WARNING] Win probability model not found at {win_prob_model_path}")
            print("Please run win_probability.py first")
            sys.exit(1)
        
        self.feature_extractor = MomentumFeatureExtractor()
        
        print("\nBacktester initialized successfully!\n")
    
    def backtest_season(self, season, pbp_data_path, simulator_config=None):
        """
        Run backtest on a single season
        """
        print(f"\n" + "="*60)
        print(f"Backtesting Season: {season}")
        print("="*60 + "\n")
        
        # Load play-by-play data
        print(f"Loading play-by-play data from {pbp_data_path}...")
        pbp_df = pd.read_csv(pbp_data_path)
        print(f"Loaded {len(pbp_df):,} plays from {pbp_df['GAME_ID'].nunique()} games")
        
        # Initialize trading simulator
        if simulator_config is None:
            simulator_config = {}
        
        simulator = TradingSimulator(**simulator_config)
        print(f"\nTrading Simulator Configuration:")
        print(f"  Initial Bankroll: ${simulator.initial_bankroll:,.2f}")
        print(f"  Position Size: {simulator.position_size_pct*100:.1f}% of bankroll")
        print(f"  Entry Fee: {simulator.entry_fee_pct*100:.2f}%")
        print(f"  Exit Fee: {simulator.exit_fee_pct*100:.2f}%")
        print(f"  Take Profit: {simulator.take_profit_pct*100:.1f}%")
        print(f"  Stop Loss: {simulator.stop_loss_pct*100:.1f}%")
        print(f"  Min Momentum Confidence: {simulator.min_momentum_confidence:.2f}")
        
        # Process games chronologically
        games = pbp_df['GAME_ID'].unique()
        print(f"\nProcessing {len(games)} games...")
        
        games_processed = 0
        
        for game_id in games:
            game_df = pbp_df[pbp_df['GAME_ID'] == game_id].copy()
            game_df = game_df.sort_values('EVENTNUM')
            
            # Run game simulation
            self._simulate_game(game_df, game_id, simulator)
            
            games_processed += 1
            if games_processed % 50 == 0:
                current_metrics = simulator.get_performance_metrics()
                print(f"  Processed {games_processed}/{len(games)} games | "
                      f"Trades: {current_metrics.get('total_trades', 0)} | "
                      f"Bankroll: ${simulator.current_bankroll:,.2f}")
        
        print(f"\nBacktest complete! Processed {games_processed} games")
        
        # Get final metrics
        metrics = simulator.get_performance_metrics()
        self._print_metrics(metrics)
        
        return simulator, metrics
    
    def _simulate_game(self, game_df, game_id, simulator):
        """
        Simulate trading for a single game
        """
        # Track game state
        score_history = []
        event_history = []
        
        for idx, row in game_df.iterrows():
            # Parse score
            if pd.isna(row['SCORE']):
                continue
            
            try:
                score_parts = str(row['SCORE']).split(' - ')
                if len(score_parts) != 2:
                    continue
                away_score = int(score_parts[0])
                home_score = int(score_parts[1])
            except:
                continue
            
            # Parse time
            try:
                period = int(row['PERIOD'])
                time_parts = str(row['PCTIMESTRING']).split(':')
                if len(time_parts) != 2:
                    continue
                mins = int(time_parts[0])
                secs = int(time_parts[1])
                time_in_period = mins * 60 + secs
            except:
                continue
            
            # Calculate time remaining
            if period <= 4:
                time_remaining = (4 - period) * 720 + time_in_period
            else:
                time_remaining = max(0, (5 - (period - 4)) * 300 + time_in_period)
            
            # Update history
            score_history.append({
                'home': home_score,
                'away': away_score,
                'time_remaining': time_remaining
            })
            
            event_history.append({
                'event_type': row['EVENTMSGTYPE'],
                'time_remaining': time_remaining
            })
            
            # Need minimum history to make predictions
            if len(score_history) < 10:
                continue
            
            # Calculate current win probability
            score_diff = home_score - away_score
            current_home_win_prob = self.win_prob_model.predict_win_probability(
                score_diff, time_remaining, period
            )
            
            # Extract momentum features for this moment
            features = self._extract_current_features(score_history, event_history, period, time_remaining)
            
            if features is None:
                continue
            
            # Create feature DataFrame for prediction
            features_df = pd.DataFrame([features])
            
            # Predict momentum extension probability
            momentum_confidence = self.momentum_model.predict(features_df)[0]
            
            # Determine team with momentum
            if features.get('home_has_momentum', 0) > 0:
                team_with_momentum = 'home'
                team_win_prob = current_home_win_prob
            elif features.get('away_has_momentum', 0) > 0:
                team_with_momentum = 'away'
                team_win_prob = 1 - current_home_win_prob
            else:
                team_with_momentum = None
                team_win_prob = None
            
            # Check if should enter trade
            if team_with_momentum and momentum_confidence >= simulator.min_momentum_confidence:
                max_run = features.get('max_current_run', 0)
                simulator.enter_trade(
                    game_id, time_remaining, team_with_momentum, 
                    team_win_prob, momentum_confidence
                )
            
            # Check exit conditions for current position
            if simulator.current_position and simulator.current_position.game_id == game_id:
                # Get current probability for position's team
                if simulator.current_position.team == 'home':
                    current_prob = current_home_win_prob
                else:
                    current_prob = 1 - current_home_win_prob
                
                # Check TP/SL
                exit_reason, exit_prob = simulator.check_exit_conditions(time_remaining, current_prob)
                if exit_reason:
                    simulator.close_trade(time_remaining, exit_prob, exit_reason)
        
        # Force close any open position at game end
        if score_history:
            final_score = score_history[-1]
            home_won = final_score['home'] > final_score['away']
            simulator.force_close_at_game_end(game_id, home_won)
    
    def _extract_current_features(self, score_history, event_history, period, time_remaining):
        """
        Extract features for current game moment
        """
        if len(score_history) < 10:
            return None
        
        # Use feature extractor's momentum calculation
        recent_scores = score_history[-20:]
        
        features = {}
        
        # Calculate current runs
        home_run, away_run = self._calculate_current_run(recent_scores, time_remaining)
        
        features['home_current_run'] = home_run
        features['away_current_run'] = away_run
        features['max_current_run'] = max(home_run, away_run)
        features['run_differential'] = home_run - away_run
        features['is_micro_run'] = int(features['max_current_run'] >= 4)
        features['is_significant_run'] = int(features['max_current_run'] >= 6)
        features['home_has_momentum'] = int(home_run > away_run and home_run >= 4)
        features['away_has_momentum'] = int(away_run > home_run and away_run >= 4)
        
        # Scoring pace
        features['points_last_2min'] = self._points_in_timeframe(recent_scores, time_remaining, 120)
        features['points_last_4min'] = self._points_in_timeframe(recent_scores, time_remaining, 240)
        features['scoring_rate_last_2min'] = features['points_last_2min'] / 2.0
        
        # Time since opponent score
        features['time_since_opponent_score'] = 60  # Simplified
        
        # Game state
        features['period'] = period
        features['time_remaining_minutes'] = time_remaining / 60.0
        features['is_close_game'] = int(abs(score_history[-1]['home'] - score_history[-1]['away']) <= 5)
        features['is_clutch_time'] = int(time_remaining <= 300 and features['is_close_game'])
        
        # Volatility
        features['score_volatility'] = self._calculate_volatility(recent_scores)
        
        return features
    
    def _calculate_current_run(self, score_history, current_time, lookback=120):
        """Calculate current run for each team"""
        if len(score_history) < 2:
            return 0, 0
        
        home_run = 0
        away_run = 0
        
        for i in range(len(score_history) - 1, 0, -1):
            current = score_history[i]
            previous = score_history[i - 1]
            
            if current_time - current['time_remaining'] > lookback:
                break
            
            home_points = current['home'] - previous['home']
            away_points = current['away'] - previous['away']
            
            if home_points > 0:
                home_run += home_points
            if away_points > 0:
                away_run += away_points
        
        return home_run, away_run
    
    def _points_in_timeframe(self, score_history, current_time, timeframe):
        """Calculate points in timeframe"""
        if len(score_history) < 2:
            return 0
        
        total_points = 0
        start_time = current_time - timeframe
        
        for i in range(len(score_history) - 1, 0, -1):
            current = score_history[i]
            
            if current['time_remaining'] < start_time:
                break
            
            if i > 0:
                previous = score_history[i - 1]
                points = (current['home'] + current['away']) - (previous['home'] + previous['away'])
                total_points += points
        
        return total_points
    
    def _calculate_volatility(self, score_history):
        """Calculate score volatility"""
        if len(score_history) < 3:
            return 0
        
        changes = []
        for i in range(1, len(score_history)):
            total = (score_history[i]['home'] + score_history[i]['away']) - \
                   (score_history[i-1]['home'] + score_history[i-1]['away'])
            changes.append(total)
        
        return np.std(changes) if changes else 0
    
    def _print_metrics(self, metrics):
        """
        Print backtest metrics
        """
        print(f"\n" + "="*60)
        print("Backtest Results")
        print("="*60 + "\n")
        
        print(f"Trading Performance:")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Profitable Trades: {metrics['profitable_trades']}")
        print(f"  Losing Trades: {metrics['losing_trades']}")
        print(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
        print(f"\nP/L:")
        print(f"  Total P/L: ${metrics['total_pnl']:,.2f}")
        print(f"  Total Fees: ${metrics['total_fees_paid']:,.2f}")
        print(f"  Net P/L: ${metrics['net_pnl']:,.2f}")
        print(f"  Avg P/L per Trade: ${metrics['avg_profit_per_trade']:,.2f}")
        print(f"  Avg Win: ${metrics['avg_win']:,.2f}")
        print(f"  Avg Loss: ${metrics['avg_loss']:,.2f}")
        print(f"\nRisk Metrics:")
        print(f"  ROI: {metrics['roi']*100:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"\nBankroll:")
        print(f"  Final Bankroll: ${metrics['final_bankroll']:,.2f}")
        print(f"\nExit Reasons:")
        for reason, count in metrics['exit_reasons'].items():
            print(f"  {reason}: {count}")


def main():
    """
    Main backtesting function
    """
    print("\n" + "#"*60)
    print("# NBA Momentum Trading Backtest")
    print("#"*60 + "\n")
    
    # Initialize backtester
    backtester = Backtester()
    
    # Backtest on 2023-24 season
    test_season = '2023-24'
    pbp_path = f'data/raw/pbp_{test_season.replace("-", "_")}.csv'
    
    if not os.path.exists(pbp_path):
        print(f"[ERROR] Test data not found: {pbp_path}")
        print("Please run data_acquisition.py first")
        return
    
    # Run backtest
    simulator, metrics = backtester.backtest_season(
        season=test_season,
        pbp_data_path=pbp_path,
        simulator_config={
            'initial_bankroll': 10000,
            'position_size_pct': 0.01,
            'entry_fee_pct': 0.015,
            'exit_fee_pct': 0.015,
            'take_profit_pct': 0.08,
            'stop_loss_pct': -0.04,
            'min_momentum_confidence': 0.65
        }
    )
    
    # Save results
    print(f"\nSaving results...")
    simulator.save_results(output_dir='results', filename_prefix=f'backtest_{test_season.replace("-", "_")}')
    
    print(f"\n" + "#"*60)
    print("# Backtest Complete!")
    print("#"*60 + "\n")


if __name__ == '__main__':
    main()

