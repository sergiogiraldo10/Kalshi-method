"""
Trading Simulator Module
Simulates realistic momentum-based trading with position management, TP/SL, and fees
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

class Trade:
    def __init__(self, trade_id, game_id, entry_time, team, entry_prob, position_size, entry_fee_pct):
        self.trade_id = trade_id
        self.game_id = game_id
        self.entry_time = entry_time
        self.team = team  # 'home' or 'away'
        self.entry_prob = entry_prob
        self.position_size = position_size
        self.entry_fee = position_size * entry_fee_pct
        self.total_cost = position_size + self.entry_fee
        
        self.exit_time = None
        self.exit_prob = None
        self.exit_reason = None
        self.exit_fee = 0
        self.pnl = None
        self.team_won = None
        
    def close(self, exit_time, exit_prob, exit_reason, exit_fee_pct, team_won=None):
        """
        Close the trade
        """
        self.exit_time = exit_time
        self.exit_prob = exit_prob
        self.exit_reason = exit_reason
        self.team_won = team_won
        
        # Calculate P/L based on exit reason
        if exit_reason == 'game_end':
            # If game ended, P/L based on whether team won
            if team_won:
                # Win: receive (stake / entry_prob)
                payout = self.position_size / self.entry_prob
                self.pnl = payout - self.total_cost
            else:
                # Loss: lose stake and fees
                self.pnl = -self.total_cost
            
            # Apply exit fee only if won
            if team_won:
                self.exit_fee = payout * exit_fee_pct
                self.pnl -= self.exit_fee
        else:
            # Exited before game end (TP or SL)
            # Estimate value based on probability change
            # If prob increased: positive P/L proportional to increase
            # If prob decreased: negative P/L proportional to decrease
            
            prob_change = exit_prob - self.entry_prob
            
            # Calculate value based on probability
            # Simplified: value = stake * (exit_prob / entry_prob)
            current_value = self.position_size * (exit_prob / self.entry_prob)
            
            self.exit_fee = current_value * exit_fee_pct
            self.pnl = current_value - self.total_cost - self.exit_fee
    
    def to_dict(self):
        """
        Convert trade to dictionary for logging
        """
        return {
            'trade_id': self.trade_id,
            'game_id': self.game_id,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'team': self.team,
            'entry_prob': self.entry_prob,
            'exit_prob': self.exit_prob,
            'position_size': self.position_size,
            'entry_fee': self.entry_fee,
            'exit_fee': self.exit_fee,
            'total_cost': self.total_cost,
            'exit_reason': self.exit_reason,
            'team_won': self.team_won,
            'pnl': self.pnl,
            'holding_time': (self.exit_time - self.entry_time) if self.exit_time else None
        }


class TradingSimulator:
    def __init__(self, 
                 initial_bankroll=10000,
                 position_size_pct=0.01,
                 entry_fee_pct=0.015,
                 exit_fee_pct=0.015,
                 take_profit_pct=0.08,
                 stop_loss_pct=-0.04,
                 min_momentum_confidence=0.6):
        
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.position_size_pct = position_size_pct
        self.entry_fee_pct = entry_fee_pct
        self.exit_fee_pct = exit_fee_pct
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.min_momentum_confidence = min_momentum_confidence
        
        self.trades = []
        self.current_position = None
        self.trade_counter = 0
        
        self.bankroll_history = [initial_bankroll]
        
    def should_enter_trade(self, momentum_confidence, current_run):
        """
        Determine if we should enter a trade based on momentum signal
        """
        # Need significant momentum and model confidence
        return (momentum_confidence >= self.min_momentum_confidence and 
                current_run >= 4)
    
    def calculate_position_size(self):
        """
        Calculate position size based on current bankroll
        """
        return self.current_bankroll * self.position_size_pct
    
    def enter_trade(self, game_id, time_remaining, team_with_momentum, win_probability, momentum_confidence):
        """
        Enter a new trade
        """
        # Check if should enter
        if not self.should_enter_trade(momentum_confidence, 4):  # Assume 4+ run
            return None
        
        # Close conflicting position
        if self.current_position is not None:
            if self.current_position.team != team_with_momentum:
                # Different team - close old position first
                self.close_trade(
                    time_remaining, 
                    win_probability if self.current_position.team == 'home' else (1 - win_probability),
                    'position_flip',
                    team_won=None
                )
        
        # Calculate position size
        position_size = self.calculate_position_size()
        
        # Create new trade
        self.trade_counter += 1
        trade = Trade(
            trade_id=self.trade_counter,
            game_id=game_id,
            entry_time=time_remaining,
            team=team_with_momentum,
            entry_prob=win_probability,
            position_size=position_size,
            entry_fee_pct=self.entry_fee_pct
        )
        
        self.current_position = trade
        self.current_bankroll -= trade.total_cost
        
        return trade
    
    def check_exit_conditions(self, time_remaining, current_win_prob):
        """
        Check if current position should be exited (TP or SL)
        """
        if self.current_position is None:
            return None, None
        
        # Calculate probability change
        prob_change = current_win_prob - self.current_position.entry_prob
        prob_change_pct = prob_change / self.current_position.entry_prob
        
        # Check take profit
        if prob_change_pct >= self.take_profit_pct:
            return 'take_profit', current_win_prob
        
        # Check stop loss
        if prob_change_pct <= self.stop_loss_pct:
            return 'stop_loss', current_win_prob
        
        return None, None
    
    def close_trade(self, time_remaining, current_win_prob, exit_reason, team_won=None):
        """
        Close the current position
        """
        if self.current_position is None:
            return
        
        self.current_position.close(
            exit_time=time_remaining,
            exit_prob=current_win_prob,
            exit_reason=exit_reason,
            exit_fee_pct=self.exit_fee_pct,
            team_won=team_won
        )
        
        # Update bankroll
        self.current_bankroll += self.current_position.pnl
        
        # Log trade
        self.trades.append(self.current_position)
        self.bankroll_history.append(self.current_bankroll)
        
        # Clear position
        self.current_position = None
    
    def force_close_at_game_end(self, game_id, home_won):
        """
        Force close any open position when game ends
        """
        if self.current_position is None or self.current_position.game_id != game_id:
            return
        
        team_won = (self.current_position.team == 'home' and home_won) or \
                   (self.current_position.team == 'away' and not home_won)
        
        final_prob = 1.0 if team_won else 0.0
        
        self.close_trade(
            time_remaining=0,
            current_win_prob=final_prob,
            exit_reason='game_end',
            team_won=team_won
        )
    
    def get_performance_metrics(self):
        """
        Calculate trading performance metrics
        """
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame([t.to_dict() for t in self.trades])
        
        total_trades = len(trades_df)
        profitable_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        total_pnl = trades_df['pnl'].sum()
        total_fees_paid = trades_df['entry_fee'].sum() + trades_df['exit_fee'].sum()
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if profitable_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Calculate by exit reason
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
        
        # Sharpe ratio (simplified)
        returns = trades_df['pnl'] / self.initial_bankroll
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        sharpe_ratio *= np.sqrt(252)  # Annualized (assuming ~252 trading days)
        
        # Maximum drawdown
        cumulative = np.array(self.bankroll_history)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        metrics = {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'losing_trades': losing_trades,
            'win_rate': profitable_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'total_fees_paid': total_fees_paid,
            'net_pnl': total_pnl,
            'avg_profit_per_trade': total_pnl / total_trades if total_trades > 0 else 0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'roi': (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll,
            'final_bankroll': self.current_bankroll,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'exit_reasons': exit_reasons,
            'avg_holding_time': trades_df['holding_time'].mean() if 'holding_time' in trades_df.columns else 0
        }
        
        return metrics
    
    def save_results(self, output_dir='results', filename_prefix='backtest'):
        """
        Save trading results to files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save trades log
        trades_df = pd.DataFrame([t.to_dict() for t in self.trades])
        trades_file = os.path.join(output_dir, f'{filename_prefix}_trades.csv')
        trades_df.to_csv(trades_file, index=False)
        print(f"Trades saved to {trades_file}")
        
        # Save performance metrics
        metrics = self.get_performance_metrics()
        metrics_file = os.path.join(output_dir, f'{filename_prefix}_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"Metrics saved to {metrics_file}")
        
        # Save bankroll history
        bankroll_df = pd.DataFrame({
            'trade_num': range(len(self.bankroll_history)),
            'bankroll': self.bankroll_history
        })
        bankroll_file = os.path.join(output_dir, f'{filename_prefix}_bankroll.csv')
        bankroll_df.to_csv(bankroll_file, index=False)
        print(f"Bankroll history saved to {bankroll_file}")
        
        return trades_file, metrics_file, bankroll_file


if __name__ == '__main__':
    # Test simulator
    print("Trading Simulator Module")
    print("This module should be used by backtest.py")

