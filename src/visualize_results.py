"""
Results Visualization Module
Creates charts and visualizations for backtest results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Set style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)

class ResultsVisualizer:
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        
    def load_results(self, filename_prefix):
        """
        Load backtest results
        """
        trades_file = os.path.join(self.results_dir, f'{filename_prefix}_trades.csv')
        metrics_file = os.path.join(self.results_dir, f'{filename_prefix}_metrics.json')
        bankroll_file = os.path.join(self.results_dir, f'{filename_prefix}_bankroll.csv')
        
        trades_df = pd.read_csv(trades_file) if os.path.exists(trades_file) else None
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        bankroll_df = pd.read_csv(bankroll_file) if os.path.exists(bankroll_file) else None
        
        return trades_df, metrics, bankroll_df
    
    def plot_cumulative_pnl(self, bankroll_df, season, save=True):
        """
        Plot cumulative P/L over time
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        initial_bankroll = bankroll_df['bankroll'].iloc[0]
        cumulative_pnl = bankroll_df['bankroll'] - initial_bankroll
        
        ax.plot(bankroll_df['trade_num'], cumulative_pnl, linewidth=2, color='#2E86AB')
        ax.fill_between(bankroll_df['trade_num'], 0, cumulative_pnl, alpha=0.3, color='#2E86AB')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Trade Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative P/L ($)', fontsize=12, fontweight='bold')
        ax.set_title(f'Cumulative P/L - Season {season}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add final value annotation
        final_pnl = cumulative_pnl.iloc[-1]
        ax.annotate(f'Final: ${final_pnl:,.2f}',
                   xy=(bankroll_df['trade_num'].iloc[-1], final_pnl),
                   xytext=(-80, 20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save:
            output_file = os.path.join(self.results_dir, f'pl_chart_{season.replace("-", "_")}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
        
        plt.close()
    
    def plot_trade_distribution(self, trades_df, season, save=True):
        """
        Plot distribution of trade profits/losses
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram of P/L
        ax1.hist(trades_df['pnl'], bins=30, color='#A23B72', alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax1.axvline(x=trades_df['pnl'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${trades_df["pnl"].mean():.2f}')
        ax1.set_xlabel('P/L per Trade ($)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Trade P/L Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Win/Loss pie chart
        profitable = len(trades_df[trades_df['pnl'] > 0])
        losses = len(trades_df[trades_df['pnl'] < 0])
        breakeven = len(trades_df[trades_df['pnl'] == 0])
        
        sizes = [profitable, losses, breakeven]
        labels = [f'Wins: {profitable}', f'Losses: {losses}', f'Breakeven: {breakeven}']
        colors = ['#06D6A0', '#EF476F', '#FFD166']
        explode = (0.05, 0.05, 0)
        
        ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax2.set_title('Win/Loss Ratio', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            output_file = os.path.join(self.results_dir, f'trade_distribution_{season.replace("-", "_")}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
        
        plt.close()
    
    def plot_exit_reasons(self, trades_df, season, save=True):
        """
        Plot distribution of exit reasons
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        exit_counts = trades_df['exit_reason'].value_counts()
        colors = ['#118AB2', '#073B4C', '#06D6A0', '#FFD166', '#EF476F']
        
        bars = ax.bar(exit_counts.index, exit_counts.values, color=colors[:len(exit_counts)])
        
        ax.set_xlabel('Exit Reason', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Trades', fontsize=12, fontweight='bold')
        ax.set_title(f'Trade Exit Reasons - Season {season}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save:
            output_file = os.path.join(self.results_dir, f'exit_reasons_{season.replace("-", "_")}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
        
        plt.close()
    
    def plot_probability_changes(self, trades_df, season, save=True):
        """
        Plot entry vs exit probabilities
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Color by P/L (green = profit, red = loss)
        colors = ['#06D6A0' if pnl > 0 else '#EF476F' for pnl in trades_df['pnl']]
        
        ax.scatter(trades_df['entry_prob'], trades_df['exit_prob'], 
                  c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Add diagonal line (no change)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='No Change')
        
        ax.set_xlabel('Entry Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Exit Probability', fontsize=12, fontweight='bold')
        ax.set_title(f'Entry vs Exit Probabilities - Season {season}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add custom legend for colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#06D6A0', edgecolor='black', label='Profitable'),
                          Patch(facecolor='#EF476F', edgecolor='black', label='Loss')]
        ax.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        
        if save:
            output_file = os.path.join(self.results_dir, f'probability_changes_{season.replace("-", "_")}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
        
        plt.close()
    
    def plot_holding_time_analysis(self, trades_df, season, save=True):
        """
        Plot holding time distribution and relationship to P/L
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Holding time distribution
        ax1.hist(trades_df['holding_time'], bins=30, color='#118AB2', alpha=0.7, edgecolor='black')
        ax1.axvline(x=trades_df['holding_time'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {trades_df["holding_time"].mean():.1f}s')
        ax1.set_xlabel('Holding Time (seconds)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Trade Holding Time Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Holding time vs P/L
        colors = ['#06D6A0' if pnl > 0 else '#EF476F' for pnl in trades_df['pnl']]
        ax2.scatter(trades_df['holding_time'], trades_df['pnl'], 
                   c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel('Holding Time (seconds)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('P/L ($)', fontsize=11, fontweight='bold')
        ax2.set_title('Holding Time vs P/L', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            output_file = os.path.join(self.results_dir, f'holding_time_{season.replace("-", "_")}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
        
        plt.close()
    
    def create_summary_report(self, metrics, season, save=True):
        """
        Create a text summary report
        """
        report = []
        report.append("="*60)
        report.append(f"NBA MOMENTUM TRADING BACKTEST RESULTS")
        report.append(f"Season: {season}")
        report.append("="*60)
        report.append("")
        
        report.append("TRADING PERFORMANCE")
        report.append("-"*60)
        report.append(f"Total Trades:              {metrics['total_trades']}")
        report.append(f"Profitable Trades:         {metrics['profitable_trades']} ({metrics['win_rate']*100:.2f}%)")
        report.append(f"Losing Trades:             {metrics['losing_trades']}")
        report.append("")
        
        report.append("PROFIT & LOSS")
        report.append("-"*60)
        report.append(f"Total P/L:                 ${metrics['total_pnl']:,.2f}")
        report.append(f"Total Fees Paid:           ${metrics['total_fees_paid']:,.2f}")
        report.append(f"Net P/L:                   ${metrics['net_pnl']:,.2f}")
        report.append(f"Average P/L per Trade:     ${metrics['avg_profit_per_trade']:,.2f}")
        report.append(f"Average Win:               ${metrics['avg_win']:,.2f}")
        report.append(f"Average Loss:              ${metrics['avg_loss']:,.2f}")
        report.append("")
        
        report.append("RISK METRICS")
        report.append("-"*60)
        report.append(f"ROI:                       {metrics['roi']*100:.2f}%")
        report.append(f"Sharpe Ratio:              {metrics['sharpe_ratio']:.3f}")
        report.append(f"Max Drawdown:              {metrics['max_drawdown']*100:.2f}%")
        report.append("")
        
        report.append("BANKROLL")
        report.append("-"*60)
        report.append(f"Initial Bankroll:          ${metrics.get('initial_bankroll', 10000):,.2f}")
        report.append(f"Final Bankroll:            ${metrics['final_bankroll']:,.2f}")
        report.append("")
        
        report.append("EXIT REASONS")
        report.append("-"*60)
        for reason, count in metrics['exit_reasons'].items():
            report.append(f"{reason:<25} {count}")
        report.append("")
        
        report.append("="*60)
        
        report_text = "\n".join(report)
        print(report_text)
        
        if save:
            output_file = os.path.join(self.results_dir, f'summary_report_{season.replace("-", "_")}.txt')
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"\nSaved: {output_file}")
        
        return report_text
    
    def visualize_all(self, filename_prefix, season):
        """
        Create all visualizations
        """
        print(f"\n{'='*60}")
        print(f"Creating Visualizations for {season}")
        print(f"{'='*60}\n")
        
        # Load results
        trades_df, metrics, bankroll_df = self.load_results(filename_prefix)
        
        if trades_df is None or bankroll_df is None:
            print("[ERROR] Could not load results files")
            return
        
        # Create all plots
        print("Creating P/L chart...")
        self.plot_cumulative_pnl(bankroll_df, season)
        
        print("Creating trade distribution chart...")
        self.plot_trade_distribution(trades_df, season)
        
        print("Creating exit reasons chart...")
        self.plot_exit_reasons(trades_df, season)
        
        print("Creating probability changes chart...")
        self.plot_probability_changes(trades_df, season)
        
        print("Creating holding time analysis...")
        self.plot_holding_time_analysis(trades_df, season)
        
        print("Creating summary report...")
        self.create_summary_report(metrics, season)
        
        print(f"\n{'='*60}")
        print("Visualization Complete!")
        print(f"{'='*60}\n")


def main():
    """
    Main visualization function
    """
    visualizer = ResultsVisualizer(results_dir='results')
    
    # Visualize 2023-24 season
    season = '2023-24'
    filename_prefix = f'backtest_{season.replace("-", "_")}'
    
    visualizer.visualize_all(filename_prefix, season)
    
    print("\nAll visualizations saved to 'results/' directory")


if __name__ == '__main__':
    # Install matplotlib and seaborn if needed
    try:
        import matplotlib
        import seaborn
    except ImportError:
        print("Installing visualization dependencies...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'matplotlib', 'seaborn'])
    
    main()

