"""
Full Pipeline Runner
Executes the complete NBA momentum trading backtest pipeline
"""

import os
import sys
import time
from datetime import datetime

def print_header(message):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {message}")
    print("="*70 + "\n")

def print_section(message):
    """Print formatted section header"""
    print("\n" + "-"*70)
    print(f"  {message}")
    print("-"*70 + "\n")

def check_file_exists(filepath):
    """Check if file exists"""
    return os.path.exists(filepath)

def main():
    """
    Run the complete pipeline
    """
    print_header("NBA MOMENTUM TRADING BACKTEST - FULL PIPELINE")
    print("This script will execute the complete pipeline:")
    print("  1. Data Acquisition (download NBA play-by-play data)")
    print("  2. Data Validation")
    print("  3. Win Probability Model Training")
    print("  4. Feature Engineering")
    print("  5. Momentum Model Training")
    print("  6. Backtesting")
    print("  7. Results Visualization")
    print("\nEstimated total time: 4-6 hours")
    print("\nPress Ctrl+C to cancel, or wait 5 seconds to begin...")
    
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(0)
    
    start_time = time.time()
    
    # Step 1: Data Acquisition
    print_header("STEP 1: DATA ACQUISITION")
    
    # Check if data already exists
    seasons_to_check = ['2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24']
    data_exists = all(check_file_exists(f'data/raw/pbp_{s.replace("-", "_")}.csv') for s in seasons_to_check)
    
    if data_exists:
        print("Data files already exist. Skipping download.")
        print("To re-download, delete files in data/raw/ and run again.")
    else:
        print("Downloading NBA play-by-play data...")
        print("This will take approximately 3-4 hours...")
        from data_acquisition import main as acquire_data
        acquire_data()
    
    # Step 2: Data Validation
    print_header("STEP 2: DATA VALIDATION")
    
    if not check_file_exists('data/raw/pbp_2023_24.csv'):
        print("[ERROR] Data files not found. Please ensure data acquisition completed successfully.")
        sys.exit(1)
    
    from data_validation import main as validate_data
    validate_data()
    
    # Step 3: Win Probability Model
    print_header("STEP 3: WIN PROBABILITY MODEL TRAINING")
    
    if check_file_exists('models/win_probability_model.pkl'):
        print("Win probability model already exists. Skipping training.")
        print("To retrain, delete models/win_probability_model.pkl and run again.")
    else:
        from win_probability import train_win_probability_model
        train_win_probability_model()
    
    # Step 4: Feature Engineering
    print_header("STEP 4: FEATURE ENGINEERING")
    
    features_exist = all(check_file_exists(f'data/processed/features_{s.replace("-", "_")}.csv') 
                        for s in seasons_to_check)
    
    if features_exist:
        print("Feature files already exist. Skipping feature extraction.")
        print("To re-extract, delete files in data/processed/ and run again.")
    else:
        from feature_engineering import extract_all_features
        extract_all_features(seasons_to_check, include_outcomes=True)
    
    # Step 5: Momentum Model Training
    print_header("STEP 5: MOMENTUM MODEL TRAINING")
    
    if check_file_exists('models/momentum_model.pkl'):
        print("Momentum model already exists. Skipping training.")
        print("To retrain, delete models/momentum_model.pkl and run again.")
    else:
        from train_model import train_momentum_model
        train_momentum_model()
    
    # Step 6: Backtesting
    print_header("STEP 6: BACKTESTING")
    
    if check_file_exists('results/backtest_2023_24_trades.csv'):
        print("Backtest results already exist. Skipping backtest.")
        print("To re-run, delete files in results/ and run again.")
    else:
        from backtest import main as run_backtest
        run_backtest()
    
    # Step 7: Visualization
    print_header("STEP 7: RESULTS VISUALIZATION")
    
    from visualize_results import main as visualize
    visualize()
    
    # Complete
    total_time = time.time() - start_time
    
    print_header("PIPELINE COMPLETE!")
    print(f"Total execution time: {total_time/3600:.2f} hours")
    print("\nResults saved to:")
    print("  - results/backtest_2023_24_trades.csv       (detailed trade log)")
    print("  - results/backtest_2023_24_metrics.json     (performance metrics)")
    print("  - results/pl_chart_2023_24.png              (P/L chart)")
    print("  - results/trade_distribution_2023_24.png    (trade distribution)")
    print("  - results/exit_reasons_2023_24.png          (exit reasons)")
    print("  - results/summary_report_2023_24.txt        (text summary)")
    print("\nModels saved to:")
    print("  - models/win_probability_model.pkl")
    print("  - models/momentum_model.pkl")
    print("\nThank you for using the NBA Momentum Trading Backtest System!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

