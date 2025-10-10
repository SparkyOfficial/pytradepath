"""
Comprehensive example demonstrating the full pytradepath framework.
This example shows how to use all components together in a complete workflow.
"""

import sys
import os
import csv
from datetime import datetime, timedelta
import warnings
import json
import random

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import framework components
from core.event import MarketEvent, SignalEvent, OrderEvent, FillEvent
from core.data_handler import HistoricCSVDataHandler
from core.strategy import BuyAndHoldStrategy
from core.portfolio import NaivePortfolio
from core.execution import SimulatedExecutionHandler
from core.engine import BacktestingEngine


def create_sample_data():
    """Create sample market data for testing."""
    # Create sample data directory
    os.makedirs("data", exist_ok=True)
    
    # Create sample data for multiple symbols
    symbols = ["sample_data"]
    
    for symbol in symbols:
        # Create sample CSV data
        with open(f"data/{symbol}.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['datetime', 'open', 'high', 'low', 'close', 'volume'])
            
            # Generate 50 days of sample data
            for i in range(50):
                date = f"2023-01-{i+1:02d}"
                open_price = 100 + i * 0.5
                high_price = open_price + random.uniform(0, 2)
                low_price = open_price - random.uniform(0, 2)
                close_price = low_price + random.uniform(0, high_price - low_price)
                volume = random.randint(1000, 10000)
                
                writer.writerow([date, open_price, high_price, low_price, close_price, volume])
    
    print(f"Created sample data for {len(symbols)} symbols")


def run_backtest_example():
    """Run a comprehensive backtest example."""
    print("=== Running Backtest Example ===")
    
    # Create sample data
    create_sample_data()
    
    # Define symbols to trade
    symbols = ["sample_data"]
    
    # Create backtesting engine
    engine = BacktestingEngine(
        data_handler=lambda: HistoricCSVDataHandler('data', symbols),
        strategy=lambda symbols: BuyAndHoldStrategy(symbols),
        portfolio=lambda data_handler, events, initial_capital: NaivePortfolio(
            data_handler, events, initial_capital=100000.0),
        execution_handler=lambda events: SimulatedExecutionHandler(events),
        symbol_list=symbols,
        initial_capital=100000.0
    )
    
    # Run backtest
    engine.run()
    
    # Print results
    print(f"Backtest completed:")
    print(f"  Signals generated: {engine.signals}")
    print(f"  Orders placed: {engine.orders}")
    print(f"  Trades executed: {engine.fills}")
    
    return engine


def main():
    """Run the example."""
    print("PyTradePath Comprehensive Example")
    print("=" * 50)
    
    try:
        # Run backtest example
        backtest_engine = run_backtest_example()
        
        print("\n" + "=" * 50)
        print("Example completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()