"""
Demonstration of the Momentum Strategy with the PyTradePath framework.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import BacktestingEngine
from core.data_handler import HistoricCSVDataHandler
from core.portfolio import NaivePortfolio
from core.execution import SimulatedExecutionHandler
from strategies.momentum_strategy import MomentumStrategy


def create_sample_data():
    """Create sample market data for demonstration."""
    import csv
    import random
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate sample data with momentum characteristics
    symbols = ['MOMENTUM_STOCK']
    
    for symbol in symbols:
        filepath = f'data/{symbol}.csv'
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['datetime', 'open', 'high', 'low', 'close', 'volume'])
            
            # Start with a base price
            price = 100.0
            
            # Generate 100 days of data with momentum patterns
            for i in range(100):
                # Create momentum by having trends
                if i < 30:  # Upward momentum
                    price_change = random.uniform(0.01, 0.03)  # 1-3% daily gains
                elif i < 60:  # Downward momentum
                    price_change = random.uniform(-0.03, -0.01)  # 1-3% daily losses
                else:  # Recovery
                    price_change = random.uniform(0.005, 0.02)  # 0.5-2% daily gains
                
                price = price * (1 + price_change)
                
                # Add some noise
                open_price = price * random.uniform(0.995, 1.005)
                high_price = price * random.uniform(1.001, 1.01)
                low_price = price * random.uniform(0.99, 0.999)
                close_price = price
                volume = random.randint(1000, 10000)
                
                writer.writerow([
                    f'2023-01-{i+1:02d}',
                    round(open_price, 2),
                    round(high_price, 2),
                    round(low_price, 2),
                    round(close_price, 2),
                    volume
                ])
    
    print(f"Created sample data for {len(symbols)} symbols")


def main():
    """Run the momentum strategy demonstration."""
    print("PyTradePath Momentum Strategy Demonstration")
    print("=" * 50)
    
    # Create sample data
    print("Creating sample market data...")
    create_sample_data()
    
    # Define symbols and capital
    symbols = ['MOMENTUM_STOCK']
    initial_capital = 100000.0
    
    # Create and run backtest
    print("\nCreating backtesting engine...")
    engine = BacktestingEngine(
        data_handler=lambda: HistoricCSVDataHandler('data', symbols),
        strategy=lambda symbols: MomentumStrategy(symbols, lookback_period=10, momentum_threshold=0.02),
        portfolio=lambda data_handler, events, initial_capital: NaivePortfolio(
            data_handler, events, initial_capital=initial_capital),
        execution_handler=lambda events: SimulatedExecutionHandler(events),
        symbol_list=symbols,
        initial_capital=initial_capital
    )
    
    print("Running backtest...")
    engine.run()
    
    # View results
    print("\nBacktest Results:")
    print(f"  Total signals generated: {engine.signals}")
    print(f"  Total orders placed: {engine.orders}")
    print(f"  Total trades executed: {engine.fills}")


if __name__ == "__main__":
    main()