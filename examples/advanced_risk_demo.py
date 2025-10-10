"""
Demonstration of Advanced Risk Management with the PyTradePath framework.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import BacktestingEngine
from core.data_handler import HistoricCSVDataHandler
from core.portfolio import NaivePortfolio
from core.execution import SimulatedExecutionHandler
from core.advanced_risk import AdvancedRiskManager, AdaptivePositionSizer
from strategies.momentum_strategy import MomentumStrategy
from core.event import SignalEvent


def create_sample_data():
    """Create sample market data for demonstration."""
    import csv
    import random
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate sample data for multiple symbols
    symbols = ['STOCK_A', 'STOCK_B', 'STOCK_C']
    
    for symbol in symbols:
        filepath = f'data/{symbol}.csv'
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['datetime', 'open', 'high', 'low', 'close', 'volume'])
            
            # Start with a base price
            price = 100.0 + random.uniform(-10, 10)
            
            # Generate 50 days of data
            for i in range(50):
                price_change = random.uniform(-0.03, 0.03)  # +/- 3% daily change
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


class RiskManagedMomentumStrategy(MomentumStrategy):
    """
    Momentum strategy with risk management integration.
    """
    
    def __init__(self, symbols, lookback_period=20, momentum_threshold=0.05):
        super().__init__(symbols, lookback_period, momentum_threshold)
        self.risk_manager = None
        
    def set_risk_manager(self, risk_manager):
        """Set the risk manager."""
        self.risk_manager = risk_manager


def main():
    """Run the advanced risk management demonstration."""
    print("PyTradePath Advanced Risk Management Demonstration")
    print("=" * 55)
    
    # Create sample data
    print("Creating sample market data...")
    create_sample_data()
    
    # Define symbols and capital
    symbols = ['STOCK_A', 'STOCK_B', 'STOCK_C']
    initial_capital = 100000.0
    
    # Create strategy
    def strategy_constructor(symbols):
        strategy = RiskManagedMomentumStrategy(symbols, lookback_period=10, momentum_threshold=0.02)
        return strategy
    
    # Create portfolio
    def portfolio_constructor(data_handler, events, initial_capital):
        portfolio = NaivePortfolio(data_handler, events, initial_capital=initial_capital)
        return portfolio
    
    # Create execution handler
    def execution_constructor(events):
        execution = SimulatedExecutionHandler(events)
        return execution
    
    # Create and run backtest
    print("\nCreating backtesting engine with advanced risk management...")
    engine = BacktestingEngine(
        data_handler=lambda: HistoricCSVDataHandler('data', symbols),
        strategy=strategy_constructor,
        portfolio=portfolio_constructor,
        execution_handler=execution_constructor,
        symbol_list=symbols,
        initial_capital=initial_capital
    )
    
    # Set up advanced risk manager
    risk_manager = AdvancedRiskManager(engine.portfolio)
    
    # Set sector data
    sector_data = {
        'STOCK_A': 'Technology',
        'STOCK_B': 'Technology',  # Same sector to demonstrate sector risk
        'STOCK_C': 'Healthcare'
    }
    
    for symbol, sector in sector_data.items():
        risk_manager.set_sector_data(symbol, sector)
    
    # Set liquidity data
    liquidity_data = {
        'STOCK_A': 5000000,  # $5M daily liquidity
        'STOCK_B': 2000000,  # $2M daily liquidity
        'STOCK_C': 10000000   # $10M daily liquidity
    }
    
    for symbol, liquidity in liquidity_data.items():
        risk_manager.set_liquidity_data(symbol, liquidity)
    
    # Set correlations (simplified)
    correlations = {
        'STOCK_A': {'STOCK_A': 1.0, 'STOCK_B': 0.85, 'STOCK_C': 0.3},
        'STOCK_B': {'STOCK_A': 0.85, 'STOCK_B': 1.0, 'STOCK_C': 0.25},
        'STOCK_C': {'STOCK_A': 0.3, 'STOCK_B': 0.25, 'STOCK_C': 1.0}
    }
    risk_manager.set_correlations(correlations)
    
    print("Running backtest with advanced risk management...")
    engine.run()
    
    # View results
    print("\nBacktest Results:")
    print(f"  Total signals generated: {engine.signals}")
    print(f"  Total orders placed: {engine.orders}")
    print(f"  Total trades executed: {engine.fills}")


if __name__ == "__main__":
    main()