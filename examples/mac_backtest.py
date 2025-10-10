"""
Example script demonstrating how to use the pytradepath framework
for backtesting a moving average crossover strategy.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import BacktestingEngine
from core.data_handler import HistoricCSVDataHandler
from strategies.moving_average_crossover import MovingAverageCrossoverStrategy
from core.portfolio import NaivePortfolio
from core.execution import SimulatedExecutionHandler


def run_mac_example():
    """
    Run a moving average crossover strategy backtest.
    """
    # Define the symbols to trade
    symbols = ['sample_data']  # Using our sample data
    
    # Define initial capital
    initial_capital = 100000.0
    
    # Create the backtesting engine
    engine = BacktestingEngine(
        data_handler=lambda: HistoricCSVDataHandler('data', symbols),
        strategy=lambda symbols: MovingAverageCrossoverStrategy(symbols, short_window=3, long_window=5),
        portfolio=lambda data_handler, events, initial_capital: NaivePortfolio(data_handler, events, initial_capital=initial_capital),
        execution_handler=lambda events: SimulatedExecutionHandler(events),
        symbol_list=symbols,
        initial_capital=initial_capital
    )
    
    # Run the backtest
    engine.run()


if __name__ == "__main__":
    run_mac_example()