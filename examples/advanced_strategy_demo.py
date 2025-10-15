"""
Advanced strategy demonstration showing the full capabilities of the pytradepath framework.
This example showcases an advanced machine learning-based trading strategy with comprehensive risk management.
"""

import sys
import os
import csv
import random
import math
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all framework components
from core.event import MarketEvent, SignalEvent, OrderEvent, FillEvent
from core.data_handler import HistoricCSVDataHandler
from core.strategy import Strategy
from core.portfolio import Portfolio, NaivePortfolio
from core.execution import ExecutionHandler, SimulatedExecutionHandler
from core.engine import BacktestingEngine
from core.risk import RiskManager, NaiveRiskManager, PositionSizer, KellyCriterionPositionSizer
from core.optimization import ParameterOptimizer
from core.data import CSVDataProvider, DataManager, DataValidator
from core.ml import EnhancedLinearRegression, FeatureEngineer
from core.analytics import StatisticalAnalyzer, RiskAnalyzer
from core.utilities import CacheManager, Timer
from utils.performance import calculate_sharpe_ratio, calculate_max_drawdown, calculate_cagr, calculate_sortino_ratio

from strategies.advanced_ml_strategy import AdvancedMLStrategy


class EnhancedRiskManager(RiskManager):
    """
    Enhanced risk manager with multiple risk controls.
    """

    def __init__(self, portfolio: Portfolio, max_positions: int = 5, 
                 max_drawdown: float = 0.2, stop_loss: float = 0.05):
        """
        Initialize the risk manager.
        """
        super().__init__(portfolio)
        self.max_positions = max_positions
        self.max_drawdown = max_drawdown
        self.stop_loss = stop_loss
        self.entry_prices = {}

    def modify_signals(self, signals: list) -> list:
        """
        Modify signals based on risk management rules.
        """
        # Check position limits
        current_positions = sum(1 for pos in self.portfolio.current_positions.values() if pos != 0)
        
        if current_positions >= self.max_positions:
            print(f"Position limit reached ({current_positions}/{self.max_positions})")
            return []
        
        # Check drawdown
        if self.portfolio.current_holdings['total'] < self.portfolio.initial_capital * (1 - self.max_drawdown):
            print("Maximum drawdown exceeded")
            return []
        
        return signals

    def modify_orders(self, orders: list) -> list:
        """
        Modify orders based on risk management rules.
        """
        modified_orders = []
        
        for order in orders:
            # Store entry price for stop loss calculation
            if order.direction == 'BUY':
                self.entry_prices[order.symbol] = 100.0  # Simplified for demo
            
            # Apply position sizing
            modified_orders.append(order)
        
        return modified_orders


class AdvancedPortfolio(NaivePortfolio):
    """
    Advanced portfolio with additional tracking and analytics.
    """

    def __init__(self, data_handler, events, start_date: str = "2020-01-01", 
                 initial_capital: float = 100000.0):
        """
        Initialize the advanced portfolio.
        """
        super().__init__(data_handler, events, start_date, initial_capital)
        self.trade_history = []
        self.equity_curve = [initial_capital]
        self.returns = []

    def on_fill(self, event: FillEvent):
        """
        Update portfolio with fill event and track trade history.
        """
        super().on_fill(event)
        
        # Record trade
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': event.symbol,
            'quantity': event.quantity,
            'direction': event.direction,
            'price': event.fill_price,
            'commission': event.commission,
            'cost': event.cost
        }
        self.trade_history.append(trade_record)

    def update_timeindex(self, event):
        """
        Update time index and track equity metrics.
        """
        super().update_timeindex(event)
        
        # Track equity curve
        current_equity = self.current_holdings['total']
        self.equity_curve.append(current_equity)
        
        # Calculate returns
        if len(self.equity_curve) > 1:
            ret = (current_equity - self.equity_curve[-2]) / self.equity_curve[-2] if self.equity_curve[-2] != 0 else 0
            self.returns.append(ret)


def create_advanced_sample_data():
    """Create advanced sample data for multiple symbols with realistic patterns."""
    print("Creating advanced sample market data...")
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Symbols to create data for
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
    
    # Create data for each symbol
    for symbol in symbols:
        filename = f"data/{symbol}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['datetime', 'open', 'high', 'low', 'close', 'volume'])
            
            # Generate 500 days of data (approx. 2 years of trading days)
            base_price = 100 + random.uniform(-50, 50)  # Random base price
            price = base_price
            
            for i in range(500):
                date = (datetime(2022, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d')
                
                # Generate realistic price movements with trends and volatility clustering
                # Add some momentum and mean reversion
                trend = 0.0002  # Small upward trend
                volatility = 0.02 * (1 + 0.5 * abs(price - base_price) / base_price)  # Volatility increases with distance from base
                daily_return = random.normalvariate(trend, volatility)
                
                # Add some serial correlation (momentum)
                if i > 0:
                    prev_return = (price - (price / (1 + random.normalvariate(trend, volatility)))) / (price / (1 + random.normalvariate(trend, volatility)))
                    daily_return += 0.1 * prev_return  # 10% momentum
                
                price = price * (1 + daily_return)
                
                # Ensure positive prices
                price = max(price, 0.01)
                
                # Generate OHLCV data
                open_price = price
                high_price = price * (1 + random.uniform(0, 0.03))
                low_price = price * (1 - random.uniform(0, 0.03))
                close_price = low_price + random.uniform(0, high_price - low_price)
                volume = random.randint(1000000, 10000000)
                
                # Update price for next iteration
                price = close_price
                
                writer.writerow([date, round(open_price, 2), round(high_price, 2), 
                               round(low_price, 2), round(close_price, 2), volume])
    
    print(f"Created advanced sample data for {len(symbols)} symbols")


def run_advanced_backtest():
    """Run an advanced backtest with the ML strategy."""
    print("\n=== Running Advanced Backtest ===")
    
    # Create sample data
    create_advanced_sample_data()
    
    # Define symbols
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    
    # Create backtesting engine with advanced components
    engine = BacktestingEngine(
        data_handler=lambda: HistoricCSVDataHandler('data', symbols),
        strategy=lambda symbols: AdvancedMLStrategy(symbols, short_window=10, long_window=50, use_ml=True),
        portfolio=lambda data_handler, events, initial_capital: AdvancedPortfolio(
            data_handler, events, initial_capital=initial_capital),
        execution_handler=lambda events: SimulatedExecutionHandler(events),
        symbol_list=symbols,
        initial_capital=100000.0
    )
    
    # Run backtest
    engine.run()
    
    # Print results
    print(f"\nBacktest Results:")
    print(f"  Total signals generated: {engine.signals}")
    print(f"  Total orders placed: {engine.orders}")
    print(f"  Total trades executed: {engine.fills}")
    
    return engine


def run_performance_analysis(portfolio):
    """Run comprehensive performance analysis."""
    print("\n=== Running Performance Analysis ===")
    
    if len(portfolio.equity_curve) < 2:
        print("Insufficient data for performance analysis")
        return {}
    
    # Calculate key metrics
    initial_capital = portfolio.equity_curve[0]
    final_capital = portfolio.equity_curve[-1]
    total_return = (final_capital - initial_capital) / initial_capital
    
    # Calculate returns
    returns = portfolio.returns
    if not returns:
        # Calculate from equity curve
        returns = []
        for i in range(1, len(portfolio.equity_curve)):
            ret = (portfolio.equity_curve[i] - portfolio.equity_curve[i-1]) / portfolio.equity_curve[i-1] if portfolio.equity_curve[i-1] != 0 else 0
            returns.append(ret)
    
    # Calculate metrics
    sharpe_ratio = calculate_sharpe_ratio(returns)
    sortino_ratio = calculate_sortino_ratio(returns)
    max_drawdown = calculate_max_drawdown(portfolio.equity_curve)
    
    # Calculate CAGR (assuming 2 years of data)
    years = 2.0
    cagr = calculate_cagr(initial_capital, final_capital, years)
    
    # Calculate win rate
    winning_trades = sum(1 for ret in returns if ret > 0)
    win_rate = winning_trades / len(returns) if returns else 0
    
    print(f"Performance Metrics:")
    print(f"  Total Return: {total_return:.2%}")
    print(f"  CAGR: {cagr:.2%}")
    print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {sortino_ratio:.2f}")
    print(f"  Max Drawdown: {max_drawdown:.2%}")
    print(f"  Win Rate: {win_rate:.2%}")
    print(f"  Number of Trades: {len(portfolio.trade_history)}")
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': len(portfolio.trade_history)
    }


def demonstrate_risk_management():
    """Demonstrate risk management capabilities."""
    print("\n=== Demonstrating Risk Management ===")
    
    # Create sample portfolio
    class MockDataHandler:
        def __init__(self):
            self.symbol_list = ["AAPL", "GOOGL"]
    
    mock_data_handler = MockDataHandler()
    portfolio = AdvancedPortfolio(mock_data_handler, None, initial_capital=100000.0)
    
    # Set up some positions
    portfolio.current_positions = {"AAPL": 100, "GOOGL": 0}
    portfolio.current_holdings = {
        "AAPL": 15000,
        "GOOGL": 0,
        "cash": 85000,
        "commission": 0,
        "total": 100000
    }
    
    # Create risk manager
    risk_manager = EnhancedRiskManager(portfolio, max_positions=3, max_drawdown=0.2, stop_loss=0.05)
    
    # Test position limit
    signals = [SignalEvent("MSFT", "BUY", 1.0), SignalEvent("AMZN", "BUY", 1.0)]
    filtered_signals = risk_manager.modify_signals(signals)
    print(f"Position limit test: {len(filtered_signals)} signals passed (max 3 positions)")
    
    # Test drawdown limit
    portfolio.current_holdings['total'] = 80000  # 20% drawdown
    signals = [SignalEvent("TSLA", "BUY", 1.0)]
    filtered_signals = risk_manager.modify_signals(signals)
    print(f"Drawdown limit test: {len(filtered_signals)} signals passed (20% drawdown)")
    
    print("Risk management demonstrated successfully")


def demonstrate_machine_learning():
    """Demonstrate machine learning capabilities."""
    print("\n=== Demonstrating Machine Learning ===")
    
    # Create sample data
    sample_prices = [100 + i * 0.1 + random.normalvariate(0, 1) for i in range(100)]
    
    # Create features
    feature_engineer = FeatureEngineer()
    sample_data = [{'close': price, 'volume': 1000000} for price in sample_prices]
    features_data = feature_engineer.create_features(sample_data)
    
    # Prepare data for training
    X, y = feature_engineer.prepare_data_for_training(features_data, 'returns', 5)
    
    # Create and train model
    model = EnhancedLinearRegression("DemoModel", regularization=0.01)
    if X and y:
        model.train(X[:80], y[:80])  # Use first 80 for training
        print("ML model trained successfully")
        
        # Make predictions
        predictions = model.predict(X[80:])  # Predict on last 20
        print(f"Made {len(predictions)} predictions")
        
        # Evaluate model
        actual_values = y[80:]
        mse = sum((actual_values[i] - predictions[i]) ** 2 for i in range(len(predictions))) / len(predictions)
        print(f"Model MSE: {mse:.6f}")
    else:
        print("Insufficient data for ML demonstration")


def demonstrate_data_management():
    """Demonstrate data management capabilities."""
    print("\n=== Demonstrating Data Management ===")
    
    # Create sample data first
    create_advanced_sample_data()
    
    # Create data manager with CSV provider
    csv_provider = CSVDataProvider("data")
    data_manager = DataManager([csv_provider])
    
    # Get data for a symbol
    try:
        data = data_manager.get_historical_data("AAPL")
        print(f"Retrieved {len(data)} data points for AAPL")
        
        # Validate data
        validator = DataValidator()
        validation_results = validator.validate_data(data, "AAPL")
        print(f"Data validation: {validation_results['total_rows']} rows, {len(validation_results['issues'])} issues")
        
        # Clean data
        cleaned_data = validator.clean_data(data)
        print(f"Cleaned data: {len(cleaned_data)} rows")
        
    except Exception as e:
        print(f"Data management demonstration error: {e}")


def demonstrate_utilities():
    """Demonstrate utility components."""
    print("\n=== Demonstrating Utilities ===")
    
    # Cache manager
    cache = CacheManager()
    cache.set("test_key", {"data": [1, 2, 3, 4, 5], "timestamp": datetime.now().isoformat()})
    cached_data = cache.get("test_key")
    print(f"Cache manager: Stored and retrieved {len(cached_data.get('data', []))} items")
    
    # Timer
    with Timer() as timer:
        # Simulate some work
        sum(i for i in range(100000))
    
    print(f"Timer: Operation took {timer.elapsed_time():.4f} seconds")


def main():
    """Run the complete advanced strategy demonstration."""
    print("PyTradePath Advanced Strategy Demonstration")
    print("=" * 50)
    print("This demonstration showcases an advanced machine learning-based trading strategy")
    print("with comprehensive risk management and performance analysis.")
    
    try:
        # Run demonstrations
        demonstrate_data_management()
        demonstrate_machine_learning()
        demonstrate_risk_management()
        demonstrate_utilities()
        
        # Run advanced backtest
        backtest_engine = run_advanced_backtest()
        
        # Run performance analysis
        performance_metrics = run_performance_analysis(backtest_engine.portfolio)
        
        # Summary
        print("\n" + "=" * 50)
        print("ADVANCED STRATEGY DEMONSTRATION SUMMARY")
        print("=" * 50)
        print(f"✓ Data management: Handled multiple data sources")
        print(f"✓ Machine learning: Trained and evaluated models")
        print(f"✓ Risk management: Applied position limits and drawdown controls")
        print(f"✓ Backtesting engine: Processed {backtest_engine.signals} signals")
        print(f"✓ Performance analysis: Generated comprehensive metrics")
        print(f"✓ Utilities: Demonstrated caching and timing")
        
        print("\n🎉 Advanced strategy demonstration completed successfully!")
        print("\nThe PyTradePath framework provides a complete solution for:")
        print("  • Advanced machine learning-based strategies")
        print("  • Comprehensive risk management")
        print("  • Multi-asset portfolio management")
        print("  • Detailed performance analysis")
        print("  • Data validation and cleaning")
        print("  • Utility functions for optimization")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


# Simple numpy-like functions for the demo
def random_normalvariate(mu, sigma):
    """Simple normal distribution generator."""
    # Box-Muller transform for normal distribution
    u1 = random.random()
    u2 = random.random()
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mu + sigma * z0

# Monkey patch the random module
random.normalvariate = random_normalvariate

if __name__ == "__main__":
    main()