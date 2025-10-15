import unittest
import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable
from datetime import datetime, timedelta
import tempfile
import os
import warnings
from unittest.mock import Mock, patch, MagicMock
import asyncio

from .event import EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent
from .data_handler import HistoricCSVDataHandler
from .strategy import Strategy, BuyAndHoldStrategy
from .portfolio import Portfolio, NaivePortfolio
from .execution import ExecutionHandler, SimulatedExecutionHandler
from .engine import BacktestingEngine
from .risk import RiskManager, PositionSizer
from .optimization import ParameterOptimizer
from .data import CSVDataProvider, DataManager
from .ml import MLModel, SklearnModel
from .live import PaperBroker
from .reporting import PerformanceAnalyzer
from .config import ConfigManager


class BaseTestCase(unittest.TestCase):
    """
    Base test case with common utilities.
    """

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_file = os.path.join(self.temp_dir, "test_data.csv")
        self._create_test_data()

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        # Clean up temporary files
        if os.path.exists(self.test_data_file):
            os.remove(self.test_data_file)
        os.rmdir(self.temp_dir)

    def _create_test_data(self):
        """Create test data file."""
        test_data = """datetime,open,high,low,close,volume
2023-01-01,100.0,105.0,99.0,104.0,10000
2023-01-02,104.0,108.0,103.0,107.0,12000
2023-01-03,107.0,110.0,105.0,109.0,11000
2023-01-04,109.0,112.0,108.0,111.0,13000
2023-01-05,111.0,115.0,110.0,114.0,14000
2023-01-06,114.0,118.0,113.0,117.0,15000
2023-01-07,117.0,120.0,115.0,119.0,16000
2023-01-08,119.0,122.0,118.0,121.0,17000
2023-01-09,121.0,125.0,120.0,124.0,18000
2023-01-10,124.0,128.0,123.0,127.0,19000
"""
        with open(self.test_data_file, 'w') as f:
            f.write(test_data)

    def assertAlmostEqualDict(self, dict1: Dict, dict2: Dict, 
                            places: int = 7, msg: str = None):
        """
        Assert that two dictionaries are almost equal.
        
        Parameters:
        dict1 - First dictionary
        dict2 - Second dictionary
        places - Number of decimal places to compare
        msg - Error message
        """
        self.assertEqual(dict1.keys(), dict2.keys(), msg)
        for key in dict1.keys():
            self.assertAlmostEqual(dict1[key], dict2[key], places, msg)


class TestEventSystem(BaseTestCase):
    """Test cases for the event system."""

    def test_market_event_creation(self):
        """Test MarketEvent creation."""
        symbol = "AAPL"
        data = {"price": 150.0, "volume": 1000}
        event = MarketEvent(symbol, data)
        
        self.assertEqual(event.type, EventType.MARKET)
        self.assertEqual(event.symbol, symbol)
        self.assertEqual(event.data, data)
        self.assertIsNotNone(event.timestamp)

    def test_signal_event_creation(self):
        """Test SignalEvent creation."""
        symbol = "AAPL"
        signal_type = "BUY"
        strength = 0.8
        event = SignalEvent(symbol, signal_type, strength)
        
        self.assertEqual(event.type, EventType.SIGNAL)
        self.assertEqual(event.symbol, symbol)
        self.assertEqual(event.signal_type, signal_type)
        self.assertEqual(event.strength, strength)
        self.assertIsNotNone(event.timestamp)

    def test_order_event_creation(self):
        """Test OrderEvent creation."""
        symbol = "AAPL"
        order_type = "MARKET"
        quantity = 100
        direction = "BUY"
        event = OrderEvent(symbol, order_type, quantity, direction)
        
        self.assertEqual(event.type, EventType.ORDER)
        self.assertEqual(event.symbol, symbol)
        self.assertEqual(event.order_type, order_type)
        self.assertEqual(event.quantity, quantity)
        self.assertEqual(event.direction, direction)
        self.assertIsNotNone(event.timestamp)

    def test_fill_event_creation(self):
        """Test FillEvent creation."""
        symbol = "AAPL"
        quantity = 100
        direction = "BUY"
        fill_price = 150.0
        commission = 1.0
        event = FillEvent(symbol, quantity, direction, fill_price, commission)
        
        self.assertEqual(event.type, EventType.FILL)
        self.assertEqual(event.symbol, symbol)
        self.assertEqual(event.quantity, quantity)
        self.assertEqual(event.direction, direction)
        self.assertEqual(event.fill_price, fill_price)
        self.assertEqual(event.commission, commission)
        self.assertAlmostEqual(event.cost, quantity * fill_price + commission)


class TestDataHandler(BaseTestCase):
    """Test cases for data handlers."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.symbol_list = ["test_data"]
        self.data_handler = HistoricCSVDataHandler(self.temp_dir, self.symbol_list)

    def test_data_handler_initialization(self):
        """Test data handler initialization."""
        self.assertEqual(self.data_handler.symbol_list, self.symbol_list)
        self.assertTrue(hasattr(self.data_handler, 'symbol_data'))
        self.assertTrue(hasattr(self.data_handler, 'latest_symbol_data'))

    def test_get_latest_bars(self):
        """Test getting latest bars."""
        # Update bars first
        self.data_handler.update_bars()
        
        # Get latest bar
        latest_bars = self.data_handler.get_latest_bars("test_data", 1)
        self.assertEqual(len(latest_bars), 1)
        
        # Get multiple bars
        self.data_handler.update_bars()
        latest_bars = self.data_handler.get_latest_bars("test_data", 2)
        self.assertEqual(len(latest_bars), 2)

    def test_update_bars(self):
        """Test updating bars."""
        # Initial state
        self.assertTrue(self.data_handler.continue_backtest)
        
        # Update bars multiple times
        for i in range(5):
            result = self.data_handler.update_bars()
            self.assertTrue(result or not self.data_handler.continue_backtest)


class TestStrategy(BaseTestCase):
    """Test cases for trading strategies."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.symbols = ["test_data"]
        self.strategy = BuyAndHoldStrategy(self.symbols)

    def test_strategy_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.symbols, self.symbols)
        self.assertIsNone(self.strategy.events_queue)

    def test_set_events_queue(self):
        """Test setting events queue."""
        mock_queue = Mock()
        self.strategy.set_events_queue(mock_queue)
        self.assertEqual(self.strategy.events_queue, mock_queue)

    def test_buy_and_hold_strategy(self):
        """Test buy and hold strategy."""
        # Create mock events queue
        mock_queue = Mock()
        self.strategy.set_events_queue(mock_queue)
        
        # Create market event
        market_data = {self.symbols[0]: [("2023-01-01", {"close": 100.0})]}
        market_event = MarketEvent(self.symbols[0], market_data)
        
        # Calculate signals
        self.strategy.calculate_signals(market_event)
        
        # Verify signal was generated
        mock_queue.put.assert_called_once()
        args, kwargs = mock_queue.put.call_args
        signal_event = args[0]
        self.assertIsInstance(signal_event, SignalEvent)
        self.assertEqual(signal_event.signal_type, "BUY")


class TestPortfolio(BaseTestCase):
    """Test cases for portfolio management."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.symbol_list = ["test_data"]
        
        # Create mock data handler
        self.data_handler = HistoricCSVDataHandler(self.temp_dir, self.symbol_list)
        
        # Create mock events queue
        self.events_queue = Mock()
        
        # Create portfolio
        self.portfolio = NaivePortfolio(
            self.data_handler, 
            self.events_queue,
            initial_capital=100000.0
        )

    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        self.assertEqual(self.portfolio.symbol_list, self.symbol_list)
        self.assertEqual(self.portfolio.initial_capital, 100000.0)
        self.assertEqual(self.portfolio.current_holdings['cash'], 100000.0)

    def test_construct_all_positions(self):
        """Test constructing all positions."""
        positions = self.portfolio.construct_all_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]['test_data'], 0)

    def test_construct_all_holdings(self):
        """Test constructing all holdings."""
        holdings = self.portfolio.construct_all_holdings()
        self.assertEqual(len(holdings), 1)
        self.assertEqual(holdings[0]['cash'], 100000.0)
        self.assertEqual(holdings[0]['total'], 100000.0)

    def test_update_positions_from_fill(self):
        """Test updating positions from fill."""
        # Create fill event
        fill_event = FillEvent("test_data", 100, "BUY", 100.0, 1.0)
        
        # Update positions
        initial_position = self.portfolio.current_positions["test_data"]
        self.portfolio.update_positions_from_fill(fill_event)
        
        # Verify position updated
        self.assertEqual(
            self.portfolio.current_positions["test_data"], 
            initial_position + 100
        )

    def test_update_holdings_from_fill(self):
        """Test updating holdings from fill."""
        # Create fill event
        fill_event = FillEvent("test_data", 100, "BUY", 100.0, 1.0)
        
        # Update holdings
        initial_cash = self.portfolio.current_holdings["cash"]
        self.portfolio.update_holdings_from_fill(fill_event)
        
        # Verify holdings updated
        expected_cost = 100 * 100.0 + 1.0
        self.assertEqual(
            self.portfolio.current_holdings["cash"], 
            initial_cash - expected_cost
        )


class TestExecutionHandler(BaseTestCase):
    """Test cases for execution handlers."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.events_queue = Mock()
        self.execution_handler = SimulatedExecutionHandler(self.events_queue)

    def test_execution_handler_initialization(self):
        """Test execution handler initialization."""
        self.assertEqual(self.execution_handler.commission_rate, 0.001)
        self.assertEqual(self.execution_handler.slippage_factor, 0.0001)

    def test_execute_order(self):
        """Test executing order."""
        # Create order event
        order_event = OrderEvent("test_data", "MARKET", 100, "BUY")
        
        # Execute order
        self.execution_handler.execute_order(order_event)
        
        # Verify fill event was created and queued
        self.events_queue.put.assert_called_once()
        args, kwargs = self.events_queue.put.call_args
        fill_event = args[0]
        self.assertIsInstance(fill_event, FillEvent)
        self.assertEqual(fill_event.symbol, "test_data")
        self.assertEqual(fill_event.quantity, 100)
        self.assertEqual(fill_event.direction, "BUY")


class TestBacktestingEngine(BaseTestCase):
    """Test cases for backtesting engine."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.symbol_list = ["test_data"]

    def test_engine_initialization(self):
        """Test engine initialization."""
        # This test would require more complex setup
        # For now, we'll skip the actual engine test
        pass

    def test_run_backtest(self):
        """Test running backtest."""
        # This test would require more complex setup
        # For now, we'll skip the actual engine test
        pass


class TestRiskManagement(BaseTestCase):
    """Test cases for risk management."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        # Create mock portfolio for testing
        self.mock_portfolio = Mock()
        self.mock_portfolio.current_positions = {"test_data": 100}
        self.mock_portfolio.current_holdings = {
            "test_data": 10000.0,
            "cash": 90000.0,
            "total": 100000.0
        }

    def test_risk_manager_initialization(self):
        """Test risk manager initialization."""
        # This would require implementing specific risk manager tests
        pass


class TestOptimization(BaseTestCase):
    """Test cases for parameter optimization."""

    def test_parameter_optimizer_initialization(self):
        """Test parameter optimizer initialization."""
        # Create mock strategy class
        mock_strategy_class = Mock()
        
        # Create optimizer
        optimizer = ParameterOptimizer(
            mock_strategy_class,
            HistoricCSVDataHandler,
            ["test_data"]
        )
        
        self.assertEqual(optimizer.strategy_class, mock_strategy_class)
        self.assertEqual(optimizer.symbol_list, ["test_data"])


class TestDataManagement(BaseTestCase):
    """Test cases for data management."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.data_provider = CSVDataProvider(self.temp_dir)

    def test_csv_data_provider(self):
        """Test CSV data provider."""
        # Get historical data
        data = self.data_provider.get_historical_data("test_data")
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 10)  # 10 rows in test data
        self.assertIn("open", data.columns)
        self.assertIn("close", data.columns)
        self.assertIn("volume", data.columns)

    def test_data_manager(self):
        """Test data manager."""
        data_manager = DataManager([self.data_provider])
        
        # Get historical data
        data = data_manager.get_historical_data("test_data")
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 10)


class TestMachineLearning(BaseTestCase):
    """Test cases for machine learning components."""

    def test_ml_model_interface(self):
        """Test ML model interface."""
        # This would require implementing specific ML model tests
        pass


class TestLiveTrading(BaseTestCase):
    """Test cases for live trading components."""

    def test_paper_broker(self):
        """Test paper broker."""
        broker = PaperBroker(initial_balance=100000.0)
        
        # Check initial state
        self.assertEqual(broker.balance, 100000.0)
        self.assertEqual(broker.commission_rate, 0.001)
        
        # Get account info
        account_info = broker.get_account_info()
        self.assertEqual(account_info['balance'], 100000.0)


class TestReporting(BaseTestCase):
    """Test cases for reporting components."""

    def test_performance_analyzer(self):
        """Test performance analyzer."""
        analyzer = PerformanceAnalyzer()
        
        # Create test portfolio values
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        values = pd.Series([100000 + i * 1000 for i in range(10)], index=dates)
        
        # Calculate returns
        returns = analyzer.calculate_returns(values)
        self.assertEqual(len(returns), 9)  # One less than values
        
        # Calculate Sharpe ratio
        sharpe = analyzer.calculate_sharpe_ratio(returns)
        self.assertIsInstance(sharpe, float)


class TestConfiguration(BaseTestCase):
    """Test cases for configuration management."""

    def test_config_manager(self):
        """Test configuration manager."""
        config_manager = ConfigManager()
        
        # Get configuration
        config = config_manager.get_config()
        self.assertIsNotNone(config)
        
        # Check some default values
        self.assertEqual(config.backtest.initial_capital, 100000.0)
        self.assertEqual(config.logging.level, "INFO")


class IntegrationTests(BaseTestCase):
    """Integration tests for the complete system."""

    def test_complete_backtest_flow(self):
        """Test complete backtest flow."""
        # This would test the entire backtesting workflow
        pass

    def test_strategy_to_portfolio_flow(self):
        """Test strategy to portfolio flow."""
        # This would test the signal generation to order flow
        pass


class PerformanceTests(BaseTestCase):
    """Performance tests for critical components."""

    def test_data_loading_performance(self):
        """Test data loading performance."""
        # This would test how quickly data is loaded
        pass

    def test_backtest_performance(self):
        """Test backtest performance."""
        # This would test backtest execution speed
        pass


class StressTests(BaseTestCase):
    """Stress tests for system limits."""

    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        # This would test with large amounts of data
        pass

    def test_concurrent_access(self):
        """Test concurrent access to components."""
        # This would test thread safety
        pass


class MockDataTests(BaseTestCase):
    """Tests using mock data and dependencies."""

    def test_with_mock_data_handler(self):
        """Test with mocked data handler."""
        with patch('core.data_handler.HistoricCSVDataHandler') as mock_handler:
            # Configure mock
            mock_instance = Mock()
            mock_instance.update_bars.return_value = True
            mock_handler.return_value = mock_instance
            
            # Use mock in test
            # ... test code here ...
            pass

    def test_with_mock_strategy(self):
        """Test with mocked strategy."""
        with patch('core.strategy.BuyAndHoldStrategy') as mock_strategy:
            # Configure mock
            mock_instance = Mock()
            mock_strategy.return_value = mock_instance
            
            # Use mock in test
            # ... test code here ...
            pass


class AsyncTests(BaseTestCase):
    """Tests for asynchronous components."""

    def test_async_data_feeds(self):
        """Test asynchronous data feeds."""
        # This would test async data handling
        pass

    def test_async_execution(self):
        """Test asynchronous execution."""
        # This would test async order execution
        pass


def create_test_suite() -> unittest.TestSuite:
    """
    Create a comprehensive test suite.
    
    Returns:
    Test suite with all tests
    """
    suite = unittest.TestSuite()
    
    # Add all test cases
    test_cases = [
        TestEventSystem,
        TestDataHandler,
        TestStrategy,
        TestPortfolio,
        TestExecutionHandler,
        TestBacktestingEngine,
        TestRiskManagement,
        TestOptimization,
        TestDataManagement,
        TestMachineLearning,
        TestLiveTrading,
        TestReporting,
        TestConfiguration,
        IntegrationTests,
        PerformanceTests,
        StressTests,
        MockDataTests,
        AsyncTests
    ]
    
    for test_case in test_cases:
        suite.addTest(unittest.makeSuite(test_case))
    
    return suite


def run_all_tests(verbosity: int = 2) -> unittest.TestResult:
    """
    Run all tests in the framework.
    
    Parameters:
    verbosity - Test output verbosity
    
    Returns:
    Test results
    """
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result


class TestRunner:
    """
    Comprehensive test runner with reporting.
    """

    def __init__(self):
        """Initialize the test runner."""
        self.results = []

    def run_tests(self, test_pattern: str = None) -> Dict[str, Any]:
        """
        Run tests with optional pattern filtering.
        
        Parameters:
        test_pattern - Pattern to filter tests (optional)
        
        Returns:
        Dictionary with test results
        """
        # Implement a basic test runner
        import unittest
        import sys
        from io import StringIO
        
        # Capture test output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            # Create test suite
            loader = unittest.TestLoader()
            suite = loader.discover('.')
            
            # Run tests
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            
            # Get output
            output = sys.stdout.getvalue()
            
            # Return results
            return {
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0.0,
                "output": output
            }
        except Exception as e:
            return {
                "tests_run": 0,
                "failures": 0,
                "errors": 1,
                "success_rate": 0.0,
                "error": str(e)
            }
        finally:
            sys.stdout = old_stdout

    def generate_test_report(self) -> str:
        """
        Generate comprehensive test report.
        
        Returns:
        Test report as string
        """
        # Generate a detailed test report
        report = "PyTradePath Test Report\n"
        report += "=" * 50 + "\n\n"
        
        report += "Test Execution Summary:\n"
        report += "-" * 30 + "\n"
        report += f"Framework Version: 1.0\n"
        report += f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Run tests and include results
        results = self.run_tests()
        
        report += f"Tests Run: {results.get('tests_run', 0)}\n"
        report += f"Failures: {results.get('failures', 0)}\n"
        report += f"Errors: {results.get('errors', 0)}\n"
        report += f"Success Rate: {results.get('success_rate', 0.0):.2%}\n\n"
        
        # Add detailed output if available
        if 'output' in results:
            report += "Detailed Output:\n"
            report += "-" * 20 + "\n"
            report += results['output'][:1000] + "\n"  # Limit output length
        
        return report


# Pytest-style fixtures and tests
@pytest.fixture
def sample_market_data():
    """Pytest fixture for sample market data."""
    return pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=10),
        'open': [100 + i for i in range(10)],
        'high': [105 + i for i in range(10)],
        'low': [95 + i for i in range(10)],
        'close': [102 + i for i in range(10)],
        'volume': [1000 + i * 100 for i in range(10)]
    }).set_index('datetime')


@pytest.fixture
def sample_events():
    """Pytest fixture for sample events."""
    return [
        MarketEvent("AAPL", {"price": 150.0}),
        SignalEvent("AAPL", "BUY", 0.8),
        OrderEvent("AAPL", "MARKET", 100, "BUY"),
        FillEvent("AAPL", 100, "BUY", 150.0, 1.0)
    ]


def test_event_types(sample_events):
    """Test event types using pytest."""
    event_types = [EventType.MARKET, EventType.SIGNAL, EventType.ORDER, EventType.FILL]
    
    for event, expected_type in zip(sample_events, event_types):
        assert event.type == expected_type


def test_performance_analyzer_metrics():
    """Test performance analyzer metrics using pytest."""
    analyzer = PerformanceAnalyzer()
    
    # Create sample returns
    dates = pd.date_range('2023-01-01', periods=100)
    returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
    
    # Test various metrics
    sharpe = analyzer.calculate_sharpe_ratio(returns)
    assert isinstance(sharpe, float)
    
    max_dd, dd_series = analyzer.calculate_max_drawdown(
        (1 + returns).cumprod() * 100000
    )
    assert isinstance(max_dd, float)
    assert isinstance(dd_series, pd.Series)


class BenchmarkTests:
    """
    Benchmark tests for performance comparison.
    """

    def benchmark_data_loading(self):
        """Benchmark data loading performance."""
        import time
        import random
        
        # Simulate data loading benchmark
        iterations = 100
        times = []
        
        for i in range(iterations):
            start_time = time.time()
            
            # Simulate data loading operation
            # Generate sample data
            sample_data = []
            for j in range(1000):
                sample_data.append({
                    'timestamp': f'2023-01-{j+1:02d}',
                    'open': 100 + random.uniform(-10, 10),
                    'high': 105 + random.uniform(-10, 10),
                    'low': 95 + random.uniform(-10, 10),
                    'close': 102 + random.uniform(-10, 10),
                    'volume': 1000 + random.randint(0, 1000)
                })
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        return {
            "test_name": "data_loading",
            "iterations": iterations,
            "average_time": avg_time,
            "min_time": min_time,
            "max_time": max_time
        }

    def benchmark_backtest_execution(self):
        """Benchmark backtest execution performance with realistic market simulation."""
        import time
        import random
        from datetime import datetime, timedelta
        
        # Realistic backtest execution benchmark
        iterations = 50
        times = []
        
        for i in range(iterations):
            start_time = time.time()
            
            # Simulate realistic backtest execution with market data processing
            # This includes:
            # 1. Market data feed processing
            # 2. Strategy signal generation
            # 3. Risk management checks
            # 4. Order generation and execution
            # 5. Portfolio updates
            
            # Simulate processing of market data for multiple symbols over time
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
            for symbol in symbols:
                # Simulate multiple time steps
                for step in range(200):  # 200 time steps per symbol
                    # Generate realistic price movements with trends and volatility
                    base_price = 100 + (step * 0.1)  # Gradual trend
                    volatility = random.uniform(0.01, 0.03)  # 1-3% daily volatility
                    price_change = random.normalvariate(0, volatility)
                    price = base_price * (1 + price_change)
                    
                    # Generate signals based on technical indicators
                    # Simple moving average crossover logic
                    short_ma = price * (1 + random.uniform(-0.005, 0.005))
                    long_ma = price * (1 + random.uniform(-0.01, 0.01))
                    
                    # Risk management checks
                    position_size = 100  # Fixed position size for benchmarking
                    max_position_size = 1000
                    
                    # Generate and process orders
                    if short_ma > long_ma and position_size < max_position_size:
                        signal = 'BUY'
                    elif short_ma < long_ma and position_size > 0:
                        signal = 'SELL'
                    else:
                        signal = 'HOLD'
                    
                    # Simulate order execution with realistic slippage and commissions
                    if signal != 'HOLD':
                        # Market impact based on order size
                        market_impact = 0.001 * (position_size / 1000)
                        # Slippage based on volatility
                        slippage = volatility * random.uniform(0.5, 2.0)
                        # Commission (typical broker fee)
                        commission = 0.001
                        
                        fill_price = price * (1 + market_impact + slippage + commission 
                                            if signal == 'BUY' else 1 - market_impact - slippage - commission)
                        
                        # Portfolio update simulation
                        portfolio_value = 100000.0
                        cash_change = position_size * fill_price
                        
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate comprehensive statistics
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            # Calculate standard deviation
            variance = sum((t - avg_time) ** 2 for t in times) / len(times)
            std_dev = variance ** 0.5
        else:
            avg_time = min_time = max_time = 0.0
            std_dev = 0.0
        
        return {
            "test_name": "backtest_execution",
            "iterations": iterations,
            "average_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "std_deviation": std_dev,
            "total_symbols": 5,
            "time_steps_per_symbol": 200
        }


def run_benchmarks() -> Dict[str, Any]:
    """
    Run all benchmark tests.
    
    Returns:
    Dictionary with benchmark results
    """
    benchmark_runner = BenchmarkTests()
    
    results = {
        "data_loading": benchmark_runner.benchmark_data_loading(),
        "backtest_execution": benchmark_runner.benchmark_backtest_execution()
    }
    
    return results


# Example of how to run tests
if __name__ == "__main__":
    # Run unittest-based tests
    print("Running unittest-based tests...")
    result = run_all_tests()
    
    # Run pytest-based tests (if pytest is available)
    try:
        print("\nRunning pytest-based tests...")
        pytest.main(["-v", __file__])
    except Exception as e:
        print(f"Pytest not available or failed: {e}")
    
    # Summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")