"""
Live trading module for the pytradepath framework.
This module provides capabilities for live trading and paper trading.
"""

import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from abc import ABC, abstractmethod
from .event import MarketEvent, SignalEvent, OrderEvent, FillEvent
from .data_handler import DataHandler
from .strategy import Strategy
from .portfolio import Portfolio
from .execution import ExecutionHandler, SimulatedExecutionHandler


class LiveDataFeed(ABC):
    """
    Abstract base class for live data feeds.
    """

    @abstractmethod
    def get_latest_data(self) -> Dict[str, Dict]:
        """
        Get the latest market data.
        
        Returns:
        Dictionary with latest data by symbol
        """
        raise NotImplementedError("Should implement get_latest_data()")

    @abstractmethod
    def subscribe(self, symbols: List[str]):
        """
        Subscribe to symbols.
        
        Parameters:
        symbols - List of symbols to subscribe to
        """
        raise NotImplementedError("Should implement subscribe()")


class MockDataFeed(LiveDataFeed):
    """
    Mock data feed for demonstration purposes.
    """

    def __init__(self, data_directory: str = "data"):
        """
        Initialize the mock data feed.
        
        Parameters:
        data_directory - Directory containing historical data
        """
        self.data_directory = data_directory
        self.subscribed_symbols = []
        self.data_position = {}  # Track position in data for each symbol
        self.data_cache = {}  # Cache loaded data

    def subscribe(self, symbols: List[str]):
        """
        Subscribe to symbols.
        
        Parameters:
        symbols - List of symbols to subscribe to
        """
        self.subscribed_symbols.extend(symbols)
        
        # Load data for subscribed symbols
        import csv
        import os
        
        for symbol in symbols:
            file_path = os.path.join(self.data_directory, f"{symbol}.csv")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    reader = csv.DictReader(f)
                    self.data_cache[symbol] = list(reader)
                    self.data_position[symbol] = 0
            else:
                # Create mock data if file doesn't exist
                self.data_cache[symbol] = self._generate_mock_data(symbol)
                self.data_position[symbol] = 0

    def _generate_mock_data(self, symbol: str) -> List[Dict]:
        """
        Generate mock data for a symbol.
        
        Parameters:
        symbol - Symbol to generate data for
        
        Returns:
        List of mock data rows
        """
        data = []
        base_price = 100 + random.uniform(-20, 20)
        price = base_price
        
        for i in range(1000):  # Generate 1000 data points
            # Simulate price movement
            daily_return = random.normalvariate(0.0005, 0.02)  # 0.05% mean, 2% std
            price = price * (1 + daily_return)
            price = max(price, 0.01)  # Ensure positive price
            
            # Generate OHLCV data
            open_price = price
            high_price = price * (1 + random.uniform(0, 0.03))
            low_price = price * (1 - random.uniform(0, 0.03))
            close_price = low_price + random.uniform(0, high_price - low_price)
            volume = random.randint(100000, 1000000)
            
            # Update price for next iteration
            price = close_price
            
            data.append({
                'datetime': (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - 
                           timedelta(days=1000-i)).strftime('%Y-%m-%d'),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
        
        return data

    def get_latest_data(self) -> Dict[str, Dict]:
        """
        Get the latest market data.
        
        Returns:
        Dictionary with latest data by symbol
        """
        latest_data = {}
        
        for symbol in self.subscribed_symbols:
            if symbol in self.data_cache and symbol in self.data_position:
                data = self.data_cache[symbol]
                position = self.data_position[symbol]
                
                if position < len(data):
                    latest_data[symbol] = data[position]
                    self.data_position[symbol] += 1
                else:
                    # Reset to beginning if we've reached the end
                    self.data_position[symbol] = 0
                    if data:
                        latest_data[symbol] = data[0]
        
        return latest_data


class PaperBroker:
    """
    Paper trading broker that simulates live trading.
    """

    def __init__(self, initial_capital: float = 100000.0, 
                 commission_rate: float = 0.001):
        """
        Initialize the paper broker.
        
        Parameters:
        initial_capital - Initial capital
        commission_rate - Commission rate per trade
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # symbol -> quantity
        self.commission_rate = commission_rate
        self.trade_history = []
        self.orders = []

    def place_order(self, symbol: str, quantity: float, direction: str, 
                   order_type: str = 'MARKET') -> Dict:
        """
        Place an order.
        
        Parameters:
        symbol - Symbol to trade
        quantity - Quantity to trade
        direction - Direction ('BUY' or 'SELL')
        order_type - Order type ('MARKET', 'LIMIT', etc.)
        
        Returns:
        Order confirmation dictionary
        """
        order_id = f"ORDER_{len(self.orders) + 1}"
        timestamp = datetime.now().isoformat()
        
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'quantity': quantity,
            'direction': direction,
            'order_type': order_type,
            'timestamp': timestamp,
            'status': 'PENDING'
        }
        
        self.orders.append(order)
        return order

    def execute_order(self, order: Dict, fill_price: float) -> Dict:
        """
        Execute an order.
        
        Parameters:
        order - Order to execute
        fill_price - Price to fill at
        
        Returns:
        Fill confirmation dictionary
        """
        # Calculate commission
        commission = self.commission_rate * order['quantity'] * fill_price
        
        # Calculate cost
        cost = order['quantity'] * fill_price
        if order['direction'] == 'BUY':
            total_cost = cost + commission
            if self.current_capital >= total_cost:
                self.current_capital -= total_cost
                self.positions[order['symbol']] = self.positions.get(order['symbol'], 0) + order['quantity']
                status = 'FILLED'
            else:
                status = 'REJECTED'
        else:  # SELL
            if self.positions.get(order['symbol'], 0) >= order['quantity']:
                self.current_capital += cost - commission
                self.positions[order['symbol']] = self.positions.get(order['symbol'], 0) - order['quantity']
                status = 'FILLED'
            else:
                status = 'REJECTED'
        
        # Update order status
        order['status'] = status
        order['fill_price'] = fill_price if status == 'FILLED' else None
        order['commission'] = commission if status == 'FILLED' else 0
        
        # Record trade if filled
        if status == 'FILLED':
            trade = {
                'order_id': order['order_id'],
                'symbol': order['symbol'],
                'quantity': order['quantity'],
                'direction': order['direction'],
                'fill_price': fill_price,
                'commission': commission,
                'timestamp': datetime.now().isoformat()
            }
            self.trade_history.append(trade)
        
        return {
            'order_id': order['order_id'],
            'status': status,
            'fill_price': fill_price if status == 'FILLED' else None,
            'commission': commission if status == 'FILLED' else 0
        }

    def get_account_info(self) -> Dict:
        """
        Get account information.
        
        Returns:
        Account information dictionary
        """
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'positions': self.positions.copy(),
            'total_orders': len(self.orders),
            'total_trades': len(self.trade_history)
        }


class LiveTradingEngine:
    """
    Live trading engine that coordinates live trading components.
    """

    def __init__(self, data_feed: LiveDataFeed, strategy: Strategy, 
                 portfolio: Portfolio, execution_handler: ExecutionHandler,
                 symbols: List[str]):
        """
        Initialize the live trading engine.
        
        Parameters:
        data_feed - Live data feed
        strategy - Trading strategy
        portfolio - Portfolio manager
        execution_handler - Execution handler
        symbols - List of symbols to trade
        """
        self.data_feed = data_feed
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.symbols = symbols
        
        # Subscribe to symbols
        self.data_feed.subscribe(symbols)
        
        # Set up strategy events queue
        import queue
        self.events = queue.Queue()
        self.strategy.set_events_queue(self.events)

    def run(self, max_iterations: int = 100):
        """
        Run the live trading engine.
        
        Parameters:
        max_iterations - Maximum number of iterations to run
        """
        print("Starting live trading engine...")
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"Iteration {iteration}/{max_iterations}")
            
            # Get latest data
            latest_data = self.data_feed.get_latest_data()
            
            if latest_data:
                # Create market event
                market_event = MarketEvent("LIVE", latest_data)
                self.events.put(market_event)
                
                # Process events
                self._process_events()
            
            # Wait before next iteration
            time.sleep(1)  # Wait 1 second between iterations
        
        print("Live trading engine stopped.")

    def _process_events(self):
        """
        Process events in the queue.
        """
        import queue
        
        while True:
            try:
                event = self.events.get(False)
            except queue.Empty:
                break
            else:
                if event is not None:
                    if event.type.name == 'MARKET':
                        # Update portfolio and run strategy
                        self.portfolio.update_timeindex(event)
                        self.strategy.calculate_signals(event)
                    elif event.type.name == 'SIGNAL':
                        # Generate order from signal
                        order = self.portfolio.generate_naive_order(event)
                        if order:
                            self.events.put(order)
                    elif event.type.name == 'ORDER':
                        # Execute order
                        self.execution_handler.execute_order(event)


def create_paper_trading_session(strategy_class: Callable, symbols: List[str],
                               initial_capital: float = 100000.0) -> Dict:
    """
    Create a paper trading session.
    
    Parameters:
    strategy_class - Strategy class to use
    symbols - List of symbols to trade
    initial_capital - Initial capital
    
    Returns:
    Dictionary with trading session components
    """
    # Create components
    data_feed = MockDataFeed()
    strategy = strategy_class(symbols)
    broker = PaperBroker(initial_capital)
    
    # For simplicity, we'll use the simulated execution handler
    import queue
    events = queue.Queue()
    execution_handler = SimulatedExecutionHandler(events)
    
    # Create mock portfolio
    class MockPortfolio:
        def __init__(self):
            self.current_positions = {}
            self.current_holdings = {
                'cash': initial_capital,
                'commission': 0.0,
                'total': initial_capital
            }
        
        def update_timeindex(self, event):
            pass
            
        def generate_naive_order(self, signal_event):
            # Simplified order generation
            return OrderEvent(signal_event.symbol, 'MARKET', 100, signal_event.signal_type)
    
    portfolio = MockPortfolio()
    
    # Create trading engine
    engine = LiveTradingEngine(data_feed, strategy, portfolio, execution_handler, symbols)
    
    return {
        'data_feed': data_feed,
        'strategy': strategy,
        'portfolio': portfolio,
        'broker': broker,
        'execution_handler': execution_handler,
        'engine': engine
    }


# Example usage
if __name__ == "__main__":
    print("PyTradePath Live Trading Module")
    print("=" * 40)
    print("This module provides live trading capabilities for the framework.")
    print("It includes mock implementations for demonstration purposes.")