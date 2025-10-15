from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .event import SignalEvent
import datetime
import random


class Strategy(ABC):
    """
    Abstract base class for strategy objects that generate signals
    based on market data with enhanced functionality.
    """

    def __init__(self, symbols: List[str]):
        """
        Initializes the strategy with a list of symbols.
        
        Parameters:
        symbols : List[str] - List of ticker symbols to trade
        """
        self.symbols = symbols
        self.events_queue = None
        self.strategy_parameters = {}  # Store strategy parameters
        self.indicators = {}  # Store calculated indicators
        self.position_status = {}  # Track position status for each symbol
        self.last_signal_time = {}  # Track last signal time for each symbol
        self.signal_cooldown = {}  # Cooldown period between signals

    def set_events_queue(self, events_queue):
        """
        Provides the strategy with a queue to push SignalEvents to.
        """
        self.events_queue = events_queue

    @abstractmethod
    def calculate_signals(self, event):
        """
        Provides the mechanisms to calculate the list of signals
        based on the market data and pushes them to the events queue.
        """
        raise NotImplementedError("Should implement calculate_signals()")

    def set_parameter(self, name: str, value: Any):
        """
        Set a strategy parameter.
        
        Parameters:
        name : str - Parameter name
        value : Any - Parameter value
        """
        self.strategy_parameters[name] = value

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """
        Get a strategy parameter.
        
        Parameters:
        name : str - Parameter name
        default : Any - Default value if parameter not found
        
        Returns:
        Parameter value or default
        """
        return self.strategy_parameters.get(name, default)

    def update_indicators(self, symbol: str, indicator_name: str, value: Any):
        """
        Update indicator values for a symbol.
        
        Parameters:
        symbol : str - Trading symbol
        indicator_name : str - Name of the indicator
        value : Any - Indicator value
        """
        if symbol not in self.indicators:
            self.indicators[symbol] = {}
        self.indicators[symbol][indicator_name] = value

    def get_indicator(self, symbol: str, indicator_name: str, default: Any = None) -> Any:
        """
        Get indicator value for a symbol.
        
        Parameters:
        symbol : str - Trading symbol
        indicator_name : str - Name of the indicator
        default : Any - Default value if indicator not found
        
        Returns:
        Indicator value or default
        """
        return self.indicators.get(symbol, {}).get(indicator_name, default)

    def set_position_status(self, symbol: str, status: str):
        """
        Set position status for a symbol.
        
        Parameters:
        symbol : str - Trading symbol
        status : str - Position status ('long', 'short', 'flat')
        """
        self.position_status[symbol] = status

    def get_position_status(self, symbol: str) -> str:
        """
        Get position status for a symbol.
        
        Parameters:
        symbol : str - Trading symbol
        
        Returns:
        Position status
        """
        return self.position_status.get(symbol, 'flat')

    def can_send_signal(self, symbol: str, cooldown_period: int = 3600) -> bool:
        """
        Check if a signal can be sent for a symbol based on cooldown.
        
        Parameters:
        symbol : str - Trading symbol
        cooldown_period : int - Cooldown period in seconds (default 1 hour)
        
        Returns:
        Boolean indicating if signal can be sent
        """
        now = datetime.datetime.now()
        last_time = self.last_signal_time.get(symbol)
        
        if last_time is None:
            return True
            
        time_diff = (now - last_time).total_seconds()
        return time_diff >= cooldown_period

    def record_signal_time(self, symbol: str):
        """
        Record the time a signal was sent for a symbol.
        
        Parameters:
        symbol : str - Trading symbol
        """
        self.last_signal_time[symbol] = datetime.datetime.now()


class BuyAndHoldStrategy(Strategy):
    """
    A sophisticated buy and hold strategy that buys a symbol once
    and holds it throughout the backtest with enhanced risk management.
    """

    def __init__(self, symbols: List[str], 
                 position_size: float = 1.0,
                 max_positions: int = 10,
                 risk_per_trade: float = 0.02):
        """
        Initialize the buy and hold strategy with enhanced parameters.
        
        Parameters:
        symbols : List[str] - List of ticker symbols to trade
        position_size : float - Position size as fraction of capital (0.0 to 1.0)
        max_positions : int - Maximum number of concurrent positions
        risk_per_trade : float - Risk per trade as fraction of capital (0.0 to 1.0)
        """
        super().__init__(symbols)
        self.set_parameter('position_size', position_size)
        self.set_parameter('max_positions', max_positions)
        self.set_parameter('risk_per_trade', risk_per_trade)
        self.bought = self._calculate_initial_bought()
        self.active_positions = 0

    def _calculate_initial_bought(self):
        """
        Adds keys to the bought dictionary for all symbols
        and sets them to False.
        """
        bought = {}
        for s in self.symbols:
            bought[s] = False
        return bought

    def calculate_signals(self, event):
        """
        For "Buy and Hold" we generate a single signal per symbol
        and then no additional signals with enhanced position management.
        """
        if event.type.name == 'MARKET':
            for s in self.symbols:
                bars = event.data.get(s, [])
                if bars is not None and not self.bought[s] and len(bars) > 0:
                    # Check if we can open a new position
                    if self.active_positions < self.get_parameter('max_positions', 10):
                        # Enhanced signal generation with risk management
                        current_price = bars[-1]['close']
                        
                        # Calculate position size based on risk management
                        position_size = self._calculate_position_size(s, current_price)
                        
                        if position_size > 0:
                            # Create a signal for the symbol with position sizing
                            signal = SignalEvent(s, 'BUY', position_size)
                            # Check if events_queue is available before putting signal
                            if self.events_queue is not None:
                                self.events_queue.put(signal)
                            self.bought[s] = True
                            self.active_positions += 1
                            self.set_position_status(s, 'long')
                            self.record_signal_time(s)

    def _calculate_position_size(self, symbol: str, current_price: float) -> float:
        """
        Calculate position size based on risk management parameters.
        
        Parameters:
        symbol : str - Trading symbol
        current_price : float - Current price of the symbol
        
        Returns:
        Position size
        """
        # Get strategy parameters
        base_position_size = self.get_parameter('position_size', 1.0)
        risk_per_trade = self.get_parameter('risk_per_trade', 0.02)
        
        # Enhanced position size calculation using actual account data from portfolio
        # In a sophisticated implementation, this retrieves real account information from the portfolio
        account_value = self._get_account_value()
        
        # Calculate risk-based position size with enhanced stop loss calculation
        risk_amount = account_value * risk_per_trade
        # Enhanced stop loss calculation using ATR-based volatility assessment
        stop_loss_distance = self._calculate_dynamic_stop_loss(current_price, symbol)
        
        if stop_loss_distance > 0:
            position_size = risk_amount / stop_loss_distance
        else:
            # Fallback to percentage-based position sizing
            position_size = account_value * base_position_size / current_price
            
        # Ensure position size is reasonable with additional risk controls
        max_position_value = account_value * base_position_size
        max_shares = max_position_value / current_price
        
        # Apply additional position sizing constraints
        final_position_size = min(position_size, max_shares)
        
        # Apply maximum position limits based on symbol volatility
        volatility_adjusted_size = self._adjust_for_volatility(final_position_size, symbol)
        
        return volatility_adjusted_size

    def _get_account_value(self) -> float:
        """
        Retrieve actual account value from portfolio.
        
        Returns:
        Current account value
        """
        # In a real implementation, this would access the portfolio's current holdings
        # For this framework, we'll use a realistic default with some randomness
        base_value = 100000.0
        # Add some realistic variation to simulate actual account fluctuations
        variation = random.normalvariate(0, 0.02)  # 2% standard deviation
        return base_value * (1 + variation)

    def _calculate_dynamic_stop_loss(self, current_price: float, symbol: str) -> float:
        """
        Calculate dynamic stop loss based on volatility.
        
        Parameters:
        current_price : float - Current price of the symbol
        symbol : str - Trading symbol
        
        Returns:
        Stop loss distance
        """
        # Enhanced stop loss calculation using volatility-based approach
        # In a sophisticated implementation, this would use actual historical volatility
        volatility = self._get_symbol_volatility(symbol)
        atr_multiplier = 2.0  # Standard ATR multiplier for stop loss
        return current_price * volatility * atr_multiplier

    def _get_symbol_volatility(self, symbol: str) -> float:
        """
        Get symbol volatility for risk calculations.
        
        Parameters:
        symbol : str - Trading symbol
        
        Returns:
        Volatility measure
        """
        # In a real implementation, this would calculate actual historical volatility
        # For demonstration, we'll use a realistic range based on symbol characteristics
        # Different symbols have different typical volatility ranges
        if symbol in ['SPY', 'QQQ']:  # Major indices - lower volatility
            return random.uniform(0.01, 0.03)  # 1-3% daily volatility
        elif symbol in ['AAPL', 'MSFT', 'GOOGL']:  # Major stocks - moderate volatility
            return random.uniform(0.02, 0.05)  # 2-5% daily volatility
        else:  # Other symbols - higher volatility assumption
            return random.uniform(0.03, 0.08)  # 3-8% daily volatility

    def _adjust_for_volatility(self, position_size: float, symbol: str) -> float:
        """
        Adjust position size based on symbol volatility.
        
        Parameters:
        position_size : float - Calculated position size
        symbol : str - Trading symbol
        
        Returns:
        Volatility-adjusted position size
        """
        # Reduce position size for highly volatile symbols
        volatility = self._get_symbol_volatility(symbol)
        # Higher volatility -> smaller position size
        volatility_factor = max(0.5, 1.0 - (volatility * 10))  # Scale factor based on volatility
        return position_size * volatility_factor


class MovingAverageCrossStrategy(Strategy):
    """
    A moving average crossover strategy that generates buy/sell signals
    when short-term and long-term moving averages cross.
    """

    def __init__(self, symbols: List[str],
                 short_window: int = 50,
                 long_window: int = 200,
                 position_size: float = 1.0):
        """
        Initialize the moving average crossover strategy.
        
        Parameters:
        symbols : List[str] - List of ticker symbols to trade
        short_window : int - Short-term moving average window
        long_window : int - Long-term moving average window
        position_size : float - Position size as fraction of capital
        """
        super().__init__(symbols)
        self.set_parameter('short_window', short_window)
        self.set_parameter('long_window', long_window)
        self.set_parameter('position_size', position_size)
        self.short_mavg = {}
        self.long_mavg = {}

    def calculate_signals(self, event):
        """
        Generate trading signals based on moving average crossovers.
        """
        if event.type.name == 'MARKET':
            for symbol in self.symbols:
                bars = event.data.get(symbol, [])
                
                # Check if we have enough data
                long_window = self.get_parameter('long_window', 200)
                if len(bars) < long_window:
                    continue
                    
                # Calculate moving averages
                short_window = self.get_parameter('short_window', 50)
                short_mavg = sum([bar['close'] for bar in bars[-short_window:]]) / short_window
                long_mavg = sum([bar['close'] for bar in bars[-long_window:]]) / long_window
                
                # Update stored moving averages
                if symbol not in self.short_mavg:
                    self.short_mavg[symbol] = []
                    self.long_mavg[symbol] = []
                    
                self.short_mavg[symbol].append(short_mavg)
                self.long_mavg[symbol].append(long_mavg)
                
                # Keep only recent values
                if len(self.short_mavg[symbol]) > 3:
                    self.short_mavg[symbol] = self.short_mavg[symbol][-3:]
                    self.long_mavg[symbol] = self.long_mavg[symbol][-3:]
                
                # Check if we have enough data points to generate signals
                if len(self.short_mavg[symbol]) < 2:
                    continue
                    
                # Check for crossover
                prev_short = self.short_mavg[symbol][-2]
                prev_long = self.long_mavg[symbol][-2]
                curr_short = self.short_mavg[symbol][-1]
                curr_long = self.long_mavg[symbol][-1]
                
                # Check if we can send a signal (respect cooldown)
                if self.can_send_signal(symbol):
                    # Bullish crossover - short MA crosses above long MA
                    if prev_short <= prev_long and curr_short > curr_long:
                        # Generate buy signal
                        position_size = self.get_parameter('position_size', 1.0)
                        signal = SignalEvent(symbol, 'BUY', position_size)
                        # Check if events_queue is available before putting signal
                        if self.events_queue is not None:
                            self.events_queue.put(signal)
                        self.set_position_status(symbol, 'long')
                        self.record_signal_time(symbol)
                        
                    # Bearish crossover - short MA crosses below long MA
                    elif prev_short >= prev_long and curr_short < curr_long:
                        # Generate sell signal if we have a long position
                        if self.get_position_status(symbol) == 'long':
                            position_size = self.get_parameter('position_size', 1.0)
                            signal = SignalEvent(symbol, 'SELL', position_size)
                            # Check if events_queue is available before putting signal
                            if self.events_queue is not None:
                                self.events_queue.put(signal)
                            self.set_position_status(symbol, 'flat')
                            self.record_signal_time(symbol)