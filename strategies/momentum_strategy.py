from core.event import SignalEvent
from core.strategy import Strategy
import math


class MomentumStrategy(Strategy):
    """
    Implements a momentum-based trading strategy.
    Goes long on assets with strong positive momentum and short on assets with negative momentum.
    """

    def __init__(self, symbols, lookback_period=20, momentum_threshold=0.05):
        """
        Initializes the momentum strategy.
        
        Parameters:
        symbols - List of ticker symbols to trade
        lookback_period - Lookback period for momentum calculation (default 20)
        momentum_threshold - Minimum momentum threshold to generate signals (default 5%)
        """
        super().__init__(symbols)
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
        
        # Initialize position tracking
        self.bought = self._calculate_initial_bought()
        
        # Price history for momentum calculation
        self.price_history = {symbol: [] for symbol in symbols}

    def _calculate_initial_bought(self):
        """
        Adds keys to the bought dictionary for all symbols
        and sets them to 'OUT'.
        """
        bought = {}
        for s in self.symbols:
            bought[s] = 'OUT'
        return bought

    def _calculate_momentum(self, prices):
        """
        Calculate momentum as the percentage change over the lookback period.
        
        Parameters:
        prices - List of closing prices
        
        Returns:
        Momentum value as percentage change
        """
        if len(prices) < self.lookback_period + 1:
            return 0.0
            
        # Calculate momentum as percentage change
        current_price = prices[-1]
        lookback_price = prices[-(self.lookback_period + 1)]
        
        if lookback_price == 0:
            return 0.0
            
        momentum = (current_price - lookback_price) / lookback_price
        return momentum

    def calculate_signals(self, event):
        """
        Generates signals based on momentum values.
        """
        if event.type.name == 'MARKET':
            for symbol in self.symbols:
                # Get the latest bars for the symbol
                bars = event.data.get(symbol, [])
                
                # Update price history
                for bar in bars:
                    self.price_history[symbol].append(bar['close'])
                
                # Keep only recent history
                if len(self.price_history[symbol]) > self.lookback_period * 2:
                    self.price_history[symbol] = self.price_history[symbol][-self.lookback_period * 2:]
                
                # Check if we have enough data
                if len(self.price_history[symbol]) >= self.lookback_period + 1:
                    # Extract closing prices
                    prices = self.price_history[symbol]
                    
                    # Calculate momentum
                    momentum = self._calculate_momentum(prices)
                    
                    # Generate signals based on momentum thresholds
                    if momentum > self.momentum_threshold and self.bought[symbol] == 'OUT':
                        # Strong positive momentum - buy signal
                        signal = SignalEvent(symbol, 'BUY', min(1.0, momentum / self.momentum_threshold))
                        self.events_queue.put(signal)
                        self.bought[symbol] = 'LONG'
                        
                    elif momentum < -self.momentum_threshold and self.bought[symbol] == 'OUT':
                        # Strong negative momentum - sell signal
                        signal = SignalEvent(symbol, 'SELL', min(1.0, abs(momentum) / self.momentum_threshold))
                        self.events_queue.put(signal)
                        self.bought[symbol] = 'SHORT'
                        
                    elif abs(momentum) < self.momentum_threshold / 2 and self.bought[symbol] != 'OUT':
                        # Momentum has weakened - exit position
                        signal = SignalEvent(symbol, 'EXIT', 1.0)
                        self.events_queue.put(signal)
                        self.bought[symbol] = 'OUT'