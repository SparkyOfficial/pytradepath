from core.event import SignalEvent
from core.strategy import Strategy


class RSIStrategy(Strategy):
    """
    Implements a Relative Strength Index (RSI) mean reversion strategy.
    Goes long when RSI < 30 (oversold) and short when RSI > 70 (overbought).
    """

    def __init__(self, symbols, period=14, overbought=70, oversold=30):
        """
        Initializes the RSI strategy.
        
        Parameters:
        symbols - List of ticker symbols to trade
        period - RSI calculation period (default 14)
        overbought - Overbought threshold (default 70)
        oversold - Oversold threshold (default 30)
        """
        super().__init__(symbols)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        
        # Initialize position tracking
        self.bought = self._calculate_initial_bought()

    def _calculate_initial_bought(self):
        """
        Adds keys to the bought dictionary for all symbols
        and sets them to 'OUT'.
        """
        bought = {}
        for s in self.symbols:
            bought[s] = 'OUT'
        return bought

    def _calculate_rsi(self, closes):
        """
        Calculate the Relative Strength Index.
        
        Parameters:
        closes - List of closing prices
        
        Returns:
        RSI value
        """
        if len(closes) < self.period + 1:
            return 50  # Not enough data, return neutral value
            
        # Calculate price changes
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        
        # Separate gains and losses
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        # Calculate average gains and losses
        avg_gain = sum(gains[-self.period:]) / self.period
        avg_loss = sum(losses[-self.period:]) / self.period
        
        if avg_loss == 0:
            return 100  # No losses, RSI is 100
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def calculate_signals(self, event):
        """
        Generates signals based on RSI values.
        """
        if event.type.name == 'MARKET':
            for symbol in self.symbols:
                # Get the latest bars for the symbol
                bars = event.data.get(symbol, [])
                
                # Check if we have enough data
                if len(bars) >= self.period:
                    # Extract closing prices
                    closes = [bar['close'] for bar in bars]
                    
                    # Calculate RSI
                    rsi = self._calculate_rsi(closes)
                    
                    # Generate signals based on RSI thresholds
                    if rsi < self.oversold and self.bought[symbol] == 'OUT':
                        # Oversold condition - buy signal
                        signal = SignalEvent(symbol, 'BUY', 1.0)
                        self.events_queue.put(signal)
                        self.bought[symbol] = 'LONG'
                        
                    elif rsi > self.overbought and self.bought[symbol] == 'OUT':
                        # Overbought condition - sell signal
                        signal = SignalEvent(symbol, 'SELL', 1.0)
                        self.events_queue.put(signal)
                        self.bought[symbol] = 'SHORT'
                        
                    elif rsi > 50 and self.bought[symbol] == 'LONG':
                        # Exit long position
                        signal = SignalEvent(symbol, 'EXIT', 1.0)
                        self.events_queue.put(signal)
                        self.bought[symbol] = 'OUT'
                        
                    elif rsi < 50 and self.bought[symbol] == 'SHORT':
                        # Exit short position
                        signal = SignalEvent(symbol, 'EXIT', 1.0)
                        self.events_queue.put(signal)
                        self.bought[symbol] = 'OUT'