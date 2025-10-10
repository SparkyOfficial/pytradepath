from core.event import SignalEvent
from core.strategy import Strategy


class MovingAverageCrossoverStrategy(Strategy):
    """
    Carries out a basic Moving Average Crossover strategy with a
    short/long simple weighted moving average. Default short/long
    windows are 100/400 periods respectively.
    """

    def __init__(self, symbols, short_window=3, long_window=5):
        """
        Initializes the Moving Average Cross Strategy.
        
        Parameters:
        symbols - List of ticker symbols to trade
        short_window - Short moving average window
        long_window - Long moving average window
        """
        super().__init__(symbols)
        self.short_window = short_window
        self.long_window = long_window
        
        # Initialize to False to indicate no position
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

    def calculate_signals(self, event):
        """
        Generates a new set of signals based on the moving average
        crossover strategy.
        """
        if event.type.name == 'MARKET':
            for symbol in self.symbols:
                # Get the latest bars for the symbol
                bars = event.data.get(symbol, [])
                
                # Check if we have enough data
                if len(bars) >= self.long_window:
                    # Calculate closing prices
                    closes = [bar['close'] for bar in bars]
                    
                    # Calculate the moving averages
                    short_sma = sum(closes[-self.short_window:]) / self.short_window
                    long_sma = sum(closes[-self.long_window:]) / self.long_window
                    
                    # Get previous values for crossover detection
                    if len(closes) >= self.long_window + 1:
                        prev_short_sma = sum(closes[-self.short_window-1:-1]) / self.short_window
                        prev_long_sma = sum(closes[-self.long_window-1:-1]) / self.long_window
                        
                        # Create signals based on moving average crossover
                        if short_sma > long_sma and prev_short_sma <= prev_long_sma:
                            # Bullish crossover - buy signal
                            if self.bought[symbol] == 'OUT':
                                signal = SignalEvent(symbol, 'BUY', 1.0)
                                self.events_queue.put(signal)
                                self.bought[symbol] = 'LONG'
                                
                        elif short_sma < long_sma and prev_short_sma >= prev_long_sma:
                            # Bearish crossover - sell signal
                            if self.bought[symbol] == 'LONG':
                                signal = SignalEvent(symbol, 'SELL', 1.0)
                                self.events_queue.put(signal)
                                self.bought[symbol] = 'OUT'