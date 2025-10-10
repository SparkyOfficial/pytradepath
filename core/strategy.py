from abc import ABC, abstractmethod
from typing import List
from .event import SignalEvent


class Strategy(ABC):
    """
    Abstract base class for strategy objects that generate signals
    based on market data.
    """

    def __init__(self, symbols: List[str]):
        """
        Initializes the strategy with a list of symbols.
        
        Parameters:
        symbols : List[str] - List of ticker symbols to trade
        """
        self.symbols = symbols
        self.events_queue = None

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


class BuyAndHoldStrategy(Strategy):
    """
    A simple buy and hold strategy that buys a symbol once
    and holds it throughout the backtest.
    """

    def __init__(self, symbols: List[str]):
        super().__init__(symbols)
        self.bought = self._calculate_initial_bought()

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
        and then no additional signals.
        """
        if event.type.name == 'MARKET':
            for s in self.symbols:
                bars = event.data
                if bars is not None and not self.bought[s]:
                    # Create a signal for the symbol
                    signal = SignalEvent(s, 'BUY', 1.0)
                    self.events_queue.put(signal)
                    self.bought[s] = True