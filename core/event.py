from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional
import time


class EventType(Enum):
    """Enumeration of all possible event types in the system."""
    MARKET = 1      # Market data update
    SIGNAL = 2      # Trading signal from strategy
    ORDER = 3       # Order to be executed
    FILL = 4        # Order execution confirmation


class Event(ABC):
    """
    Base class for all events in the system.
    """
    def __init__(self, event_type: EventType):
        self.type = event_type
        self.timestamp = time.time()

    @abstractmethod
    def __str__(self) -> str:
        pass


class MarketEvent(Event):
    """
    Event representing new market data.
    """
    def __init__(self, symbol: str, data: Any):
        super().__init__(EventType.MARKET)
        self.symbol = symbol
        self.data = data

    def __str__(self) -> str:
        return f"MarketEvent: {self.symbol} at {self.timestamp}"


class SignalEvent(Event):
    """
    Event representing a trading signal from a strategy.
    """
    def __init__(self, symbol: str, signal_type: str, strength: float):
        super().__init__(EventType.SIGNAL)
        self.symbol = symbol
        self.signal_type = signal_type  # 'BUY', 'SELL', 'EXIT'
        self.strength = strength  # Confidence level (0.0 to 1.0)

    def __str__(self) -> str:
        return f"SignalEvent: {self.signal_type} {self.symbol} with strength {self.strength}"


class OrderEvent(Event):
    """
    Event representing an order to be placed.
    """
    def __init__(self, symbol: str, order_type: str, quantity: float, direction: str, market_price: Optional[float] = None):
        super().__init__(EventType.ORDER)
        self.symbol = symbol
        self.order_type = order_type  # 'MARKET', 'LIMIT', etc.
        self.quantity = quantity
        self.direction = direction  # 'BUY' or 'SELL'
        self.market_price = market_price  # Optional market price for order execution

    def __str__(self) -> str:
        return f"OrderEvent: {self.direction} {self.quantity} {self.symbol} ({self.order_type})"


class FillEvent(Event):
    """
    Event representing a filled order.
    """
    def __init__(self, symbol: str, quantity: float, direction: str, 
                 fill_price: float, commission: float = 0.0):
        super().__init__(EventType.FILL)
        self.symbol = symbol
        self.quantity = quantity
        self.direction = direction  # 'BUY' or 'SELL'
        self.fill_price = fill_price
        self.commission = commission
        self.cost = quantity * fill_price + commission

    def __str__(self) -> str:
        return f"FillEvent: {self.direction} {self.quantity} {self.symbol} at {self.fill_price}"