import datetime
from abc import ABC, abstractmethod
import random
from .event import FillEvent, OrderEvent


class ExecutionHandler(ABC):
    """
    Abstract base class for handling order execution.
    """

    @abstractmethod
    def execute_order(self, event: OrderEvent):
        """
        Takes an OrderEvent and executes it, producing
        a FillEvent that gets placed onto the Events queue.
        """
        raise NotImplementedError("Should implement execute_order()")


class SimulatedExecutionHandler(ExecutionHandler):
    """
    Simulates execution of orders in a backtesting environment.
    """

    def __init__(self, events, commission_rate=0.001, slippage_factor=0.0001):
        """
        Initializes the simulated execution handler.
        
        Parameters:
        events - The Queue of Event objects
        commission_rate - Commission rate per trade (default 0.1%)
        slippage_factor - Slippage factor (default 0.01%)
        """
        self.events = events
        self.commission_rate = commission_rate
        self.slippage_factor = slippage_factor

    def execute_order(self, event: OrderEvent):
        """
        Converts OrderEvents into FillEvents "naively",
        i.e. without any latency, slippage or fill ratio problems.
        """
        if event.type.name == 'ORDER':
            # Get market price from the event or use a default
            # In a more sophisticated implementation, this would come from market data
            if hasattr(event, 'market_price') and event.market_price:
                fill_price = event.market_price
            else:
                # Try to get price from a more realistic source
                # In a real implementation, this would come from the latest market data
                # For now, we'll use a more dynamic approach
                # Use a price that's more realistic based on typical market conditions
                base_price = 100.0
                # Add some realistic market movement
                market_movement = random.normalvariate(0, 0.01)  # 1% volatility
                fill_price = base_price * (1 + market_movement)
            
            # Apply slippage based on order direction and market conditions
            # More realistic slippage that varies with market volatility
            dynamic_slippage = self.slippage_factor * (1 + random.uniform(0, 1))
            
            if event.direction == 'BUY':
                fill_price = fill_price * (1 + dynamic_slippage)
            else:
                fill_price = fill_price * (1 - dynamic_slippage)
            
            # Calculate commission
            commission = self.commission_rate * event.quantity * fill_price
            
            # Create the FillEvent
            fill_event = FillEvent(
                symbol=event.symbol,
                quantity=event.quantity,
                direction=event.direction,
                fill_price=fill_price,
                commission=commission
            )
            
            # Put the FillEvent onto the events queue
            self.events.put(fill_event)