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
            # For simplicity, we'll assume a fixed price
            fill_price = 100.0
            
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