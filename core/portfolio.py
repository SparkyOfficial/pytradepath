import datetime
from abc import ABC, abstractmethod
from typing import Dict, List
import queue
from .event import OrderEvent, FillEvent


class Portfolio(ABC):
    """
    Abstract base class representing a portfolio of positions 
    (including both current and past positions).
    """

    def __init__(self, data_handler, events, start_date: str = "2023-01-01", initial_capital: float = 100000.0):
        """
        Initializes the portfolio with data on the current bars and
        an event queue.
        
        Parameters:
        data_handler - The data handler object
        events - The Event queue
        start_date - Start date of the portfolio
        initial_capital - Starting capital in USD
        """
        self.data_handler = data_handler
        self.events = events
        self.symbol_list = self.data_handler.symbol_list
        self.start_date = start_date
        self.initial_capital = initial_capital
        self.all_positions = self.construct_all_positions()
        self.current_positions = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        self.all_holdings = self.construct_all_holdings()
        self.current_holdings = self.construct_current_holdings()

    def construct_all_positions(self) -> List[Dict]:
        """
        Constructs the positions list using the start_date
        to determine when the time index will begin.
        """
        d = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        d['datetime'] = self.start_date
        return [d]

    def construct_all_holdings(self) -> List[Dict]:
        """
        Constructs the holdings list using the start_date
        to determine when the time index will begin.
        """
        d = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
        d['datetime'] = self.start_date
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return [d]

    def construct_current_holdings(self) -> Dict:
        """
        This constructs the dictionary which will hold the instantaneous
        value of the portfolio across all symbols.
        """
        d = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return d

    def update_timeindex(self, event):
        """
        Adds a new record to the positions matrix for the current
        market data bar. This reflects the PREVIOUS bar, i.e. all
        current market data at this stage is known (OHLCV).
        """
        # For simplicity, we'll use a counter instead of datetime
        latest_datetime = f"Bar_{len(self.all_positions)}"

        # Update positions
        dp = dict((k, v) for k, v in [(s, self.current_positions[s]) for s in self.symbol_list])
        dp['datetime'] = latest_datetime
        self.all_positions.append(dp)

        # Update holdings
        dh = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        dh['datetime'] = latest_datetime
        dh['cash'] = self.current_holdings['cash']
        dh['commission'] = self.current_holdings['commission']
        dh['total'] = self.current_holdings['cash'] + self.current_holdings['commission']

        # For simplicity, we'll assume a fixed price
        for s in self.symbol_list:
            # Approximation to the real value
            market_value = self.current_positions[s] * 100  # Fixed price assumption
            dh[s] = market_value
            dh['total'] += market_value

        self.all_holdings.append(dh)

    def update_positions_from_fill(self, fill: FillEvent):
        """
        Takes a FillEvent and updates the position matrix to
        reflect the new position.
        """
        # Check whether the fill is a buy or sell
        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1

        # Update positions list with new quantities
        self.current_positions[fill.symbol] += fill_dir * fill.quantity

    def update_holdings_from_fill(self, fill: FillEvent):
        """
        Takes a FillEvent and updates the holdings matrix to
        reflect the holdings value.
        """
        # Check whether the fill is a buy or sell
        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1

        # Update holdings list with new quantities
        cost = fill_dir * fill.fill_price * fill.quantity
        self.current_holdings[fill.symbol] += cost
        self.current_holdings['commission'] += fill.commission
        self.current_holdings['cash'] -= (cost + fill.commission)
        # Fix: Total should be cash + market value of positions, not subtracting cost
        # Recalculate total properly
        total = self.current_holdings['cash'] + self.current_holdings['commission']
        for symbol in self.symbol_list:
            # Approximate market value (this should be updated with real prices in a real implementation)
            market_value = self.current_positions[symbol] * 100  # Fixed price assumption
            total += market_value
        self.current_holdings['total'] = total

    def on_fill(self, event: FillEvent):
        """
        Updates the portfolio current positions and holdings 
        from a FillEvent.
        """
        self.update_positions_from_fill(event)
        self.update_holdings_from_fill(event)

    def generate_naive_order(self, signal):
        """
        Simply transacts an OrderEvent object as a constant quantity
        sizing of the signal object, without risk management or
        position sizing considerations.
        
        Parameters:
        signal - The SignalEvent signal information.
        """
        order = None

        symbol = signal.symbol
        signal_type = signal.signal_type
        strength = signal.strength

        mkt_quantity = 100
        cur_quantity = self.current_positions[symbol]
        order_type = 'MARKET'

        if signal_type == 'BUY' and cur_quantity == 0:
            order = OrderEvent(symbol, order_type, mkt_quantity, 'BUY')
        if signal_type == 'SELL' and cur_quantity == 0:
            order = OrderEvent(symbol, order_type, mkt_quantity, 'SELL')
        if signal_type == 'EXIT' and cur_quantity > 0:
            order = OrderEvent(symbol, order_type, abs(cur_quantity), 'SELL')
        if signal_type == 'EXIT' and cur_quantity < 0:
            order = OrderEvent(symbol, order_type, abs(cur_quantity), 'BUY')
        
        return order

    def on_signal(self, event):
        """
        Generates a list of orders based on the SignalEvent.
        """
        order_event = self.generate_naive_order(event)
        if order_event:
            self.events.put(order_event)


class NaivePortfolio(Portfolio):
    """
    The NaivePortfolio object is designed to send orders to
    a brokerage object with a constant quantity size blindly,
    i.e. without any risk management or position sizing. It is
    used to test simpler strategies.
    """
    
    def __init__(self, data_handler, events, start_date: str = "2023-01-01", initial_capital: float = 100000.0):
        super().__init__(data_handler, events, start_date, initial_capital)