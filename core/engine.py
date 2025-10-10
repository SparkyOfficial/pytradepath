import queue
import time
from typing import Type, Callable
from .data_handler import DataHandler
from .strategy import Strategy
from .portfolio import Portfolio
from .execution import ExecutionHandler
from .event import MarketEvent, SignalEvent, OrderEvent, FillEvent


class BacktestingEngine:
    """
    Main backtesting engine that encapsulates the event loop
    and coordinates all components of the system.
    """

    def __init__(self, 
                 data_handler: Callable,
                 strategy: Callable,
                 portfolio: Callable,
                 execution_handler: Callable,
                 symbol_list: list,
                 initial_capital: float = 100000.0):
        """
        Initializes the backtesting engine.
        
        Parameters:
        data_handler - Function that returns a data handler instance
        strategy - Function that returns a strategy instance
        portfolio - Function that returns a portfolio instance
        execution_handler - Function that returns an execution handler instance
        symbol_list - List of symbol strings
        initial_capital - Starting capital for the portfolio
        """
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        
        # Data handlers
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        
        # Event queue
        self.events = queue.Queue()
        
        # Signals, orders, fills
        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1
       
        # Market data
        self.continue_backtest = True
        self.current_bar_date = None
        
        # Performance tracking
        self.equity_curve = []
        
        self._generate_trading_instances()

    def _generate_trading_instances(self):
        """
        Generates trading instances for each component.
        """
        print("Creating DataHandler, Strategy, Portfolio and ExecutionHandler")
        self.data_handler = self.data_handler()
        self.strategy = self.strategy(self.symbol_list)
        self.portfolio = self.portfolio(self.data_handler, self.events, 
                                       initial_capital=self.initial_capital)
        self.execution_handler = self.execution_handler(self.events)
        
        # Set the events queue for the strategy
        self.strategy.set_events_queue(self.events)

    def _run_strategy(self, market_event: MarketEvent):
        """
        Executes the strategy on the market event.
        """
        self.strategy.calculate_signals(market_event)

    def _run_portfolio(self, signal_event: SignalEvent):
        """
        Executes the portfolio on the signal event.
        """
        self.portfolio.on_signal(signal_event)

    def _run_execution(self, order_event: OrderEvent):
        """
        Executes the order in the execution handler.
        """
        self.execution_handler.execute_order(order_event)

    def _run_fill(self, fill_event: FillEvent):
        """
        Updates the portfolio with the fill event.
        """
        self.portfolio.on_fill(fill_event)

    def _update_portfolio(self, market_event: MarketEvent):
        """
        Updates the portfolio with the latest market data.
        """
        self.portfolio.update_timeindex(market_event)

    def run(self):
        """
        Executes the backtest by running the event loop.
        """
        i = 0
        while self.continue_backtest:
            i += 1
            print(f"{i} ", end="")
            
            # Update the market bars
            if self.data_handler.update_bars():
                # Handle market event
                # Fix: Pass the latest symbol data correctly
                market_event = MarketEvent("MARKET_DATA", self.data_handler.latest_symbol_data)
                self.events.put(market_event)
            else:
                self.continue_backtest = False
            
            # Process events
            while True:
                try:
                    event = self.events.get(False)
                except queue.Empty:
                    break
                else:
                    if event is not None:
                        if event.type.name == 'MARKET':
                            self._update_portfolio(event)
                            self._run_strategy(event)
                        elif event.type.name == 'SIGNAL':
                            self.signals += 1
                            self._run_portfolio(event)
                        elif event.type.name == 'ORDER':
                            self.orders += 1
                            self._run_execution(event)
                        elif event.type.name == 'FILL':
                            self.fills += 1
                            self._run_fill(event)
            
            if i % 10 == 0:
                print(f"\nProcessed {i} bars")
        
        print("\nBacktest complete.")
        self._output_performance()

    def _output_performance(self):
        """
        Outputs the performance of the strategy.
        """
        print(f"Signals: {self.signals}")
        print(f"Orders: {self.orders}")
        print(f"Fills: {self.fills}")