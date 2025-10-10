import csv
from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Optional
from collections import deque
from .event import MarketEvent


class DataHandler(ABC):
    """
    Abstract base class for data handlers that provides market data
    to the backtesting engine.
    """

    def __init__(self):
        self.latest_data = {}
        self.continue_backtest = True

    @abstractmethod
    def get_latest_bars(self, symbol: str, n: int = 1) -> list:
        """
        Return the last N bars from the latest_symbol list,
        or fewer if less bars are available.
        """
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def update_bars(self) -> bool:
        """
        Pushes the latest bar to the latest symbol structure
        for all symbols in the symbol list.
        Returns True if successful, False otherwise.
        """
        raise NotImplementedError("Should implement update_bars()")


class HistoricCSVDataHandler(DataHandler):
    """
    HistoricCSVDataHandler reads CSV files containing OHLCV data
    and provides an interface to get market data.
    """

    def __init__(self, csv_dir: str, symbol_list: list):
        """
        Initializes the data handler with a CSV directory and symbol list.
        
        Parameters:
        csv_dir : str - Path to the CSV files directory
        symbol_list : list - List of symbol strings
        """
        super().__init__()
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self.bar_index = 0
        
        self._open_convert_csv_files()

    def _open_convert_csv_files(self):
        """
        Opens the CSV files from the data directory, converting
        them into iterators.
        """
        for s in self.symbol_list:
            # Load the CSV file
            with open(f"{self.csv_dir}/{s}.csv", 'r') as f:
                reader = csv.DictReader(f)
                data = list(reader)
                
            # Convert to proper data types
            for row in data:
                row['open'] = float(row['open'])
                row['high'] = float(row['high'])
                row['low'] = float(row['low'])
                row['close'] = float(row['close'])
                row['volume'] = int(row['volume'])
                
            self.symbol_data[s] = iter(data)
            self.latest_symbol_data[s] = deque(maxlen=1000)  # Keep last 1000 bars

    def get_latest_bars(self, symbol: str, n: int = 1) -> list:
        """
        Returns the last N bars from the latest_symbol list,
        or N-k if less available.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print(f"That symbol is not available in the historical data set.")
            raise
        else:
            return list(bars_list)[-n:] if len(bars_list) >= n else list(bars_list)

    def update_bars(self) -> bool:
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol_list.
        """
        any_data = False
        for s in self.symbol_list:
            try:
                bar = next(self.symbol_data[s])
                self.latest_symbol_data[s].append(bar)
                any_data = True
            except StopIteration:
                pass
                
        self.continue_backtest = any_data
        self.bar_index += 1
        return self.continue_backtest