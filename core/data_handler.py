import csv
import json
import random
from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Optional
from collections import deque
from .event import MarketEvent
import datetime


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

    def __init__(self, csv_dir: str, symbol_list: list, 
                 data_frequency: str = 'daily',
                 adjust_data: bool = True):
        """
        Initializes the data handler with a CSV directory and symbol list.
        
        Parameters:
        csv_dir : str - Path to the CSV files directory
        symbol_list : list - List of symbol strings
        data_frequency : str - Data frequency ('daily', 'hourly', 'minute')
        adjust_data : bool - Whether to adjust prices for splits/dividends
        """
        super().__init__()
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.data_frequency = data_frequency
        self.adjust_data = adjust_data
        
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self.bar_index = 0
        self.data_providers = {}  # Support for multiple data providers
        
        self._open_convert_csv_files()

    def _open_convert_csv_files(self):
        """
        Opens the CSV files from the data directory, converting
        them into iterators with enhanced data processing.
        """
        for s in self.symbol_list:
            try:
                # Load the CSV file
                with open(f"{self.csv_dir}/{s}.csv", 'r') as f:
                    reader = csv.DictReader(f)
                    data = list(reader)
                
                # Convert to proper data types with error handling
                processed_data = []
                for row in data:
                    try:
                        processed_row = {
                            'symbol': s,
                            'datetime': self._parse_datetime(row.get('date', row.get('timestamp', ''))),
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': int(row['volume']),
                            'adj_close': float(row.get('adj_close', row['close']))
                        }
                        
                        # Apply data adjustments if requested
                        if self.adjust_data and 'adj_close' in row:
                            processed_row = self._adjust_prices(processed_row, row)
                            
                        processed_data.append(processed_row)
                    except (ValueError, KeyError) as e:
                        # Skip rows with invalid data
                        continue
                
                # Sort data by datetime to ensure proper order
                processed_data.sort(key=lambda x: x['datetime'])
                
                self.symbol_data[s] = iter(processed_data)
                self.latest_symbol_data[s] = deque(maxlen=10000)  # Keep last 10000 bars for memory efficiency
                
            except FileNotFoundError:
                print(f"Warning: CSV file for symbol {s} not found in {self.csv_dir}")
                self.symbol_data[s] = iter([])
                self.latest_symbol_data[s] = deque(maxlen=10000)
            except Exception as e:
                print(f"Error loading data for symbol {s}: {e}")
                self.symbol_data[s] = iter([])
                self.latest_symbol_data[s] = deque(maxlen=10000)

    def _parse_datetime(self, date_str: str) -> datetime.datetime:
        """
        Parse datetime string with multiple format support.
        
        Parameters:
        date_str : str - Date/time string
        
        Returns:
        datetime object
        """
        formats = [
            '%Y-%m-%d',
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d',
            '%Y/%m/%d %H:%M:%S',
            '%d-%m-%Y',
            '%d/%m/%Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.datetime.strptime(date_str, fmt)
            except ValueError:
                continue
                
        # If all formats fail, return current time
        return datetime.datetime.now()

    def _adjust_prices(self, processed_row: dict, original_row: dict) -> dict:
        """
        Adjust prices for splits and dividends.
        
        Parameters:
        processed_row : dict - Processed data row
        original_row : dict - Original CSV row
        
        Returns:
        Adjusted data row
        """
        try:
            adj_close = float(original_row.get('adj_close', processed_row['close']))
            close = processed_row['close']
            
            if close != 0 and adj_close != close:
                adjustment_factor = adj_close / close
                
                # Apply adjustment factor to all price fields
                processed_row['open'] *= adjustment_factor
                processed_row['high'] *= adjustment_factor
                processed_row['low'] *= adjustment_factor
                processed_row['close'] = adj_close  # Use adjusted close directly
                
            return processed_row
        except (ValueError, ZeroDivisionError):
            # If adjustment fails, return original row
            return processed_row

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
        for all symbols in the symbol_list with enhanced error handling.
        """
        any_data = False
        for s in self.symbol_list:
            try:
                bar = next(self.symbol_data[s])
                # Add realistic market microstructure effects
                bar = self._add_market_effects(bar)
                self.latest_symbol_data[s].append(bar)
                any_data = True
            except StopIteration:
                # No more data for this symbol
                pass
            except Exception as e:
                # Handle other potential errors
                print(f"Error updating bar for symbol {s}: {e}")
                
        self.continue_backtest = any_data
        self.bar_index += 1
        return self.continue_backtest

    def _add_market_effects(self, bar: dict) -> dict:
        """
        Add realistic market microstructure effects to price data.
        
        Parameters:
        bar : dict - Price bar data
        
        Returns:
        Enhanced bar data with market effects
        """
        # Add realistic bid-ask spread
        spread = random.uniform(0.001, 0.005) * bar['close']  # 0.1% to 0.5% spread
        bar['bid'] = bar['close'] - spread / 2
        bar['ask'] = bar['close'] + spread / 2
        
        # Add volume clustering effects
        if bar['volume'] > 0:
            # Add some randomness to volume while maintaining realistic patterns
            volume_noise = random.uniform(-0.2, 0.2)
            bar['volume'] = max(0, int(bar['volume'] * (1 + volume_noise)))
        
        return bar

    def get_bar_datetime(self, symbol: str) -> Optional[datetime.datetime]:
        """
        Get the datetime of the latest bar for a symbol.
        
        Parameters:
        symbol : str - Trading symbol
        
        Returns:
        datetime of latest bar or None
        """
        try:
            latest_bars = self.get_latest_bars(symbol, 1)
            if latest_bars:
                return latest_bars[-1]['datetime']
            return None
        except:
            return None

    def get_symbols(self) -> list:
        """
        Get the list of available symbols.
        
        Returns:
        List of symbol strings
        """
        return list(self.symbol_data.keys())