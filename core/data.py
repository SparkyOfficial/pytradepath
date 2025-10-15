import csv
import json
import os
import sqlite3
import warnings
import math
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta


class DataProvider(ABC):
    """
    Abstract base class for data providers.
    """

    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: str, 
                           end_date: str) -> list:
        """
        Get historical data for a symbol.
        
        Parameters:
        symbol - Symbol to get data for
        start_date - Start date (YYYY-MM-DD)
        end_date - End date (YYYY-MM-DD)
        
        Returns:
        List with historical data
        """
        raise NotImplementedError("Should implement get_historical_data()")

    @abstractmethod
    def get_realtime_data(self, symbol: str) -> Dict:
        """
        Get real-time data for a symbol.
        
        Parameters:
        symbol - Symbol to get data for
        
        Returns:
        Dictionary with real-time data
        """
        raise NotImplementedError("Should implement get_realtime_data()")


class CSVDataProvider(DataProvider):
    """
    Data provider for CSV files.
    """

    def __init__(self, data_directory: str = "data"):
        """
        Initialize the CSV data provider.
        
        Parameters:
        data_directory - Directory containing CSV files
        """
        self.data_directory = data_directory

    def get_historical_data(self, symbol: str, start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> list:
        """
        Get historical data from CSV file.
        
        Parameters:
        symbol - Symbol to get data for
        start_date - Start date (YYYY-MM-DD)
        end_date - End date (YYYY-MM-DD)
        
        Returns:
        List with historical data
        """
        # Construct file path
        file_path = os.path.join(self.data_directory, f"{symbol}.csv")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Read CSV file
        data = []
        try:
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Filter by date range if specified
                    if start_date and row['datetime'] < start_date:
                        continue
                    if end_date and row['datetime'] > end_date:
                        continue
                    data.append(row)
        except Exception as e:
            raise ValueError(f"Error reading CSV file {file_path}: {e}")
        
        return data

    def get_realtime_data(self, symbol: str) -> Dict:
        """
        Get "real-time" data (for CSV, this is the latest row).
        
        Parameters:
        symbol - Symbol to get data for
        
        Returns:
        Dictionary with latest data
        """
        data = self.get_historical_data(symbol)
        if not data:
            return {}
        
        latest_row = data[-1]
        return {
            'symbol': symbol,
            'timestamp': latest_row['datetime'],
            'open': float(latest_row['open']),
            'high': float(latest_row['high']),
            'low': float(latest_row['low']),
            'close': float(latest_row['close']),
            'volume': int(latest_row['volume'])
        }


class DatabaseDataProvider(DataProvider):
    """
    Data provider for SQLite database.
    """

    def __init__(self, db_path: str = "data/market_data.db"):
        """
        Initialize the database data provider.
        
        Parameters:
        db_path - Path to SQLite database
        """
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """
        Initialize the database with required tables.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for market data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
            ON market_data (symbol, timestamp)
        ''')
        
        conn.commit()
        conn.close()

    def get_historical_data(self, symbol: str, start_date: str = None, 
                           end_date: str = None) -> list:
        """
        Get historical data from database.
        
        Parameters:
        symbol - Symbol to get data for
        start_date - Start date (YYYY-MM-DD) (optional)
        end_date - End date (YYYY-MM-DD) (optional)
        
        Returns:
        List with historical data
        """
        conn = sqlite3.connect(self.db_path)
        
        # Build query
        query = "SELECT * FROM market_data WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
            
        query += " ORDER BY timestamp"
        
        # Execute query
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Convert to list of dictionaries
        columns = [description[0] for description in cursor.description]
        data = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        return data

    def get_realtime_data(self, symbol: str) -> Dict:
        """
        Get latest data from database.
        
        Parameters:
        symbol - Symbol to get data for
        
        Returns:
        Dictionary with latest data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM market_data 
            WHERE symbol = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
        ''', (symbol,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return {}
        
        columns = [description[0] for description in cursor.description]
        data_dict = dict(zip(columns, row))
        
        return {
            'symbol': data_dict['symbol'],
            'timestamp': data_dict['timestamp'],
            'open': data_dict['open'],
            'high': data_dict['high'],
            'low': data_dict['low'],
            'close': data_dict['close'],
            'volume': data_dict['volume']
        }

    def insert_data(self, symbol: str, data: list):
        """
        Insert data into database.
        
        Parameters:
        symbol - Symbol for the data
        data - List with data to insert
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert data
        for row in data:
            cursor.execute('''
                INSERT OR REPLACE INTO market_data 
                (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                row['timestamp'],
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume']
            ))
        
        conn.commit()
        conn.close()


class APIBasedDataProvider(DataProvider):
    """
    Data provider for API-based data sources.
    """

    def __init__(self, api_key: str = None, base_url: str = "https://api.example.com"):
        """
        Initialize the API data provider.
        
        Parameters:
        api_key - API key for authentication
        base_url - Base URL for the API
        """
        self.api_key = api_key
        self.base_url = base_url

    def get_historical_data(self, symbol: str, start_date: str, 
                           end_date: str) -> list:
        """
        Get historical data from API.
        
        Parameters:
        symbol - Symbol to get data for
        start_date - Start date (YYYY-MM-DD)
        end_date - End date (YYYY-MM-DD)
        
        Returns:
        List with historical data
        """
        # Generate synthetic data that simulates real API response
        print(f"API Provider: Generating synthetic data for {symbol} from {start_date} to {end_date}")
        
        # Parse dates
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            # Fallback to default range if dates are invalid
            start_dt = datetime.now() - timedelta(days=365)
            end_dt = datetime.now()
        
        # Generate data points with realistic market characteristics
        data = []
        current_dt = start_dt
        price = 100.0  # Starting price
        
        while current_dt <= end_dt:
            # Generate realistic price movement with trends and volatility
            # This simulates a more realistic market data generation
            trend = 0.0001  # Small upward trend
            volatility = 0.02  # 2% daily volatility
            random_shock = random.normalvariate(0, volatility)
            price_change = trend + random_shock
            price = price * (1 + price_change)
            
            # Ensure realistic OHLC values
            open_price = price
            high_price = open_price * (1 + random.uniform(0, 0.01))
            low_price = open_price * (1 - random.uniform(0, 0.01))
            close_price = open_price * (1 + random.uniform(-0.01, 0.01))
            
            volume = random.randint(1000, 100000)
            
            data.append({
                'symbol': symbol,
                'timestamp': current_dt.strftime("%Y-%m-%d"),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
            
            current_dt += timedelta(days=1)
        
        return data

    def get_realtime_data(self, symbol: str) -> Dict:
        """
        Get real-time data from API.
        
        Parameters:
        symbol - Symbol to get data for
        
        Returns:
        Dictionary with real-time data
        """
        # Generate synthetic real-time data with realistic characteristics
        price = 100.0 + random.uniform(-10, 10)
        
        # Add realistic market movements
        volatility = 0.01  # 1% volatility
        random_shock = random.normalvariate(0, volatility)
        price = price * (1 + random_shock)
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'open': round(price, 2),
            'high': round(price * (1 + random.uniform(0, 0.005)), 2),
            'low': round(price * (1 - random.uniform(0, 0.005)), 2),
            'close': round(price * (1 + random.uniform(-0.005, 0.005)), 2),
            'volume': random.randint(100, 10000)
        }


class DataManager:
    """
    Manages data from multiple sources and provides unified interface.
    """

    def __init__(self, providers: List[DataProvider] = None):
        """
        Initialize the data manager.
        
        Parameters:
        providers - List of data providers
        """
        self.providers = providers if providers else []
        self.cache = {}  # Cache for frequently accessed data

    def add_provider(self, provider: DataProvider):
        """
        Add a data provider.
        
        Parameters:
        provider - Data provider to add
        """
        self.providers.append(provider)

    def get_historical_data(self, symbol: str, start_date: str = None, 
                           end_date: str = None, use_cache: bool = True) -> list:
        """
        Get historical data from available providers.
        
        Parameters:
        symbol - Symbol to get data for
        start_date - Start date (YYYY-MM-DD)
        end_date - End date (YYYY-MM-DD)
        use_cache - Whether to use cached data
        
        Returns:
        List with historical data
        """
        # Check cache first
        cache_key = f"{symbol}_{start_date}_{end_date}"
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try each provider
        for provider in self.providers:
            try:
                data = provider.get_historical_data(symbol, start_date, end_date)
                if data:
                    # Cache the result
                    self.cache[cache_key] = data
                    return data
            except Exception as e:
                print(f"Error getting data from {provider.__class__.__name__}: {e}")
                continue
        
        # If no provider worked, return empty list
        return []

    def get_realtime_data(self, symbol: str) -> Dict:
        """
        Get real-time data from available providers.
        
        Parameters:
        symbol - Symbol to get data for
        
        Returns:
        Dictionary with real-time data
        """
        # Try each provider
        for provider in self.providers:
            try:
                data = provider.get_realtime_data(symbol)
                if data:
                    return data
            except Exception as e:
                print(f"Error getting real-time data from {provider.__class__.__name__}: {e}")
                continue
        
        # If no provider worked, return empty dictionary
        return {}

    def clear_cache(self):
        """
        Clear the data cache.
        """
        self.cache.clear()


class DataValidator:
    """
    Validates data quality and integrity.
    """

    def __init__(self):
        """
        Initialize the data validator.
        """
        pass

    def validate_data(self, data: list, symbol: str = "") -> Dict:
        """
        Validate data quality.
        
        Parameters:
        data - List to validate
        symbol - Symbol for the data (for reporting)
        
        Returns:
        Dictionary with validation results
        """
        results = {
            'symbol': symbol,
            'total_rows': len(data),
            'missing_values': {},
            'duplicates': 0,
            'issues': []
        }
        
        if not data:
            results['issues'].append("No data provided")
            return results
        
        # Check for missing values
        missing_counts = {}
        for row in data:
            for key, value in row.items():
                if value == "" or value is None:
                    if key not in missing_counts:
                        missing_counts[key] = 0
                    missing_counts[key] += 1
        
        results['missing_values'] = missing_counts
        
        # Check for negative prices or volume
        negative_prices = 0
        negative_volume = 0
        zero_prices = 0
        
        for row in data:
            try:
                close_price = float(row.get('close', 0))
                volume = int(row.get('volume', 0))
                
                if close_price < 0:
                    negative_prices += 1
                if volume < 0:
                    negative_volume += 1
                if close_price == 0:
                    zero_prices += 1
            except (ValueError, TypeError):
                pass
        
        if negative_prices > 0:
            results['issues'].append(f"{negative_prices} negative close prices")
        
        if negative_volume > 0:
            results['issues'].append(f"{negative_volume} negative volumes")
        
        if zero_prices > 0:
            results['issues'].append(f"{zero_prices} zero close prices")
        
        return results

    def clean_data(self, data: list) -> list:
        """
        Clean data by removing duplicates and handling missing values.
        
        Parameters:
        data - List to clean
        
        Returns:
        Cleaned list
        """
        if not data:
            return data
        
        # Remove duplicates based on timestamp
        seen_timestamps = set()
        cleaned_data = []
        
        for row in data:
            timestamp = row.get('timestamp')
            if timestamp not in seen_timestamps:
                seen_timestamps.add(timestamp)
                cleaned_data.append(row)
        
        # Forward fill missing values
        if cleaned_data:
            last_valid_row = cleaned_data[0].copy()
            for i, row in enumerate(cleaned_data):
                for key, value in row.items():
                    if value == "" or value is None:
                        cleaned_data[i][key] = last_valid_row.get(key, "")
                    else:
                        last_valid_row[key] = value
        
        return cleaned_data


class DataTransformer:
    """
    Transforms raw data into features for machine learning.
    """

    def __init__(self):
        """
        Initialize the data transformer.
        """
        pass

    def add_technical_indicators(self, data: list) -> list:
        """
        Add common technical indicators to the data.
        
        Parameters:
        data - List with OHLCV data
        
        Returns:
        List with added technical indicators
        """
        if not data:
            return data
        
        transformed_data = []
        
        # Add moving averages
        for i, row in enumerate(data):
            new_row = row.copy()
            
            # Enhanced moving averages with proper window validation
            # Using more robust calculation methods
            if i >= 9:
                # Calculate 10-period moving average with proper data validation
                ma_window = [float(data[j]['close']) for j in range(i-9, i+1) if 'close' in data[j]]
                if ma_window:
                    ma_10 = sum(ma_window) / len(ma_window)
                    new_row['MA_10'] = ma_10
                else:
                    new_row['MA_10'] = float(row['close'])
            else:
                new_row['MA_10'] = float(row['close'])
            
            if i >= 49:
                # Calculate 50-period moving average with proper data validation
                ma_window = [float(data[j]['close']) for j in range(i-49, i+1) if 'close' in data[j]]
                if ma_window:
                    ma_50 = sum(ma_window) / len(ma_window)
                    new_row['MA_50'] = ma_50
                else:
                    new_row['MA_50'] = float(row['close'])
            else:
                new_row['MA_50'] = float(row['close'])
            
            transformed_data.append(new_row)
        
        return transformed_data

    def add_lag_features(self, data: list, 
                        columns: List[str] = None, 
                        lags: List[int] = [1, 2, 3]) -> list:
        """
        Add lagged features to the data.
        
        Parameters:
        data - List to add features to
        columns - Columns to create lags for (default: all numeric columns)
        lags - List of lag periods
        
        Returns:
        List with added lag features
        """
        if not data:
            return data
        
        transformed_data = []
        
        # If no columns specified, use common columns
        if columns is None:
            columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Add lag features
        for i, row in enumerate(data):
            new_row = row.copy()
            
            for col in columns:
                if col in row:
                    for lag in lags:
                        if i >= lag:
                            lag_value = data[i - lag].get(col, 0)
                            new_row[f'{col}_lag_{lag}'] = lag_value
                        else:
                            new_row[f'{col}_lag_{lag}'] = row.get(col, 0)
            
            transformed_data.append(new_row)
        
        return transformed_data

    def add_return_features(self, data: list, 
                           periods: List[int] = [1, 5, 10]) -> list:
        """
        Add return features to the data.
        
        Parameters:
        data - List to add features to
        periods - List of return periods
        
        Returns:
        List with added return features
        """
        if not data:
            return data
        
        transformed_data = []
        
        # Add return features
        for i, row in enumerate(data):
            new_row = row.copy()
            
            for period in periods:
                if i >= period:
                    close_current = float(row['close'])
                    close_past = float(data[i - period]['close'])
                    if close_past != 0:
                        return_val = (close_current - close_past) / close_past
                    else:
                        return_val = 0
                    new_row[f'return_{period}'] = return_val
                else:
                    new_row[f'return_{period}'] = 0
            
            transformed_data.append(new_row)
        
        return transformed_data


class DataSampler:
    """
    Samples data for backtesting at different frequencies.
    """

    def __init__(self):
        """
        Initialize the data sampler.
        """
        pass

    def resample_data(self, data: list, 
                     frequency: str = '1D') -> list:
        """
        Resample data to a different frequency.
        
        Parameters:
        data - List to resample
        frequency - Target frequency (e.g., '1H', '1D', '1W')
        
        Returns:
        Resampled list
        """
        if not data:
            return data
        
        # Parse frequency
        if frequency.endswith('H'):
            target_freq = int(frequency[:-1]) * 60  # Convert to minutes
        elif frequency.endswith('D'):
            target_freq = int(frequency[:-1]) * 1440  # Convert to minutes
        elif frequency.endswith('W'):
            target_freq = int(frequency[:-1]) * 10080  # Convert to minutes
        else:
            target_freq = 1440  # Default to daily
        
        # Group data by frequency
        resampled_data = []
        current_group = []
        last_timestamp = None
        
        for row in data:
            timestamp = row.get('timestamp')
            if not timestamp:
                continue
                
            # Parse actual timestamp for more accurate grouping
            try:
                # Try to parse the timestamp - handles multiple formats
                from datetime import datetime
                if ' ' in timestamp:
                    # Format: YYYY-MM-DD HH:MM:SS
                    parsed_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                else:
                    # Format: YYYY-MM-DD
                    parsed_time = datetime.strptime(timestamp, "%Y-%m-%d")
            except ValueError:
                # Fallback to simple string comparison
                parsed_time = timestamp
            
            if not current_group:
                current_group.append(row)
                last_timestamp = parsed_time
            else:
                # Add to current group
                current_group.append(row)
                
                # Check if we should finalize this group based on time difference
                if isinstance(parsed_time, datetime) and isinstance(last_timestamp, datetime):
                    # Calculate actual time difference in minutes
                    time_diff = (parsed_time - last_timestamp).total_seconds() / 60
                    if time_diff >= target_freq:
                        # Create aggregated bar
                        resampled_row = self._aggregate_group(current_group)
                        resampled_data.append(resampled_row)
                        current_group = []
                else:
                    # Fallback to simplified grouping approach
                    if len(current_group) >= target_freq // 10:  # Simplified grouping
                        # Create aggregated bar
                        resampled_row = self._aggregate_group(current_group)
                        resampled_data.append(resampled_row)
                        current_group = []
                
                last_timestamp = parsed_time
        
        # Add final group if it exists
        if current_group:
            resampled_row = self._aggregate_group(current_group)
            resampled_data.append(resampled_row)
        
        return resampled_data

    def _aggregate_group(self, group: list) -> dict:
        """
        Aggregate a group of data points into a single bar.
        
        Parameters:
        group - List of data points to aggregate
        
        Returns:
        Aggregated data point
        """
        if not group:
            return {}
        
        # Create new row with aggregated values
        first_row = group[0]
        last_row = group[-1]
        
        # For numeric fields, calculate OHLC values
        opens = [float(row.get('open', 0)) for row in group if row.get('open')]
        highs = [float(row.get('high', 0)) for row in group if row.get('high')]
        lows = [float(row.get('low', 0)) for row in group if row.get('low')]
        closes = [float(row.get('close', 0)) for row in group if row.get('close')]
        volumes = [int(row.get('volume', 0)) for row in group if row.get('volume')]
        
        aggregated_row = {
            'timestamp': first_row.get('timestamp', ''),
            'open': opens[0] if opens else 0,
            'high': max(highs) if highs else 0,
            'low': min(lows) if lows else 0,
            'close': closes[-1] if closes else 0,
            'volume': sum(volumes) if volumes else 0
        }
        
        return aggregated_row

    def sample_data_by_volume(self, data: list, 
                             target_volume: int) -> list:
        """
        Sample data to create bars with approximately equal volume.
        
        Parameters:
        data - List to sample
        target_volume - Target volume per bar
        
        Returns:
        Volume-sampled list
        """
        if not data or target_volume <= 0:
            return data
        
        volume_sampled_data = []
        current_bar = None
        current_volume = 0
        
        for row in data:
            volume = int(row.get('volume', 0))
            
            if current_bar is None:
                # Start new bar
                current_bar = row.copy()
                current_volume = volume
            else:
                # Add to current bar
                current_volume += volume
                current_bar['high'] = max(float(current_bar.get('high', 0)), float(row.get('high', 0)))
                current_bar['low'] = min(float(current_bar.get('low', 0)), float(row.get('low', 0)))
                current_bar['close'] = float(row.get('close', 0))
                current_bar['volume'] = current_volume
            
            # Check if we've reached target volume
            if current_volume >= target_volume:
                volume_sampled_data.append(current_bar)
                current_bar = None
                current_volume = 0
        
        # Add final bar if it exists
        if current_bar is not None:
            volume_sampled_data.append(current_bar)
        
        return volume_sampled_data


class DataExporter:
    """
    Exports data to various formats.
    """

    def __init__(self):
        """
        Initialize the data exporter.
        """
        pass

    def export_to_csv(self, data: list, filename: str, 
                     directory: str = "exported_data"):
        """
        Export data to CSV.
        
        Parameters:
        data - List to export
        filename - Name of the file
        directory - Directory to save the file
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Export to CSV
        filepath = os.path.join(directory, filename)
        if data:
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
        print(f"Data exported to {filepath}")

    def export_to_json(self, data: list, filename: str,
                      directory: str = "exported_data"):
        """
        Export data to JSON.
        
        Parameters:
        data - List to export
        filename - Name of the file
        directory - Directory to save the file
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Export to JSON
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Data exported to {filepath}")

    def export_to_database(self, data: list, symbol: str,
                          db_path: str = "exported_data/exported.db"):
        """
        Export data to SQLite database.
        
        Parameters:
        data - List to export
        symbol - Symbol for the data
        db_path - Path to the database
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Export to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Insert data
        for row in data:
            cursor.execute('''
                INSERT OR REPLACE INTO market_data 
                (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                row.get('timestamp', ''),
                row.get('open', 0),
                row.get('high', 0),
                row.get('low', 0),
                row.get('close', 0),
                row.get('volume', 0)
            ))
        
        conn.commit()
        conn.close()
        print(f"Data exported to database {db_path}")


class DataImporter:
    """
    Imports data from various sources.
    """

    def __init__(self):
        """
        Initialize the data importer.
        """
        pass

    def import_from_csv(self, filepath: str) -> list:
        """
        Import data from CSV.
        
        Parameters:
        filepath - Path to the CSV file
        
        Returns:
        List with imported data
        """
        try:
            data = []
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
            print(f"Data imported from {filepath}")
            return data
        except Exception as e:
            print(f"Error importing data from {filepath}: {e}")
            return []

    def import_from_json(self, filepath: str) -> list:
        """
        Import data from JSON.
        
        Parameters:
        filepath - Path to the JSON file
        
        Returns:
        List with imported data
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            print(f"Data imported from {filepath}")
            return data
        except Exception as e:
            print(f"Error importing data from {filepath}: {e}")
            return []

    def import_from_database(self, db_path: str, symbol: str,
                            start_date: str = None, end_date: str = None) -> list:
        """
        Import data from SQLite database.
        
        Parameters:
        db_path - Path to the database
        symbol - Symbol to import
        start_date - Start date filter
        end_date - End date filter
        
        Returns:
        List with imported data
        """
        try:
            conn = sqlite3.connect(db_path)
            
            # Build query
            query = "SELECT * FROM market_data WHERE symbol = ?"
            params = [symbol]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
                
            query += " ORDER BY timestamp"
            
            # Execute query
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            columns = [description[0] for description in cursor.description]
            data = [dict(zip(columns, row)) for row in rows]
            
            conn.close()
            print(f"Data imported from database {db_path}")
            return data
        except Exception as e:
            print(f"Error importing data from database {db_path}: {e}")
            return []


class DataSynthesizer:
    """
    Synthesizes artificial market data for testing.
    """

    def __init__(self):
        """
        Initialize the data synthesizer.
        """
        pass

    def generate_brownian_motion(self, initial_price: float = 100.0,
                                drift: float = 0.0,
                                volatility: float = 0.2,
                                steps: int = 1000,
                                dt: float = 1/252) -> list:
        """
        Generate synthetic price data using geometric Brownian motion.
        
        Parameters:
        initial_price - Initial price
        drift - Drift term (expected return)
        volatility - Volatility (standard deviation of returns)
        steps - Number of time steps
        dt - Time step size
        
        Returns:
        List with synthetic price data
        """
        import random
        
        # Generate random returns
        rand_returns = [random.normalvariate(0, 1) for _ in range(steps)]
        
        # Calculate price path
        prices = [initial_price]
        for i in range(steps):
            price = prices[-1] * math.exp((drift - 0.5 * volatility**2) * dt + 
                                        volatility * math.sqrt(dt) * rand_returns[i])
            prices.append(price)
        
        # Generate timestamps
        start_date = datetime.now()
        data = []
        for i in range(steps):
            timestamp = start_date - timedelta(days=steps-i)
            open_price = prices[i]
            high_price = open_price * (1 + random.uniform(0, 0.02))
            low_price = open_price * (1 - random.uniform(0, 0.02))
            close_price = prices[i+1]
            volume = random.randint(1000, 10000)
            
            data.append({
                'timestamp': timestamp.strftime('%Y-%m-%d'),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
        
        return data

    def generate_mean_reverting(self, mean_price: float = 100.0,
                               speed: float = 0.1,
                               volatility: float = 0.2,
                               steps: int = 1000) -> list:
        """
        Generate synthetic mean-reverting price data.
        
        Parameters:
        mean_price - Long-term mean price
        speed - Speed of mean reversion
        volatility - Volatility
        steps - Number of time steps
        
        Returns:
        List with synthetic price data
        """
        import random
        
        # Generate random returns
        rand_returns = [random.normalvariate(0, 1) for _ in range(steps)]
        
        # Calculate price path
        prices = [mean_price]
        for i in range(steps):
            dx = speed * (mean_price - prices[-1]) + volatility * rand_returns[i]
            price = prices[-1] + dx
            prices.append(price)
        
        # Generate timestamps and data
        start_date = datetime.now()
        data = []
        for i in range(steps):
            timestamp = start_date - timedelta(days=steps-i)
            open_price = prices[i]
            high_price = open_price * (1 + random.uniform(0, 0.02))
            low_price = open_price * (1 - random.uniform(0, 0.02))
            close_price = prices[i+1]
            volume = random.randint(1000, 10000)
            
            data.append({
                'timestamp': timestamp.strftime('%Y-%m-%d'),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
        
        return data


def compare_data_providers(symbol: str, start_date: str, end_date: str,
                          providers: List[DataProvider]) -> Dict:
    """
    Compare data from different providers.
    
    Parameters:
    symbol - Symbol to compare
    start_date - Start date
    end_date - End date
    providers - List of data providers
    
    Returns:
    Dictionary with comparison results
    """
    results = {}
    
    for i, provider in enumerate(providers):
        try:
            data = provider.get_historical_data(symbol, start_date, end_date)
            results[f'provider_{i}'] = {
                'class': provider.__class__.__name__,
                'rows': len(data),
                'start_date': data[0]['timestamp'] if data else None,
                'end_date': data[-1]['timestamp'] if data else None,
                'missing_values': sum(1 for row in data for value in row.values() if value == "" or value is None) if data else 0
            }
        except Exception as e:
            results[f'provider_{i}'] = {
                'class': provider.__class__.__name__,
                'error': str(e)
            }
    
    return results