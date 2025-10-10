"""
Comprehensive utilities module for the pytradepath framework.
This module provides various utility functions and classes used throughout the framework.
"""

import os
import json
import csv
import pickle
import hashlib
import base64
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
import threading
import time
import warnings
import functools


class CacheManager:
    """
    Manages caching of data and results to improve performance.
    """

    def __init__(self, cache_directory: str = "cache", max_size: int = 1000):
        """
        Initialize the cache manager.
        
        Parameters:
        cache_directory - Directory to store cache files
        max_size - Maximum number of items to cache
        """
        self.cache_directory = cache_directory
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.lock = threading.Lock()
        
        # Create cache directory
        os.makedirs(cache_directory, exist_ok=True)

    def get(self, key: str) -> Any:
        """
        Get item from cache.
        
        Parameters:
        key - Cache key
        
        Returns:
        Cached item or None if not found
        """
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            
            # Try to load from file
            filepath = os.path.join(self.cache_directory, f"{key}.pkl")
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        item = pickle.load(f)
                    self.cache[key] = item
                    self.access_times[key] = time.time()
                    return item
                except Exception as e:
                    print(f"Error loading cache item {key}: {e}")
            
            return None

    def set(self, key: str, value: Any):
        """
        Set item in cache.
        
        Parameters:
        key - Cache key
        value - Value to cache
        """
        with self.lock:
            self.cache[key] = value
            self.access_times[key] = time.time()
            
            # Save to file
            filepath = os.path.join(self.cache_directory, f"{key}.pkl")
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(value, f)
            except Exception as e:
                print(f"Error saving cache item {key}: {e}")
            
            # Manage cache size
            self._manage_cache_size()

    def _manage_cache_size(self):
        """Manage cache size by removing old items."""
        if len(self.cache) > self.max_size:
            # Find oldest items
            sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
            items_to_remove = len(self.cache) - self.max_size // 2
            
            for i in range(items_to_remove):
                key = sorted_items[i][0]
                if key in self.cache:
                    del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                
                # Remove file
                filepath = os.path.join(self.cache_directory, f"{key}.pkl")
                if os.path.exists(filepath):
                    os.remove(filepath)

    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            
            # Remove all cache files
            for filename in os.listdir(self.cache_directory):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_directory, filename))

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
        Dictionary with cache statistics
        """
        with self.lock:
            return {
                "cached_items": len(self.cache),
                "max_size": self.max_size,
                "cache_directory": self.cache_directory,
                "oldest_item": min(self.access_times.values()) if self.access_times else None,
                "newest_item": max(self.access_times.values()) if self.access_times else None
            }


class Timer:
    """
    Utility class for timing operations.
    """

    def __init__(self):
        """Initialize the timer."""
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start the timer."""
        self.start_time = time.time()

    def stop(self) -> float:
        """
        Stop the timer.
        
        Returns:
        Elapsed time in seconds
        """
        self.end_time = time.time()
        return self.elapsed_time()

    def elapsed_time(self) -> float:
        """
        Get elapsed time.
        
        Returns:
        Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class MemoryMonitor:
    """
    Monitors memory usage.
    """

    def __init__(self):
        """Initialize the memory monitor."""
        self.initial_memory = self._get_memory_usage()

    def _get_memory_usage(self) -> int:
        """
        Get current memory usage.
        
        Returns:
        Memory usage in bytes
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except ImportError:
            # If psutil is not available, return 0
            warnings.warn("psutil not available, memory monitoring disabled")
            return 0

    def get_memory_increase(self) -> int:
        """
        Get memory increase since initialization.
        
        Returns:
        Memory increase in bytes
        """
        current_memory = self._get_memory_usage()
        return current_memory - self.initial_memory

    def get_memory_usage_mb(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
        Memory usage in MB
        """
        return self._get_memory_usage() / (1024 * 1024)

    def get_memory_increase_mb(self) -> float:
        """
        Get memory increase in MB.
        
        Returns:
        Memory increase in MB
        """
        return self.get_memory_increase() / (1024 * 1024)


class DataConverter:
    """
    Converts data between different formats.
    """

    @staticmethod
    def dict_to_csv(data: List[Dict], filepath: str, 
                   fieldnames: Optional[List[str]] = None):
        """
        Convert list of dictionaries to CSV.
        
        Parameters:
        data - List of dictionaries
        filepath - Output CSV file path
        fieldnames - Field names (optional, inferred from data if not provided)
        """
        if not data:
            raise ValueError("No data to convert")
        
        if fieldnames is None:
            fieldnames = list(data[0].keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    @staticmethod
    def csv_to_dict(filepath: str) -> List[Dict]:
        """
        Convert CSV to list of dictionaries.
        
        Parameters:
        filepath - Input CSV file path
        
        Returns:
        List of dictionaries
        """
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            return list(reader)

    @staticmethod
    def dict_to_json(data: Dict, filepath: str, indent: int = 4):
        """
        Convert dictionary to JSON.
        
        Parameters:
        data - Dictionary to convert
        filepath - Output JSON file path
        indent - JSON indentation
        """
        with open(filepath, 'w', encoding='utf-8') as jsonfile:
            json.dump(data, jsonfile, indent=indent, default=str)

    @staticmethod
    def json_to_dict(filepath: str) -> Dict:
        """
        Convert JSON to dictionary.
        
        Parameters:
        filepath - Input JSON file path
        
        Returns:
        Dictionary
        """
        with open(filepath, 'r', encoding='utf-8') as jsonfile:
            return json.load(jsonfile)


class HashUtility:
    """
    Utility class for hashing data.
    """

    @staticmethod
    def hash_string(text: str, algorithm: str = 'sha256') -> str:
        """
        Hash a string.
        
        Parameters:
        text - String to hash
        algorithm - Hash algorithm (default: sha256)
        
        Returns:
        Hex digest of hash
        """
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(text.encode('utf-8'))
        return hash_obj.hexdigest()

    @staticmethod
    def hash_file(filepath: str, algorithm: str = 'sha256') -> str:
        """
        Hash a file.
        
        Parameters:
        filepath - Path to file to hash
        algorithm - Hash algorithm (default: sha256)
        
        Returns:
        Hex digest of hash
        """
        hash_obj = hashlib.new(algorithm)
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    @staticmethod
    def hash_object(obj: Any, algorithm: str = 'sha256') -> str:
        """
        Hash a Python object.
        
        Parameters:
        obj - Object to hash
        algorithm - Hash algorithm (default: sha256)
        
        Returns:
        Hex digest of hash
        """
        try:
            # Try to serialize the object
            serialized = pickle.dumps(obj)
            hash_obj = hashlib.new(algorithm)
            hash_obj.update(serialized)
            return hash_obj.hexdigest()
        except Exception as e:
            # If serialization fails, hash the string representation
            return HashUtility.hash_string(str(obj), algorithm)


class ThreadSafeCounter:
    """
    Thread-safe counter.
    """

    def __init__(self, initial_value: int = 0):
        """
        Initialize the counter.
        
        Parameters:
        initial_value - Initial counter value
        """
        self._value = initial_value
        self._lock = threading.Lock()

    def increment(self, amount: int = 1) -> int:
        """
        Increment the counter.
        
        Parameters:
        amount - Amount to increment by
        
        Returns:
        New counter value
        """
        with self._lock:
            self._value += amount
            return self._value

    def decrement(self, amount: int = 1) -> int:
        """
        Decrement the counter.
        
        Parameters:
        amount - Amount to decrement by
        
        Returns:
        New counter value
        """
        with self._lock:
            self._value -= amount
            return self._value

    def get_value(self) -> int:
        """
        Get current counter value.
        
        Returns:
        Current counter value
        """
        with self._lock:
            return self._value

    def set_value(self, value: int):
        """
        Set counter value.
        
        Parameters:
        value - New counter value
        """
        with self._lock:
            self._value = value


class RateLimiter:
    """
    Limits the rate of operations.
    """

    def __init__(self, max_calls: int, time_window: float = 1.0):
        """
        Initialize the rate limiter.
        
        Parameters:
        max_calls - Maximum number of calls per time window
        time_window - Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = threading.Lock()

    def acquire(self) -> bool:
        """
        Acquire permission to make a call.
        
        Returns:
        True if call is allowed, False if rate limit exceeded
        """
        with self.lock:
            now = time.time()
            
            # Remove old calls
            self.calls = [call_time for call_time in self.calls 
                         if now - call_time < self.time_window]
            
            # Check if we can make another call
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            else:
                return False

    def wait_for_permission(self):
        """Wait until permission is granted."""
        while not self.acquire():
            time.sleep(0.1)


class RetryDecorator:
    """
    Decorator for retrying failed operations.
    """

    def __init__(self, max_attempts: int = 3, delay: float = 1.0,
                 backoff: float = 2.0, exceptions: tuple = (Exception,)):
        """
        Initialize the retry decorator.
        
        Parameters:
        max_attempts - Maximum number of attempts
        delay - Initial delay between attempts in seconds
        backoff - Backoff multiplier for delay
        exceptions - Tuple of exceptions to catch
        """
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff = backoff
        self.exceptions = exceptions

    def __call__(self, func: Callable) -> Callable:
        """
        Apply the retry decorator.
        
        Parameters:
        func - Function to decorate
        
        Returns:
        Decorated function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = self.delay
            
            for attempt in range(self.max_attempts):
                try:
                    return func(*args, **kwargs)
                except self.exceptions as e:
                    if attempt == self.max_attempts - 1:
                        # Last attempt, re-raise the exception
                        raise e
                    
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= self.backoff
            
        return wrapper


class SingletonMeta(type):
    """
    Metaclass for implementing the Singleton pattern.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class ConfigurationManager(metaclass=SingletonMeta):
    """
    Singleton configuration manager.
    """

    def __init__(self):
        """Initialize the configuration manager."""
        self.config = {}
        self._lock = threading.Lock()

    def set_config(self, key: str, value: Any):
        """
        Set configuration value.
        
        Parameters:
        key - Configuration key
        value - Configuration value
        """
        with self._lock:
            self.config[key] = value

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Parameters:
        key - Configuration key
        default - Default value if key not found
        
        Returns:
        Configuration value
        """
        with self._lock:
            return self.config.get(key, default)

    def load_config_file(self, filepath: str):
        """
        Load configuration from file.
        
        Parameters:
        filepath - Path to configuration file
        """
        with self._lock:
            try:
                with open(filepath, 'r') as f:
                    if filepath.endswith('.json'):
                        self.config.update(json.load(f))
                    else:
                        # Assume it's a Python file
                        # This is a simplified implementation
                        warnings.warn("Python config file loading is not fully implemented")
            except Exception as e:
                print(f"Error loading config file {filepath}: {e}")

    def save_config_file(self, filepath: str):
        """
        Save configuration to file.
        
        Parameters:
        filepath - Path to configuration file
        """
        with self._lock:
            try:
                with open(filepath, 'w') as f:
                    if filepath.endswith('.json'):
                        json.dump(self.config, f, indent=4)
                    else:
                        # Assume it's a Python file
                        # This is a simplified implementation
                        warnings.warn("Python config file saving is not fully implemented")
            except Exception as e:
                print(f"Error saving config file {filepath}: {e}")


class PerformanceMonitor:
    """
    Monitors performance metrics.
    """

    def __init__(self):
        """Initialize the performance monitor."""
        self.metrics = {}
        self.lock = threading.Lock()

    def record_metric(self, name: str, value: float):
        """
        Record a performance metric.
        
        Parameters:
        name - Metric name
        value - Metric value
        """
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)

    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a metric.
        
        Parameters:
        name - Metric name
        
        Returns:
        Dictionary with metric statistics
        """
        with self.lock:
            if name not in self.metrics or not self.metrics[name]:
                return {}
            
            values = self.metrics[name]
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "median": sorted(values)[len(values) // 2]
            }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all metrics.
        
        Returns:
        Dictionary with all metric statistics
        """
        with self.lock:
            return {name: self.get_metric_stats(name) for name in self.metrics}


def create_temp_directory(prefix: str = "pytradepath_") -> str:
    """
    Create a temporary directory.
    
    Parameters:
    prefix - Directory name prefix
    
    Returns:
    Path to temporary directory
    """
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    return temp_dir


def format_time(seconds: float) -> str:
    """
    Format time in a human-readable format.
    
    Parameters:
    seconds - Time in seconds
    
    Returns:
    Formatted time string
    """
    if seconds < 1:
        return f"{seconds*1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes in a human-readable format.
    
    Parameters:
    bytes_value - Number of bytes
    
    Returns:
    Formatted bytes string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f}PB"


def safe_divide(numerator: float, denominator: float, 
               default: float = 0.0) -> float:
    """
    Safely divide two numbers.
    
    Parameters:
    numerator - Numerator
    denominator - Denominator
    default - Default value if denominator is zero
    
    Returns:
    Division result or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp a value between minimum and maximum values.
    
    Parameters:
    value - Value to clamp
    min_value - Minimum value
    max_value - Maximum value
    
    Returns:
    Clamped value
    """
    return max(min_value, min(value, max_value))


def normalize(value: float, min_value: float, max_value: float) -> float:
    """
    Normalize a value to the range [0, 1].
    
    Parameters:
    value - Value to normalize
    min_value - Minimum value
    max_value - Maximum value
    
    Returns:
    Normalized value
    """
    if max_value == min_value:
        return 0.0
    return (value - min_value) / (max_value - min_value)


def lerp(start: float, end: float, t: float) -> float:
    """
    Linear interpolation between two values.
    
    Parameters:
    start - Start value
    end - End value
    t - Interpolation factor (0.0 to 1.0)
    
    Returns:
        Interpolated value
    """
    return start + (end - start) * t


# Example usage and testing
if __name__ == "__main__":
    # Test cache manager
    print("Testing CacheManager...")
    cache = CacheManager()
    cache.set("test_key", "test_value")
    retrieved = cache.get("test_key")
    print(f"Retrieved from cache: {retrieved}")
    
    # Test timer
    print("\nTesting Timer...")
    with Timer() as timer:
        time.sleep(0.1)
    print(f"Elapsed time: {timer.elapsed_time():.4f} seconds")
    
    # Test data converter
    print("\nTesting DataConverter...")
    test_data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    DataConverter.dict_to_csv(test_data, "test.csv")
    loaded_data = DataConverter.csv_to_dict("test.csv")
    print(f"Loaded CSV data: {loaded_data}")
    
    # Clean up test file
    if os.path.exists("test.csv"):
        os.remove("test.csv")
    
    # Test hash utility
    print("\nTesting HashUtility...")
    test_string = "Hello, World!"
    hash_value = HashUtility.hash_string(test_string)
    print(f"Hash of '{test_string}': {hash_value}")
    
    # Test thread-safe counter
    print("\nTesting ThreadSafeCounter...")
    counter = ThreadSafeCounter(10)
    counter.increment(5)
    print(f"Counter value: {counter.get_value()}")
    
    # Test rate limiter
    print("\nTesting RateLimiter...")
    limiter = RateLimiter(max_calls=3, time_window=1.0)
    for i in range(5):
        if limiter.acquire():
            print(f"Call {i+1}: Allowed")
        else:
            print(f"Call {i+1}: Rate limited")
    
    # Test retry decorator
    print("\nTesting RetryDecorator...")
    @RetryDecorator(max_attempts=3, delay=0.1)
    def unreliable_function():
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise Exception("Random failure")
        return "Success"
    
    try:
        result = unreliable_function()
        print(f"Function result: {result}")
    except Exception as e:
        print(f"Function failed after retries: {e}")
    
    # Test configuration manager
    print("\nTesting ConfigurationManager...")
    config_manager = ConfigurationManager()
    config_manager.set_config("test_key", "test_value")
    retrieved_config = config_manager.get_config("test_key")
    print(f"Retrieved config: {retrieved_config}")
    
    # Test performance monitor
    print("\nTesting PerformanceMonitor...")
    perf_monitor = PerformanceMonitor()
    perf_monitor.record_metric("test_metric", 1.0)
    perf_monitor.record_metric("test_metric", 2.0)
    perf_monitor.record_metric("test_metric", 3.0)
    stats = perf_monitor.get_metric_stats("test_metric")
    print(f"Metric stats: {stats}")
    
    print("\nAll utility tests completed!")