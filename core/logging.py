"""
Logging module for the pytradepath framework.
This module provides comprehensive logging capabilities.
"""

import logging
import logging.handlers
import os
import sys
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import traceback
from .config import ConfigManager, LoggingConfig


class Logger:
    """
    Comprehensive logging system for the trading framework.
    """

    def __init__(self, name: str = "pytradepath", config: LoggingConfig = None):
        """
        Initialize the logger.
        
        Parameters:
        name - Name of the logger
        config - Logging configuration
        """
        self.name = name
        self.config = config or LoggingConfig()
        self.logger = logging.getLogger(name)
        self._setup_logger()

    def _setup_logger(self):
        """Set up the logger with handlers and formatters."""
        # Set logging level
        level = getattr(logging, self.config.level.upper(), logging.INFO)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(self.config.format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if enabled)
        if self.config.file_logging:
            # Create log directory
            os.makedirs(self.config.log_directory, exist_ok=True)
            
            # Create file handler with rotation
            log_file = os.path.join(self.config.log_directory, f"{self.name}.log")
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def debug(self, message: str, **kwargs):
        """
        Log debug message.
        
        Parameters:
        message - Message to log
        kwargs - Additional context data
        """
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """
        Log info message.
        
        Parameters:
        message - Message to log
        kwargs - Additional context data
        """
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """
        Log warning message.
        
        Parameters:
        message - Message to log
        kwargs - Additional context data
        """
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """
        Log error message.
        
        Parameters:
        message - Message to log
        kwargs - Additional context data
        """
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """
        Log critical message.
        
        Parameters:
        message - Message to log
        kwargs - Additional context data
        """
        self._log(logging.CRITICAL, message, **kwargs)

    def exception(self, message: str, **kwargs):
        """
        Log exception with traceback.
        
        Parameters:
        message - Message to log
        kwargs - Additional context data
        """
        self._log(logging.ERROR, message, exc_info=True, **kwargs)

    def _log(self, level: int, message: str, **kwargs):
        """
        Internal logging method.
        
        Parameters:
        level - Logging level
        message - Message to log
        kwargs - Additional context data
        """
        # Add timestamp and context to message
        timestamp = datetime.now().isoformat()
        context = self._format_context(kwargs)
        
        if context:
            full_message = f"[{timestamp}] {message} | Context: {context}"
        else:
            full_message = f"[{timestamp}] {message}"
        
        # Log message
        self.logger.log(level, full_message)

    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        Format context data for logging.
        
        Parameters:
        context - Context data
        
        Returns:
        Formatted context string
        """
        if not context:
            return ""
        
        try:
            return json.dumps(context, default=str)
        except Exception:
            # If JSON serialization fails, use string representation
            return str(context)

    def log_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        Log a structured event.
        
        Parameters:
        event_type - Type of event
        event_data - Event data
        """
        event = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": event_data
        }
        
        self.info(f"EVENT: {event_type}", **event_data)

    def log_trade(self, symbol: str, quantity: float, price: float, 
                 direction: str, commission: float = 0.0):
        """
        Log a trade execution.
        
        Parameters:
        symbol - Trading symbol
        quantity - Trade quantity
        price - Execution price
        direction - Trade direction (BUY/SELL)
        commission - Commission paid
        """
        trade_data = {
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "direction": direction,
            "commission": commission,
            "value": quantity * price
        }
        
        self.info(f"TRADE_EXECUTED: {direction} {quantity} {symbol} @ {price}", **trade_data)

    def log_signal(self, symbol: str, signal_type: str, strength: float):
        """
        Log a trading signal.
        
        Parameters:
        symbol - Trading symbol
        signal_type - Type of signal (BUY/SELL/EXIT)
        strength - Signal strength (0.0 to 1.0)
        """
        signal_data = {
            "symbol": symbol,
            "signal_type": signal_type,
            "strength": strength
        }
        
        self.info(f"SIGNAL_GENERATED: {signal_type} {symbol} (strength: {strength})", **signal_data)

    def log_performance(self, metric_name: str, value: float, 
                       threshold: Optional[float] = None):
        """
        Log a performance metric.
        
        Parameters:
        metric_name - Name of the metric
        value - Metric value
        threshold - Threshold value for alerts
        """
        perf_data = {
            "metric": metric_name,
            "value": value
        }
        
        if threshold is not None:
            perf_data["threshold"] = threshold
            if value < threshold:
                self.warning(f"PERFORMANCE_ALERT: {metric_name} below threshold", **perf_data)
            else:
                self.info(f"PERFORMANCE_METRIC: {metric_name}", **perf_data)
        else:
            self.info(f"PERFORMANCE_METRIC: {metric_name}", **perf_data)

    def log_risk_violation(self, violation_type: str, details: Dict[str, Any]):
        """
        Log a risk management violation.
        
        Parameters:
        violation_type - Type of violation
        details - Violation details
        """
        violation_data = {
            "violation_type": violation_type,
            "details": details
        }
        
        self.warning(f"RISK_VIOLATION: {violation_type}", **details)

    def log_system_status(self, component: str, status: str, details: Dict[str, Any] = None):
        """
        Log system component status.
        
        Parameters:
        component - Component name
        status - Component status
        details - Additional details
        """
        status_data = {
            "component": component,
            "status": status
        }
        
        if details:
            status_data.update(details)
        
        self.info(f"SYSTEM_STATUS: {component} is {status}", **status_data)


class PerformanceLogger:
    """
    Specialized logger for performance monitoring.
    """

    def __init__(self, logger: Logger, config_manager: ConfigManager = None):
        """
        Initialize the performance logger.
        
        Parameters:
        logger - Base logger
        config_manager - Configuration manager
        """
        self.logger = logger
        self.config = config_manager.get_config().performance if config_manager else None
        self.metrics = {}
        
        # Create performance directory if enabled
        if self.config and self.config.track_execution_time:
            os.makedirs(self.config.performance_directory, exist_ok=True)

    def start_timer(self, operation_name: str):
        """
        Start timing an operation.
        
        Parameters:
        operation_name - Name of the operation
        """
        if self.config and self.config.track_execution_time:
            self.metrics[operation_name] = {
                "start_time": datetime.now(),
                "end_time": None,
                "duration": None
            }

    def end_timer(self, operation_name: str):
        """
        End timing an operation.
        
        Parameters:
        operation_name - Name of the operation
        """
        if (self.config and self.config.track_execution_time and 
            operation_name in self.metrics):
            end_time = datetime.now()
            start_time = self.metrics[operation_name]["start_time"]
            duration = (end_time - start_time).total_seconds()
            
            self.metrics[operation_name].update({
                "end_time": end_time,
                "duration": duration
            })
            
            # Log performance
            self.logger.log_performance(f"{operation_name}_duration", duration)

    def log_memory_usage(self, usage_mb: float):
        """
        Log memory usage.
        
        Parameters:
        usage_mb - Memory usage in MB
        """
        if self.config and self.config.track_memory_usage:
            self.logger.log_performance("memory_usage_mb", usage_mb)

    def log_cpu_usage(self, usage_percent: float):
        """
        Log CPU usage.
        
        Parameters:
        usage_percent - CPU usage percentage
        """
        if self.config and self.config.track_cpu_usage:
            self.logger.log_performance("cpu_usage_percent", usage_percent)

    def save_performance_metrics(self, filename: str = None):
        """
        Save performance metrics to file.
        
        Parameters:
        filename - Name of the file (default: auto-generated)
        """
        if not self.config or not self.config.track_execution_time:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_metrics_{timestamp}.json"
        
        filepath = os.path.join(self.config.performance_directory, filename)
        
        # Save metrics
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=4, default=str)
        
        self.logger.info(f"Performance metrics saved to {filepath}")


class AuditLogger:
    """
    Logger for audit trails and compliance.
    """

    def __init__(self, logger: Logger, audit_directory: str = "audit"):
        """
        Initialize the audit logger.
        
        Parameters:
        logger - Base logger
        audit_directory - Directory for audit logs
        """
        self.logger = logger
        self.audit_directory = audit_directory
        os.makedirs(audit_directory, exist_ok=True)

    def log_user_action(self, user: str, action: str, details: Dict[str, Any] = None):
        """
        Log user action for audit trail.
        
        Parameters:
        user - User identifier
        action - Action performed
        details - Action details
        """
        audit_data = {
            "user": user,
            "action": action,
            "timestamp": datetime.now().isoformat()
        }
        
        if details:
            audit_data.update(details)
        
        # Log to main logger
        self.logger.info(f"AUDIT: User {user} performed {action}", **audit_data)
        
        # Save to audit file
        self._save_audit_record(audit_data)

    def log_system_change(self, component: str, change_type: str, 
                         old_value: Any, new_value: Any, user: str = "system"):
        """
        Log system configuration change.
        
        Parameters:
        component - Component changed
        change_type - Type of change
        old_value - Old value
        new_value - New value
        user - User making the change
        """
        change_data = {
            "user": user,
            "component": component,
            "change_type": change_type,
            "old_value": old_value,
            "new_value": new_value,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log to main logger
        self.logger.info(f"AUDIT: System change in {component}", **change_data)
        
        # Save to audit file
        self._save_audit_record(change_data)

    def log_security_event(self, event_type: str, severity: str, 
                          details: Dict[str, Any] = None):
        """
        Log security-related event.
        
        Parameters:
        event_type - Type of security event
        severity - Event severity (LOW/MEDIUM/HIGH/CRITICAL)
        details - Event details
        """
        security_data = {
            "event_type": event_type,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        }
        
        if details:
            security_data.update(details)
        
        # Log to main logger
        level = logging.WARNING if severity in ["LOW", "MEDIUM"] else logging.ERROR
        self.logger.log(level, f"SECURITY: {event_type} ({severity})", **security_data)
        
        # Save to audit file
        self._save_audit_record(security_data)

    def _save_audit_record(self, record: Dict[str, Any]):
        """
        Save audit record to file.
        
        Parameters:
        record - Audit record to save
        """
        # Create daily audit file
        date_str = datetime.now().strftime("%Y%m%d")
        audit_file = os.path.join(self.audit_directory, f"audit_{date_str}.jsonl")
        
        # Append record to file
        with open(audit_file, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')


class ErrorTracker:
    """
    Tracks and analyzes errors and exceptions.
    """

    def __init__(self, logger: Logger, max_errors: int = 1000):
        """
        Initialize the error tracker.
        
        Parameters:
        logger - Base logger
        max_errors - Maximum number of errors to track
        """
        self.logger = logger
        self.max_errors = max_errors
        self.errors = []
        self.error_counts = {}

    def track_error(self, error_type: str, message: str, 
                   traceback_str: str = None, context: Dict[str, Any] = None):
        """
        Track an error.
        
        Parameters:
        error_type - Type of error
        message - Error message
        traceback_str - Error traceback
        context - Error context
        """
        error_record = {
            "error_type": error_type,
            "message": message,
            "traceback": traceback_str,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to error list
        self.errors.append(error_record)
        
        # Keep only recent errors
        if len(self.errors) > self.max_errors:
            self.errors.pop(0)
        
        # Update error counts
        if error_type in self.error_counts:
            self.error_counts[error_type] += 1
        else:
            self.error_counts[error_type] = 1
        
        # Log error
        self.logger.error(f"ERROR_TRACKED: {error_type} - {message}", **context)

    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics.
        
        Returns:
        Dictionary with error statistics
        """
        return {
            "total_errors": len(self.errors),
            "error_counts": self.error_counts,
            "most_common_error": max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None
        }

    def get_recent_errors(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent errors.
        
        Parameters:
        count - Number of recent errors to return
        
        Returns:
        List of recent errors
        """
        return self.errors[-count:]

    def clear_errors(self):
        """Clear tracked errors."""
        self.errors.clear()
        self.error_counts.clear()


def setup_global_logger(config_manager: ConfigManager = None) -> Logger:
    """
    Set up global logger for the application.
    
    Parameters:
    config_manager - Configuration manager
    
    Returns:
    Configured logger
    """
    # Get logging configuration
    if config_manager:
        logging_config = config_manager.get_config().logging
    else:
        logging_config = LoggingConfig()
    
    # Create and configure logger
    logger = Logger("pytradepath", logging_config)
    
    # Set as global logger
    logging.getLogger().handlers = logger.logger.handlers
    logging.getLogger().setLevel(logger.logger.level)
    
    return logger


def get_logger(name: str = "pytradepath") -> Logger:
    """
    Get a logger instance.
    
    Parameters:
    name - Name of the logger
    
    Returns:
    Logger instance
    """
    return Logger(name)


# Global logger instance
_global_logger = None


def get_global_logger() -> Logger:
    """
    Get the global logger instance.
    
    Returns:
    Global logger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_global_logger()
    return _global_logger


def log_function_call(func):
    """
    Decorator to log function calls.
    
    Parameters:
    func - Function to decorate
    """
    def wrapper(*args, **kwargs):
        logger = get_global_logger()
        logger.debug(f"Calling function {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Function {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.exception(f"Function {func.__name__} failed", 
                           exception=str(e), 
                           args=str(args), 
                           kwargs=str(kwargs))
            raise
    
    return wrapper


class LogAggregator:
    """
    Aggregates and analyzes log data.
    """

    def __init__(self, log_directory: str = "logs"):
        """
        Initialize the log aggregator.
        
        Parameters:
        log_directory - Directory containing log files
        """
        self.log_directory = log_directory

    def analyze_log_patterns(self, hours: int = 24) -> Dict[str, Any]:
        """
        Analyze log patterns over a time period.
        
        Parameters:
        hours - Number of hours to analyze
        
        Returns:
        Dictionary with analysis results
        """
        # This is a simplified implementation
        # In practice, you would parse log files and analyze patterns
        return {
            "time_period_hours": hours,
            "total_log_entries": 0,
            "error_rate": 0.0,
            "warning_rate": 0.0,
            "most_common_messages": []
        }

    def generate_log_summary(self) -> Dict[str, Any]:
        """
        Generate summary of log data.
        
        Returns:
        Dictionary with log summary
        """
        # This is a simplified implementation
        return {
            "log_files_analyzed": 0,
            "total_entries": 0,
            "error_count": 0,
            "warning_count": 0,
            "info_count": 0
        }