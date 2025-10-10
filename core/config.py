"""
Configuration module for the pytradepath framework.
This module provides configuration management capabilities.
"""

import json
import os
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict


@dataclass
class DataConfig:
    """Configuration for data handling."""
    data_directory: str = "data"
    supported_formats: List[str] = None
    csv_delimiter: str = ","
    missing_data_strategy: str = "forward_fill"
    validate_data: bool = True
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ["csv", "parquet", "json"]


@dataclass
class StrategyConfig:
    """Configuration for trading strategies."""
    max_positions: int = 10
    position_sizing_method: str = "fixed"
    risk_per_trade: float = 0.01  # 1% of portfolio
    max_correlation: float = 0.8
    diversification_required: bool = True


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    max_drawdown: float = 0.2  # 20%
    max_daily_loss: float = 0.05  # 5%
    stop_loss_percent: float = 0.05  # 5%
    take_profit_percent: float = 0.1  # 10%
    position_limits: Dict[str, float] = None
    sector_exposure_limit: float = 0.3  # 30%
    
    def __post_init__(self):
        if self.position_limits is None:
            self.position_limits = {}


@dataclass
class ExecutionConfig:
    """Configuration for order execution."""
    slippage_factor: float = 0.0001  # 0.01%
    commission_rate: float = 0.001  # 0.1%
    execution_delay: float = 0.0  # seconds
    market_impact_model: str = "linear"
    fill_probability: float = 1.0  # 100%


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    frequency: str = "1D"
    transaction_cost: float = 0.001  # 0.1%
    include_dividends: bool = False
    include_splits: bool = False
    save_results: bool = True
    results_directory: str = "results"


@dataclass
class LiveConfig:
    """Configuration for live trading."""
    broker: str = "paper"
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    paper_balance: float = 100000.0
    enable_paper_trading: bool = True
    enable_live_trading: bool = False
    order_confirmation_required: bool = False
    risk_management_enabled: bool = True


@dataclass
class ReportingConfig:
    """Configuration for reporting."""
    generate_html_report: bool = True
    generate_pdf_report: bool = False
    report_directory: str = "reports"
    plot_dpi: int = 150
    plot_style: str = "default"
    save_metrics: bool = True
    metrics_format: str = "json"


@dataclass
class MLConfig:
    """Configuration for machine learning."""
    model_type: str = "linear_regression"
    train_test_split: float = 0.8
    cross_validation_folds: int = 5
    feature_scaling: bool = True
    hyperparameter_tuning: bool = True
    ensemble_methods: List[str] = None
    
    def __post_init__(self):
        if self.ensemble_methods is None:
            self.ensemble_methods = ["bagging"]


@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization."""
    method: str = "grid_search"
    max_iterations: int = 100
    population_size: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    parallel_execution: bool = True
    save_optimization_results: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_logging: bool = True
    log_directory: str = "logs"
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring."""
    track_memory_usage: bool = True
    track_cpu_usage: bool = True
    track_execution_time: bool = True
    performance_directory: str = "performance"
    sampling_frequency: float = 1.0  # seconds


@dataclass
class SystemConfig:
    """Main system configuration."""
    data: DataConfig
    strategy: StrategyConfig
    risk: RiskConfig
    execution: ExecutionConfig
    backtest: BacktestConfig
    live: LiveConfig
    reporting: ReportingConfig
    ml: MLConfig
    optimization: OptimizationConfig
    logging: LoggingConfig
    performance: PerformanceConfig
    
    # Global settings
    timezone: str = "UTC"
    currency: str = "USD"
    base_url: str = "https://api.example.com"
    cache_enabled: bool = True
    cache_directory: str = "cache"
    temp_directory: str = "temp"


class ConfigManager:
    """Manages configuration loading, saving, and validation."""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Parameters:
        config_file - Path to configuration file (optional)
        """
        self.config_file = config_file
        self.config = self._create_default_config()
        
        if config_file:
            self.load_config(config_file)

    def _create_default_config(self) -> SystemConfig:
        """Create default configuration."""
        return SystemConfig(
            data=DataConfig(),
            strategy=StrategyConfig(),
            risk=RiskConfig(),
            execution=ExecutionConfig(),
            backtest=BacktestConfig(),
            live=LiveConfig(),
            reporting=ReportingConfig(),
            ml=MLConfig(),
            optimization=OptimizationConfig(),
            logging=LoggingConfig(),
            performance=PerformanceConfig()
        )

    def load_config(self, config_file: str):
        """
        Load configuration from file.
        
        Parameters:
        config_file - Path to configuration file
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        # Load configuration
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        # Update configuration
        self._update_config_from_dict(config_dict)
        self.config_file = config_file

    def save_config(self, config_file: str = None):
        """
        Save configuration to file.
        
        Parameters:
        config_file - Path to configuration file (uses loaded file if None)
        """
        if config_file is None:
            config_file = self.config_file
        
        if config_file is None:
            raise ValueError("No configuration file specified")
        
        # Convert configuration to dictionary
        config_dict = self._config_to_dict(self.config)
        
        # Save configuration
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        self.config_file = config_file

    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """
        Update configuration from dictionary.
        
        Parameters:
        config_dict - Dictionary with configuration values
        """
        # Update each section
        for section_name, section_dict in config_dict.items():
            if hasattr(self.config, section_name):
                section = getattr(self.config, section_name)
                if hasattr(section, '__dataclass_fields__'):
                    # Update dataclass fields
                    for field_name, field_value in section_dict.items():
                        if hasattr(section, field_name):
                            setattr(section, field_name, field_value)
                else:
                    # Update simple attribute
                    setattr(self.config, section_name, section_dict)

    def _config_to_dict(self, config) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Parameters:
        config - Configuration object
        
        Returns:
        Dictionary representation of configuration
        """
        if hasattr(config, '__dataclass_fields__'):
            # Convert dataclass to dictionary
            return asdict(config)
        elif isinstance(config, dict):
            # Convert dictionary, recursively converting values
            return {k: self._config_to_dict(v) for k, v in config.items()}
        else:
            # Return as-is
            return config

    def get_config(self) -> SystemConfig:
        """
        Get current configuration.
        
        Returns:
        Current configuration
        """
        return self.config

    def set_config(self, config: SystemConfig):
        """
        Set configuration.
        
        Parameters:
        config - New configuration
        """
        self.config = config

    def get_section(self, section_name: str) -> Any:
        """
        Get configuration section.
        
        Parameters:
        section_name - Name of section to get
        
        Returns:
        Configuration section
        """
        if hasattr(self.config, section_name):
            return getattr(self.config, section_name)
        else:
            raise AttributeError(f"Configuration section '{section_name}' not found")

    def set_section(self, section_name: str, section_config: Any):
        """
        Set configuration section.
        
        Parameters:
        section_name - Name of section to set
        section_config - New section configuration
        """
        if hasattr(self.config, section_name):
            setattr(self.config, section_name, section_config)
        else:
            raise AttributeError(f"Configuration section '{section_name}' not found")

    def validate_config(self) -> List[str]:
        """
        Validate configuration.
        
        Returns:
        List of validation errors
        """
        errors = []
        
        # Validate risk configuration
        if self.config.risk.max_drawdown <= 0 or self.config.risk.max_drawdown > 1:
            errors.append("Risk max_drawdown must be between 0 and 1")
        
        if self.config.risk.max_daily_loss <= 0 or self.config.risk.max_daily_loss > 1:
            errors.append("Risk max_daily_loss must be between 0 and 1")
        
        # Validate execution configuration
        if self.config.execution.commission_rate < 0:
            errors.append("Execution commission_rate must be non-negative")
        
        if self.config.execution.slippage_factor < 0:
            errors.append("Execution slippage_factor must be non-negative")
        
        # Validate backtest configuration
        if self.config.backtest.initial_capital <= 0:
            errors.append("Backtest initial_capital must be positive")
        
        # Validate ML configuration
        if self.config.ml.train_test_split <= 0 or self.config.ml.train_test_split >= 1:
            errors.append("ML train_test_split must be between 0 and 1")
        
        return errors


def create_default_config_file(filename: str = "config.json"):
    """
    Create a default configuration file.
    
    Parameters:
    filename - Name of the configuration file
    """
    # Create default configuration
    config_manager = ConfigManager()
    
    # Save configuration
    config_manager.save_config(filename)
    print(f"Default configuration saved to {filename}")


class Configurable:
    """
    Base class for configurable components.
    """

    def __init__(self, config: SystemConfig = None):
        """
        Initialize configurable component.
        
        Parameters:
        config - System configuration
        """
        self.config = config or ConfigManager().get_config()

    def update_config(self, config: SystemConfig):
        """
        Update configuration.
        
        Parameters:
        config - New configuration
        """
        self.config = config


def merge_configs(config1: SystemConfig, config2: SystemConfig) -> SystemConfig:
    """
    Merge two configurations, with config2 overriding config1.
    
    Parameters:
    config1 - Base configuration
    config2 - Configuration with overrides
    
    Returns:
    Merged configuration
    """
    # Convert to dictionaries
    dict1 = asdict(config1)
    dict2 = asdict(config2)
    
    # Merge dictionaries recursively
    merged_dict = _merge_dicts(dict1, dict2)
    
    # Convert back to SystemConfig
    return _dict_to_system_config(merged_dict)


def _merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Recursively merge two dictionaries.
    
    Parameters:
    dict1 - Base dictionary
    dict2 - Dictionary with overrides
    
    Returns:
    Merged dictionary
    """
    merged = dict1.copy()
    
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def _dict_to_system_config(config_dict: Dict) -> SystemConfig:
    """
    Convert dictionary to SystemConfig.
    
    Parameters:
    config_dict - Dictionary with configuration
    
    Returns:
    SystemConfig object
    """
    # Convert nested dictionaries to dataclasses
    data_config = DataConfig(**config_dict.get('data', {}))
    strategy_config = StrategyConfig(**config_dict.get('strategy', {}))
    risk_config = RiskConfig(**config_dict.get('risk', {}))
    execution_config = ExecutionConfig(**config_dict.get('execution', {}))
    backtest_config = BacktestConfig(**config_dict.get('backtest', {}))
    live_config = LiveConfig(**config_dict.get('live', {}))
    reporting_config = ReportingConfig(**config_dict.get('reporting', {}))
    ml_config = MLConfig(**config_dict.get('ml', {}))
    optimization_config = OptimizationConfig(**config_dict.get('optimization', {}))
    logging_config = LoggingConfig(**config_dict.get('logging', {}))
    performance_config = PerformanceConfig(**config_dict.get('performance', {}))
    
    return SystemConfig(
        data=data_config,
        strategy=strategy_config,
        risk=risk_config,
        execution=execution_config,
        backtest=backtest_config,
        live=live_config,
        reporting=reporting_config,
        ml=ml_config,
        optimization=optimization_config,
        logging=logging_config,
        performance=performance_config,
        timezone=config_dict.get('timezone', 'UTC'),
        currency=config_dict.get('currency', 'USD'),
        base_url=config_dict.get('base_url', 'https://api.example.com'),
        cache_enabled=config_dict.get('cache_enabled', True),
        cache_directory=config_dict.get('cache_directory', 'cache'),
        temp_directory=config_dict.get('temp_directory', 'temp')
    )


# Default configuration values
DEFAULT_CONFIG = {
    "data": {
        "data_directory": "data",
        "supported_formats": ["csv", "parquet", "json"],
        "csv_delimiter": ",",
        "missing_data_strategy": "forward_fill",
        "validate_data": True
    },
    "strategy": {
        "max_positions": 10,
        "position_sizing_method": "fixed",
        "risk_per_trade": 0.01,
        "max_correlation": 0.8,
        "diversification_required": True
    },
    "risk": {
        "max_drawdown": 0.2,
        "max_daily_loss": 0.05,
        "stop_loss_percent": 0.05,
        "take_profit_percent": 0.1,
        "position_limits": {},
        "sector_exposure_limit": 0.3
    },
    "execution": {
        "slippage_factor": 0.0001,
        "commission_rate": 0.001,
        "execution_delay": 0.0,
        "market_impact_model": "linear",
        "fill_probability": 1.0
    },
    "backtest": {
        "initial_capital": 100000.0,
        "frequency": "1D",
        "transaction_cost": 0.001,
        "include_dividends": False,
        "include_splits": False,
        "save_results": True,
        "results_directory": "results"
    },
    "live": {
        "broker": "paper",
        "paper_balance": 100000.0,
        "enable_paper_trading": True,
        "enable_live_trading": False,
        "order_confirmation_required": False,
        "risk_management_enabled": True
    },
    "reporting": {
        "generate_html_report": True,
        "generate_pdf_report": False,
        "report_directory": "reports",
        "plot_dpi": 150,
        "plot_style": "default",
        "save_metrics": True,
        "metrics_format": "json"
    },
    "ml": {
        "model_type": "linear_regression",
        "train_test_split": 0.8,
        "cross_validation_folds": 5,
        "feature_scaling": True,
        "hyperparameter_tuning": True,
        "ensemble_methods": ["bagging"]
    },
    "optimization": {
        "method": "grid_search",
        "max_iterations": 100,
        "population_size": 20,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8,
        "parallel_execution": True,
        "save_optimization_results": True
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_logging": True,
        "log_directory": "logs",
        "max_file_size": 10485760,
        "backup_count": 5
    },
    "performance": {
        "track_memory_usage": True,
        "track_cpu_usage": True,
        "track_execution_time": True,
        "performance_directory": "performance",
        "sampling_frequency": 1.0
    },
    "timezone": "UTC",
    "currency": "USD",
    "base_url": "https://api.example.com",
    "cache_enabled": True,
    "cache_directory": "cache",
    "temp_directory": "temp"
}