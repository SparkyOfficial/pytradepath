# PyTradePath - Algorithmic Trading Framework

PyTradePath is a comprehensive, self-contained framework for backtesting and algorithmic trading. It provides all the necessary components to develop, test, and analyze trading strategies without any external dependencies.

## Features

- **Event-Driven Architecture**: Prevents lookahead bias and ensures realistic backtesting
- **Multiple Data Sources**: Support for CSV, database, and API data sources
- **Strategy Development**: Clean API for developing custom trading strategies
- **Risk Management**: Comprehensive risk controls including position sizing and stop-losses
- **Machine Learning**: Integrated ML capabilities for predictive modeling
- **Parameter Optimization**: Multiple optimization algorithms including genetic algorithms and Bayesian optimization
- **Performance Analysis**: Detailed performance metrics and risk analysis
- **Backtesting Engine**: Complete backtesting system with realistic market simulation
- **Portfolio Management**: Advanced portfolio tracking and management
- **Execution Simulation**: Realistic order execution with commissions and slippage

## Architecture

The framework follows a modular architecture with the following core components:

```
pytradepath/
├── core/              # Core framework components
├── strategies/        # Trading strategies
├── utils/             # Utility functions
├── data/              # Sample data
├── tests/             # Unit tests
├── examples/          # Example implementations
└── docs/              # Documentation
```

## Core Components

### Event System
- MarketEvent, SignalEvent, OrderEvent, FillEvent

### Data Management
- DataHandler, HistoricCSVDataHandler
- CSVDataProvider, DatabaseDataProvider, APIBasedDataProvider

### Strategy Development
- Strategy base class and multiple example strategies
- BuyAndHoldStrategy, MovingAverageCrossoverStrategy, RSIStrategy
- AdvancedMLStrategy with machine learning capabilities

### Portfolio Management
- Portfolio base class and implementations
- NaivePortfolio, AdvancedPortfolio

### Execution Handling
- ExecutionHandler, SimulatedExecutionHandler

### Risk Management
- RiskManager with multiple implementations
- Position sizing algorithms (Fixed, Kelly Criterion, Volatility)

### Machine Learning
- SimpleLinearRegression, SimpleDecisionTree
- FeatureEngineer for creating trading features

### Optimization
- ParameterOptimizer with multiple algorithms
- WalkForwardOptimizer, MonteCarloOptimizer, BayesianOptimizer

### Analytics
- StatisticalAnalyzer, RiskAnalyzer, TimeSeriesAnalyzer
- Performance metrics calculation

## Installation

The framework is completely self-contained with no external dependencies. All you need is Python 3.6+.

```bash
git clone https://github.com/yourusername/pytradepath.git
cd pytradepath
```

## Usage

### Running Examples

```bash
# Simple moving average crossover backtest
python examples/mac_backtest.py

# Comprehensive framework demonstration
python examples/full_framework_demo.py

# Advanced ML strategy demonstration
python examples/advanced_strategy_demo.py
```

### Running Tests

```bash
# Run unit tests
python tests/test_event_system.py
python tests/test_advanced_components.py
```

## Example Strategies

1. **Buy and Hold**: Simple buy-and-hold strategy
2. **Moving Average Crossover**: Dual moving average crossover strategy
3. **RSI Strategy**: RSI mean-reversion strategy
4. **Advanced ML Strategy**: Machine learning-based strategy with technical indicators

## Performance Metrics

The framework calculates comprehensive performance metrics:
- Total Return
- CAGR (Compound Annual Growth Rate)
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Maximum Drawdown
- Win Rate
- Risk-Adjusted Returns

## Risk Management

The framework includes robust risk management features:
- Position sizing algorithms
- Stop-loss management
- Drawdown controls
- Position limits
- Sector exposure limits
- Liquidity risk management

## Machine Learning

Integrated machine learning capabilities:
- Linear regression models
- Decision tree classifiers
- Feature engineering
- Model evaluation
- Trading performance evaluation

## Optimization

Multiple parameter optimization algorithms:
- Grid search
- Random search
- Genetic algorithms
- Bayesian optimization
- Walk-forward optimization
- Monte Carlo optimization

## Data Management

Flexible data management system:
- CSV data support
- Database support (SQLite)
- API data sources
- Data validation
- Data cleaning
- Data transformation

## Documentation

See the [documentation](docs/framework_documentation.md) for detailed information about the framework components and usage.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This framework is provided for educational and research purposes.

## Acknowledgments

This framework was developed as a comprehensive solution for algorithmic trading research and education.