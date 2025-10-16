# PyTradePath - Algorithmic Trading Framework

PyTradePath is a comprehensive, event-driven framework for backtesting and algorithmic trading. It provides a complete ecosystem for developing, testing, and deploying trading strategies with realistic market simulation. Enhanced to over 14,000 lines of code with sophisticated algorithms and realistic market modeling.

## Features

- **Event-Driven Architecture**: Prevents lookahead bias with proper event loop implementation
- **Realistic Market Simulation**: Advanced models for commissions, slippage, liquidity, and market microstructure
- **Extensible Strategy Framework**: Clean API for developing custom trading strategies
- **Risk Management**: Sophisticated position sizing, stop-losses, and portfolio-level controls
- **Performance Analysis**: Comprehensive metrics including Sharpe ratio, drawdowns, and PnL
- **Parameter Optimization**: Grid search, random search, and genetic algorithms
- **Machine Learning Integration**: Feature engineering and model deployment capabilities
- **Live Trading Simulation**: Paper trading with real-time data handling
- **Pure Python Implementation**: No external dependencies for easy deployment
- **Comprehensive Documentation**: Extensive guides and examples
- **Enhanced Statistical Methods**: Advanced analytics including Hurst exponent, correlation analysis, and robust regression
- **Realistic Execution Modeling**: Tiered commissions, partial fills, and market impact simulation

## Installation

### From PyPI (Recommended)
```bash
pip install pytradepath
```

### From Release Package
1. Download the `pytradepath-release.zip` file from the latest release
2. Extract it to your desired location
3. Navigate to the extracted directory

### Direct Usage
Since PyTradePath is a pure Python framework with no external dependencies, you can run it directly:
```bash
cd pytradepath-directory
python examples/simple_backtest.py
```

## Quick Start

### Using the CLI
PyTradePath comes with a command-line interface for common tasks:

```bash
# Show help
python -m pytradepath.cli --help

# Run demonstrations
python -m pytradepath.cli demo simple    # Simple moving average crossover
python -m pytradepath.cli demo full      # Full framework demonstration
python -m pytradepath.cli demo mac       # Moving Average Crossover strategy
python -m pytradepath.cli demo ml        # Machine Learning strategy
python -m pytradepath.cli demo momentum  # Momentum strategy

# Run tests
python -m pytradepath.cli test all       # Run all tests
python -m pytradepath.cli test event     # Run event system tests
python -m pytradepath.cli test advanced  # Run advanced components tests

# Create a new strategy template
python -m pytradepath.cli create MyStrategy
```

### Running Examples Directly
You can also run examples directly:
```bash
python examples/simple_backtest.py           # Simple backtest example
python examples/full_framework_demo.py       # Full framework demonstration
python examples/mac_backtest.py              # Moving Average Crossover backtest
python examples/momentum_strategy_demo.py    # Momentum strategy example
python examples/advanced_strategy_demo.py    # Advanced ML strategy example
```

## Framework Structure

The framework is organized into the following modules:

- **[core/](file:///C:/Users/%D0%91%D0%BE%D0%B3%D0%B4%D0%B0%D0%BD/Desktop/pytradepath/core/)** - Core components (events, data handlers, portfolio, execution)
- **[strategies/](file:///C:/Users/%D0%91%D0%BE%D0%B3%D0%B4%D0%B0%D0%BD/Desktop/pytradepath/strategies/)** - Trading strategies
- **[risk/](file:///C:/Users/%D0%91%D0%BE%D0%B3%D0%B4%D0%B0%D0%BD/Desktop/pytradepath/risk/)** - Risk management components
- **[optimization/](file:///C:/Users/%D0%91%D0%BE%D0%B3%D0%B4%D0%B0%D0%BD/Desktop/pytradepath/optimization/)** - Parameter optimization tools
- **[ml/](file:///C:/Users/%D0%91%D0%BE%D0%B3%D0%B4%D0%B0%D0%BD/Desktop/pytradepath/ml/)** - Machine learning components
- **[live/](file:///C:/Users/%D0%91%D0%BE%D0%B3%D0%B4%D0%B0%D0%BD/Desktop/pytradepath/live/)** - Live trading simulation
- **[analytics/](file:///C:/Users/%D0%91%D0%BE%D0%B3%D0%B4%D0%B0%D0%BD/Desktop/pytradepath/analytics/)** - Performance analysis and reporting
- **[utils/](file:///C:/Users/%D0%91%D0%BE%D0%B3%D0%B4%D0%B0%D0%BD/Desktop/pytradepath/utils/)** - Utility functions
- **[data/](file:///C:/Users/%D0%91%D0%BE%D0%B3%D0%B4%D0%B0%D0%BD/Desktop/pytradepath/data/)** - Data management
- **[config/](file:///C:/Users/%D0%91%D0%BE%D0%B3%D0%B4%D0%B0%D0%BD/Desktop/pytradepath/config/)** - Configuration management
- **[logging/](file:///C:/Users/%D0%91%D0%BE%D0%B3%D0%B4%D0%B0%D0%BD/Desktop/pytradepath/logging/)** - Logging system
- **[tests/](file:///C:/Users/%D0%91%D0%BE%D0%B3%D0%B4%D0%B0%D0%BD/Desktop/pytradepath/tests/)** - Unit tests
- **[docs/](file:///C:/Users/%D0%91%D0%BE%D0%B3%D0%B4%D0%B0%D0%BD/Desktop/pytradepath/docs/)** - Documentation
- **[examples/](file:///C:/Users/%D0%91%D0%BE%D0%B3%D0%B4%D0%B0%D0%BD/Desktop/pytradepath/examples/)** - Example scripts

## Creating Custom Strategies

The easiest way to create a new strategy is using the CLI:
```bash
python -m pytradepath.cli create MyNewStrategy
```

This creates a template file in `strategies/mynewstrategy_strategy.py` that you can customize.

## Performance Analysis

The framework provides comprehensive performance analysis tools that calculate key metrics:
- Total returns and annualized returns
- Sharpe ratio and Sortino ratio
- Maximum drawdown and drawdown duration
- Volatility and Value at Risk (VaR)
- Trade statistics and exposure analysis

## Advanced Features

### Risk Management
- Position sizing algorithms
- Stop-loss and take-profit orders
- Trailing stops
- Portfolio-level risk controls

### Parameter Optimization
- Grid search optimization
- Random search algorithms
- Genetic algorithm optimization
- Walk-forward analysis

### Machine Learning Integration
- Feature engineering pipelines
- Model training and evaluation
- Prediction integration with trading strategies
- Ensemble methods and stacking

### Live Trading Simulation
- Real-time data handling
- Order execution simulation
- Position tracking and monitoring
- Risk monitoring and alerts

## Documentation

Detailed documentation is available in the [docs/](file:///C:/Users/%D0%91%D0%BE%D0%B3%D0%B4%D0%B0%D0%BD/Desktop/pytradepath/docs/) directory, including:
- API reference
- Strategy development guide
- Risk management best practices
- Performance analysis techniques
- Live trading setup instructions

For personal guidance on how to work with the library, see [PERSONAL_WORKING_GUIDE.md](file:///C:/Users/%D0%91%D0%BE%D0%B3%D0%B4%D0%B0%D0%BD/Desktop/pytradepath/PERSONAL_WORKING_GUIDE.md).

## Release Package

A ready-to-use release package is available as `pytradepath-release.zip`. This package contains:
- All source code
- Examples and demos
- Documentation
- Tests
- CLI tools
- Configuration files

To create a new release package, run:
```bash
# Using the batch script (Windows)
make_release.bat

# Using the shell script (Unix/Linux/Mac)
./make_release.sh

# Using Python directly
python release.py --package-only
```

## Automated Publishing

This repository uses GitHub Actions for automated building and publishing:

- **[Test Build](file:///C:/Users/%D0%91%D0%BE%D0%B3%D0%B4%D0%B0%D0%BD/Desktop/pytradepath/.github/workflows/test-build.yml)**: Runs on every push/PR to test building the package
- **[Create Release](file:///C:/Users/%D0%91%D0%BE%D0%B3%D0%B4%D0%B0%D0%BD/Desktop/pytradepath/.github/workflows/release.yml)**: Creates GitHub releases when tags are pushed
- **[Publish to PyPI](file:///C:/Users/%D0%91%D0%BE%D0%B3%D0%B4%D0%B0%D0%BD/Desktop/pytradepath/.github/workflows/publish.yml)**: Publishes to PyPI using OIDC authentication

To create a new release:
```bash
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

The GitHub Actions workflows will automatically:
1. Build the package
2. Create a GitHub release
3. Publish to PyPI using secure OIDC authentication

## Testing

Run the test suite to verify everything is working correctly:
```bash
python -m pytradepath.cli test all
```

Or run tests directly:
```bash
python tests/test_event_system.py
python tests/test_advanced_components.py
```

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](file:///C:/Users/%D0%91%D0%BE%D0%B3%D0%B4%D0%B0%D0%BD/Desktop/pytradepath/LICENSE) file for details.

## Support

For support, please open an issue on the GitHub repository or contact the development team.