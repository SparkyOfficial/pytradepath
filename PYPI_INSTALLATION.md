# Installing PyTradePath from PyPI

## Installation

You can install PyTradePath directly from PyPI using pip:

```bash
pip install pytradepath
```

## Usage

After installation, you can use PyTradePath in several ways:

### 1. Using the CLI
```bash
# Show help
pytradepath --help

# Run demonstrations
pytradepath demo simple    # Simple moving average crossover
pytradepath demo full      # Full framework demonstration
pytradepath demo mac       # Moving Average Crossover strategy
pytradepath demo ml        # Machine Learning strategy
pytradepath demo momentum  # Momentum strategy

# Run tests
pytradepath test all       # Run all tests
pytradepath test event     # Run event system tests
pytradepath test advanced  # Run advanced components tests

# Create a new strategy template
pytradepath create MyStrategy
```

### 2. Importing in Python code
```python
from pytradepath.core import EventEngine
from pytradepath.strategies import MovingAverageCrossoverStrategy

# Create and run your trading strategy
# ... your code here ...
```

### 3. Running examples
After installation, you can copy examples from the repository and run them directly.

## Requirements

PyTradePath is a pure Python framework with no external dependencies, making it easy to install and use. It requires Python 3.6 or higher.