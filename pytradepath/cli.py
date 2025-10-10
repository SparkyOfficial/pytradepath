"""
Command Line Interface for PyTradePath
"""

import argparse
import sys
import os

def main():
    """Main entry point for the PyTradePath CLI."""
    parser = argparse.ArgumentParser(
        prog="pytradepath",
        description="PyTradePath - Algorithmic Trading Framework",
        epilog="For more information, visit: https://github.com/yourusername/pytradepath"
    )
    
    parser.add_argument(
        "-v", "--version",
        action="version",
        version="PyTradePath 1.0.0"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run framework demonstrations")
    demo_parser.add_argument(
        "demo_type",
        choices=["simple", "full", "mac", "ml", "momentum"],
        help="Type of demonstration to run"
    )
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run framework tests")
    test_parser.add_argument(
        "test_type",
        choices=["all", "event", "advanced"],
        help="Type of tests to run"
    )
    
    # Create strategy command
    create_parser = subparsers.add_parser("create", help="Create a new strategy template")
    create_parser.add_argument(
        "name",
        help="Name of the new strategy"
    )
    
    args = parser.parse_args()
    
    if args.command == "demo":
        run_demo(args.demo_type)
    elif args.command == "test":
        run_tests(args.test_type)
    elif args.command == "create":
        create_strategy(args.name)
    else:
        parser.print_help()

def run_demo(demo_type):
    """Run a demonstration."""
    print(f"Running {demo_type} demonstration...")
    
    # Import and run the appropriate demo
    try:
        if demo_type == "simple":
            from examples.simple_backtest import run_buy_and_hold_example
            run_buy_and_hold_example()
        elif demo_type == "full":
            from examples.full_framework_demo import main as full_demo
            full_demo()
        elif demo_type == "mac":
            from examples.mac_backtest import run_mac_example
            run_mac_example()
        elif demo_type == "ml":
            from examples.advanced_strategy_demo import main as ml_demo
            ml_demo()
        elif demo_type == "momentum":
            from examples.momentum_strategy_demo import main as momentum_demo
            momentum_demo()
        print(f"\n{demo_type.capitalize()} demonstration completed successfully!")
    except ImportError as e:
        print(f"Error running {demo_type} demonstration: {e}")
        sys.exit(1)

def run_tests(test_type):
    """Run tests."""
    print(f"Running {test_type} tests...")
    
    # Import and run the appropriate tests
    try:
        if test_type == "all":
            import subprocess
            import sys
            
            # Run both test suites using unittest
            result1 = subprocess.run([
                sys.executable, "-m", "unittest", "tests.test_event_system", "-v"
            ], capture_output=True, text=True)
            
            result2 = subprocess.run([
                sys.executable, "-m", "unittest", "tests.test_advanced_components", "-v"
            ], capture_output=True, text=True)
            
            print("Event System Tests:")
            print(result1.stdout)
            if result1.stderr:
                print("Errors:", result1.stderr)
                
            print("\nAdvanced Components Tests:")
            print(result2.stdout)
            if result2.stderr:
                print("Errors:", result2.stderr)
                
        elif test_type == "event":
            import unittest
            from tests.test_event_system import TestEventSystem
            suite = unittest.TestLoader().loadTestsFromTestCase(TestEventSystem)
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            if not result.wasSuccessful():
                sys.exit(1)
        elif test_type == "advanced":
            import unittest
            from tests.test_advanced_components import TestAdvancedMLStrategy, TestPositionSizers, TestMachineLearningModels, TestAnalytics
            suite = unittest.TestSuite()
            suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAdvancedMLStrategy))
            suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPositionSizers))
            suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMachineLearningModels))
            suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAnalytics))
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            if not result.wasSuccessful():
                sys.exit(1)
        print(f"\n{test_type.capitalize()} tests completed!")
    except Exception as e:
        print(f"Error running {test_type} tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def create_strategy(name):
    """Create a new strategy template."""
    print(f"Creating new strategy: {name}")
    
    # Create the strategy file
    strategy_filename = f"{name.lower()}_strategy.py"
    strategy_path = os.path.join("strategies", strategy_filename)
    
    if os.path.exists(strategy_path):
        print(f"Strategy {name} already exists!")
        return
    
    strategy_template = f'''from core.event import SignalEvent
from core.strategy import Strategy


class {name}Strategy(Strategy):
    """
    {name} trading strategy.
    """

    def __init__(self, symbols):
        """
        Initialize the {name} strategy.
        
        Parameters:
        symbols - List of ticker symbols to trade
        """
        super().__init__(symbols)
        # TODO: Initialize strategy parameters

    def calculate_signals(self, event):
        """
        Generate trading signals.
        """
        if event.type.name == 'MARKET':
            for symbol in self.symbols:
                # TODO: Implement your strategy logic here
                # Example:
                # signal = SignalEvent(symbol, 'BUY', 1.0)
                # self.events_queue.put(signal)
                pass
'''
    
    try:
        with open(strategy_path, 'w') as f:
            f.write(strategy_template)
        print(f"Strategy template created at: {strategy_path}")
        print("You can now edit this file to implement your strategy logic.")
    except Exception as e:
        print(f"Error creating strategy: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()