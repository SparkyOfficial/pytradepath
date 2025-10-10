"""
Comprehensive documentation system for the pytradepath framework.
This module provides tools for generating documentation, tutorials, and examples.
"""

import os
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import inspect
import warnings


class DocumentationType(Enum):
    """Types of documentation."""
    TUTORIAL = "tutorial"
    API_REFERENCE = "api_reference"
    EXAMPLE = "example"
    GUIDE = "guide"
    FAQ = "faq"


@dataclass
class DocumentationSection:
    """Represents a section of documentation."""
    title: str
    content: str
    section_type: DocumentationType
    tags: List[str] = None
    related_sections: List[str] = None
    examples: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.related_sections is None:
            self.related_sections = []
        if self.examples is None:
            self.examples = []


@dataclass
class CodeExample:
    """Represents a code example."""
    title: str
    description: str
    code: str
    language: str = "python"
    expected_output: Optional[str] = None
    difficulty: str = "beginner"


class DocumentationGenerator:
    """Generates comprehensive documentation for the framework."""

    def __init__(self, output_directory: str = "docs"):
        """
        Initialize the documentation generator.
        
        Parameters:
        output_directory - Directory to save documentation files
        """
        self.output_directory = output_directory
        self.sections = []
        self.examples = []
        self.tutorials = []
        
        # Create output directory
        os.makedirs(output_directory, exist_ok=True)

    def add_section(self, section: DocumentationSection):
        """
        Add a documentation section.
        
        Parameters:
        section - Documentation section to add
        """
        self.sections.append(section)

    def add_example(self, example: CodeExample):
        """
        Add a code example.
        
        Parameters:
        example - Code example to add
        """
        self.examples.append(example)

    def generate_api_reference(self, modules: List[str]) -> DocumentationSection:
        """
        Generate API reference documentation.
        
        Parameters:
        modules - List of module names to document
        
        Returns:
        Documentation section with API reference
        """
        content = "# API Reference\n\n"
        
        for module_name in modules:
            content += f"## Module: {module_name}\n\n"
            
            try:
                # Import module
                module = __import__(module_name)
                
                # Get module docstring
                if module.__doc__:
                    content += f"{module.__doc__}\n\n"
                
                # Get classes and functions
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and not name.startswith('_'):
                        content += f"### Class: {name}\n\n"
                        if obj.__doc__:
                            content += f"{obj.__doc__}\n\n"
                        
                        # Get methods
                        for method_name, method in inspect.getmembers(obj):
                            if inspect.isfunction(method) and not method_name.startswith('_'):
                                content += f"#### Method: {name}.{method_name}\n\n"
                                if method.__doc__:
                                    content += f"{method.__doc__}\n\n"
                    
                    elif inspect.isfunction(obj) and not name.startswith('_'):
                        content += f"### Function: {name}\n\n"
                        if obj.__doc__:
                            content += f"{obj.__doc__}\n\n"
                            
            except Exception as e:
                content += f"Error documenting module {module_name}: {e}\n\n"
        
        return DocumentationSection(
            title="API Reference",
            content=content,
            section_type=DocumentationType.API_REFERENCE,
            tags=["api", "reference", "modules"]
        )

    def generate_tutorial(self, title: str, steps: List[Dict[str, str]]) -> DocumentationSection:
        """
        Generate a tutorial.
        
        Parameters:
        title - Tutorial title
        steps - List of tutorial steps
        
        Returns:
        Documentation section with tutorial
        """
        content = f"# {title}\n\n"
        
        for i, step in enumerate(steps, 1):
            content += f"## Step {i}: {step['title']}\n\n"
            content += f"{step['description']}\n\n"
            
            if 'code' in step:
                content += "``python\n"
                content += f"{step['code']}\n"
                content += "```\n\n"
        
        return DocumentationSection(
            title=title,
            content=content,
            section_type=DocumentationType.TUTORIAL,
            tags=["tutorial", "getting_started", "beginner"]
        )

    def generate_example_documentation(self, example: CodeExample) -> DocumentationSection:
        """
        Generate documentation for a code example.
        
        Parameters:
        example - Code example
        
        Returns:
        Documentation section
        """
        content = f"# {example.title}\n\n"
        content += f"{example.description}\n\n"
        
        content += f"## Code ({example.language})\n\n"
        content += f"``{example.language}\n"
        content += f"{example.code}\n"
        content += "```\n\n"
        
        if example.expected_output:
            content += "## Expected Output\n\n"
            content += f"```\n{example.expected_output}\n```\n\n"
        
        return DocumentationSection(
            title=example.title,
            content=content,
            section_type=DocumentationType.EXAMPLE,
            tags=["example", "code", example.difficulty]
        )

    def save_documentation(self, format: str = "markdown"):
        """
        Save all documentation to files.
        
        Parameters:
        format - Output format (markdown, html, json)
        """
        if format.lower() == "markdown":
            self._save_markdown()
        elif format.lower() == "html":
            self._save_html()
        elif format.lower() == "json":
            self._save_json()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _save_markdown(self):
        """Save documentation as Markdown files."""
        # Save sections
        for section in self.sections:
            filename = f"{section.title.lower().replace(' ', '_')}.md"
            filepath = os.path.join(self.output_directory, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {section.title}\n\n")
                f.write(section.content)
            
            print(f"Saved {filepath}")

    def _save_html(self):
        """Save documentation as HTML files."""
        # Convert Markdown to HTML
        try:
            import markdown
            html_content = "<!DOCTYPE html>\n<html>\n<head>\n<title>PyTradePath Documentation</title>\n"
            html_content += "<style>body { font-family: Arial, sans-serif; margin: 40px; }</style>\n"
            html_content += "</head>\n<body>\n"
        
            for section in self.sections:
                # Convert markdown to HTML
                html_section = markdown.markdown(f"# {section.title}\n\n{section.content}")
                html_content += html_section + "\n<hr>\n"
        
            html_content += "</body>\n</html>"
        
            filepath = os.path.join(self.output_directory, "documentation.html")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
            print(f"Saved {filepath}")
        except ImportError:
            # Fallback to simple HTML generation without markdown
            html_content = "<!DOCTYPE html>\n<html>\n<head>\n<title>PyTradePath Documentation</title>\n"
            html_content += "<style>body { font-family: Arial, sans-serif; margin: 40px; }</style>\n"
            html_content += "</head>\n<body>\n"
        
            for section in self.sections:
                html_content += f"<h1>{section.title}</h1>\n"
                # Simple conversion of markdown to HTML
                content = section.content.replace('\n\n', '</p>\n<p>').replace('\n', '<br>\n')
                content = f"<p>{content}</p>"
                html_content += content + "\n<hr>\n"
        
            html_content += "</body>\n</html>"
        
            filepath = os.path.join(self.output_directory, "documentation.html")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
            print(f"Saved {filepath} (without markdown formatting)")

    def _save_json(self):
        """Save documentation as JSON files."""
        # Save sections as JSON
        sections_data = [asdict(section) for section in self.sections]
        filepath = os.path.join(self.output_directory, "documentation.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sections_data, f, indent=4, default=str)
        
        print(f"Saved {filepath}")


class TutorialGenerator:
    """Generates interactive tutorials."""

    def __init__(self):
        """Initialize the tutorial generator."""
        self.tutorials = []

    def create_backtesting_tutorial(self) -> DocumentationSection:
        """Create a backtesting tutorial."""
        steps = [
            {
                "title": "Setting Up Your Environment",
                "description": "First, make sure you have the pytradepath framework installed and set up.",
                "code": """import sys
sys.path.append('path/to/pytradepath')

from core.engine import BacktestingEngine
from core.data_handler import HistoricCSVDataHandler
from core.strategy import BuyAndHoldStrategy
from core.portfolio import NaivePortfolio
from core.execution import SimulatedExecutionHandler"""
            },
            {
                "title": "Preparing Your Data",
                "description": "Prepare your market data in CSV format with the required columns.",
                "code": """# Create sample data
import csv

with open('data/sample_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['datetime', 'open', 'high', 'low', 'close', 'volume'])
    # Add your data rows here"""
            },
            {
                "title": "Creating the Backtesting Engine",
                "description": "Set up the backtesting engine with your components.",
                "code": """symbols = ['sample_data']
initial_capital = 100000.0

engine = BacktestingEngine(
    data_handler=lambda: HistoricCSVDataHandler('data', symbols),
    strategy=lambda symbols: BuyAndHoldStrategy(symbols),
    portfolio=lambda data_handler, events, initial_capital: NaivePortfolio(
        data_handler, events, initial_capital=initial_capital),
    execution_handler=lambda events: SimulatedExecutionHandler(events),
    symbol_list=symbols,
    initial_capital=initial_capital
)"""
            },
            {
                "title": "Running the Backtest",
                "description": "Execute the backtest and analyze the results.",
                "code": """# Run the backtest
engine.run()

# View results
print(f"Signals generated: {engine.signals}")
print(f"Orders placed: {engine.orders}")
print(f"Trades executed: {engine.fills}")"""
            }
        ]
        
        return DocumentationGenerator().generate_tutorial(
            "Backtesting Tutorial", steps
        )

    def create_strategy_development_tutorial(self) -> DocumentationSection:
        """Create a strategy development tutorial."""
        steps = [
            {
                "title": "Understanding the Strategy Base Class",
                "description": "All strategies must inherit from the Strategy base class.",
                "code": """from core.strategy import Strategy
from core.event import SignalEvent

class MyStrategy(Strategy):
    def __init__(self, symbols):
        super().__init__(symbols)
        # Initialize your strategy parameters here
        
    def calculate_signals(self, event):
        # Implement your signal generation logic here
        pass"""
            },
            {
                "title": "Implementing Signal Generation",
                "description": "Generate trading signals based on market data.",
                "code": """def calculate_signals(self, event):
    if event.type.name == 'MARKET':
        for symbol in self.symbols:
            # Your signal logic here
            if self._should_buy(symbol):
                signal = SignalEvent(symbol, 'BUY', 1.0)
                self.events_queue.put(signal)
            elif self._should_sell(symbol):
                signal = SignalEvent(symbol, 'SELL', 1.0)
                self.events_queue.put(signal)"""
            },
            {
                "title": "Testing Your Strategy",
                "description": "Test your strategy with sample data.",
                "code": """# Create and test your strategy
strategy = MyStrategy(['AAPL'])
# ... test code ..."""
            }
        ]
        
        return DocumentationGenerator().generate_tutorial(
            "Strategy Development Tutorial", steps
        )


class ExampleLibrary:
    """Library of code examples."""

    def __init__(self):
        """Initialize the example library."""
        self.examples = []

    def create_basic_backtest_example(self) -> CodeExample:
        """Create a basic backtest example."""
        code = """from core.engine import BacktestingEngine
from core.data_handler import HistoricCSVDataHandler
from core.strategy import BuyAndHoldStrategy
from core.portfolio import NaivePortfolio
from core.execution import SimulatedExecutionHandler

# Define symbols and capital
symbols = ['sample_data']
initial_capital = 100000.0

# Create and run backtest
engine = BacktestingEngine(
    data_handler=lambda: HistoricCSVDataHandler('data', symbols),
    strategy=lambda symbols: BuyAndHoldStrategy(symbols),
    portfolio=lambda data_handler, events, initial_capital: NaivePortfolio(
        data_handler, events, initial_capital=initial_capital),
    execution_handler=lambda events: SimulatedExecutionHandler(events),
    symbol_list=symbols,
    initial_capital=initial_capital
)

engine.run()"""
        
        return CodeExample(
            title="Basic Backtest Example",
            description="A simple example showing how to run a basic backtest.",
            code=code,
            difficulty="beginner"
        )

    def create_custom_strategy_example(self) -> CodeExample:
        """Create a custom strategy example."""
        code = """from core.strategy import Strategy
from core.event import SignalEvent

class MovingAverageCrossoverStrategy(Strategy):
    def __init__(self, symbols, short_window=10, long_window=50):
        super().__init__(symbols)
        self.short_window = short_window
        self.long_window = long_window
        self.bought = self._calculate_initial_bought()

    def _calculate_initial_bought(self):
        bought = {}
        for s in self.symbols:
            bought[s] = False
        return bought

    def calculate_signals(self, event):
        if event.type.name == 'MARKET':
            # Implement your moving average crossover logic here
            # This is a simplified example
            for s in self.symbols:
                if not self.bought[s]:
                    signal = SignalEvent(s, 'BUY', 1.0)
                    self.events_queue.put(signal)
                    self.bought[s] = True"""
        
        return CodeExample(
            title="Custom Strategy Example",
            description="An example showing how to create a custom trading strategy.",
            code=code,
            difficulty="intermediate"
        )

    def create_risk_management_example(self) -> CodeExample:
        """Create a risk management example."""
        code = """from core.risk import NaiveRiskManager
from core.portfolio import NaivePortfolio

# Create portfolio
portfolio = NaivePortfolio(data_handler, events)

# Create risk manager
risk_manager = NaiveRiskManager(
    portfolio,
    max_positions=10,
    max_percent_per_position=0.1,
    stop_loss_percent=0.05
)

# Use risk manager in your trading system
# ... implementation code ..."""
        
        return CodeExample(
            title="Risk Management Example",
            description="An example showing how to implement risk management.",
            code=code,
            difficulty="advanced"
        )


class FAQGenerator:
    """Generates frequently asked questions."""

    def __init__(self):
        """Initialize the FAQ generator."""
        self.faqs = []

    def generate_common_faqs(self) -> List[DocumentationSection]:
        """Generate common FAQs."""
        faqs = [
            {
                "question": "What is pytradepath?",
                "answer": "PyTradePath is an event-driven backtesting and algorithmic trading framework designed for testing trading strategies on historical data and running them in real-time. It focuses on architectural clarity, extensibility, and realistic simulation."
            },
            {
                "question": "How do I install pytradepath?",
                "answer": "PyTradePath can be installed by cloning the repository and installing the required dependencies. The main dependencies are Python 3.7+ and standard libraries. No external packages are required for basic functionality."
            },
            {
                "question": "What data formats are supported?",
                "answer": "The framework currently supports CSV files for historical data. The data should include datetime, open, high, low, close, and volume columns. Additional data handlers can be implemented for other formats."
            },
            {
                "question": "How do I create a custom trading strategy?",
                "answer": "To create a custom strategy, inherit from the Strategy base class and implement the calculate_signals method. This method should generate SignalEvents based on market data and push them to the events queue."
            },
            {
                "question": "Can I use pytradepath for live trading?",
                "answer": "Yes, the framework includes components for live trading, including a paper trading broker and live data feeds. However, you should thoroughly test your strategies in backtesting and paper trading before using real money."
            },
            {
                "question": "How accurate is the market simulation?",
                "answer": "The framework includes realistic market simulation features such as commissions, slippage, and fill probability. However, it's important to remember that backtested results do not guarantee future performance."
            }
        ]
        
        documentation_sections = []
        for faq in faqs:
            content = f"## Question\n\n{faq['question']}\n\n"
            content += f"## Answer\n\n{faq['answer']}\n"
            
            section = DocumentationSection(
                title=f"FAQ: {faq['question']}",
                content=content,
                section_type=DocumentationType.FAQ,
                tags=["faq", "help", "troubleshooting"]
            )
            documentation_sections.append(section)
        
        return documentation_sections


class DocumentationManager:
    """Manages all documentation components."""

    def __init__(self, output_directory: str = "docs"):
        """
        Initialize the documentation manager.
        
        Parameters:
        output_directory - Directory to save documentation files
        """
        self.output_directory = output_directory
        self.generator = DocumentationGenerator(output_directory)
        self.tutorial_generator = TutorialGenerator()
        self.example_library = ExampleLibrary()
        self.faq_generator = FAQGenerator()

    def generate_complete_documentation(self):
        """Generate complete documentation."""
        print("Generating complete documentation...")
        
        # Generate API reference
        try:
            api_section = self.generator.generate_api_reference([
                "core.event",
                "core.data_handler",
                "core.strategy",
                "core.portfolio",
                "core.execution",
                "core.engine"
            ])
            self.generator.add_section(api_section)
        except Exception as e:
            print(f"Error generating API reference: {e}")
        
        # Generate tutorials
        try:
            backtesting_tutorial = self.tutorial_generator.create_backtesting_tutorial()
            self.generator.add_section(backtesting_tutorial)
            
            strategy_tutorial = self.tutorial_generator.create_strategy_development_tutorial()
            self.generator.add_section(strategy_tutorial)
        except Exception as e:
            print(f"Error generating tutorials: {e}")
        
        # Generate examples
        try:
            basic_example = self.example_library.create_basic_backtest_example()
            self.generator.add_example(basic_example)
            basic_section = self.generator.generate_example_documentation(basic_example)
            self.generator.add_section(basic_section)
            
            custom_example = self.example_library.create_custom_strategy_example()
            self.generator.add_example(custom_example)
            custom_section = self.generator.generate_example_documentation(custom_example)
            self.generator.add_section(custom_section)
            
            risk_example = self.example_library.create_risk_management_example()
            self.generator.add_example(risk_example)
            risk_section = self.generator.generate_example_documentation(risk_example)
            self.generator.add_section(risk_section)
        except Exception as e:
            print(f"Error generating examples: {e}")
        
        # Generate FAQs
        try:
            faq_sections = self.faq_generator.generate_common_faqs()
            for section in faq_sections:
                self.generator.add_section(section)
        except Exception as e:
            print(f"Error generating FAQs: {e}")
        
        # Save documentation
        try:
            self.generator.save_documentation("markdown")
            self.generator.save_documentation("json")
            print("Documentation saved successfully!")
        except Exception as e:
            print(f"Error saving documentation: {e}")

    def get_documentation_stats(self) -> Dict[str, int]:
        """
        Get documentation statistics.
        
        Returns:
        Dictionary with documentation statistics
        """
        return {
            "sections": len(self.generator.sections),
            "examples": len(self.generator.examples),
            "tutorials": len(self.tutorial_generator.tutorials),
            "output_directory": self.output_directory
        }


def create_documentation():
    """Create comprehensive documentation for the framework."""
    print("Creating comprehensive documentation...")
    
    # Create documentation manager
    doc_manager = DocumentationManager("docs")
    
    # Generate complete documentation
    doc_manager.generate_complete_documentation()
    
    # Print statistics
    stats = doc_manager.get_documentation_stats()
    print(f"Documentation statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return doc_manager


# Example usage
if __name__ == "__main__":
    # Create documentation
    documentation = create_documentation()
    
    print("\nDocumentation generation completed!")
    print("Check the 'docs' directory for generated files.")