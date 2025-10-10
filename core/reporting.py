"""
Reporting module for the pytradepath framework.
This module provides performance analysis and reporting capabilities.
"""

import json
import os
import base64
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from .portfolio import Portfolio
from utils.performance import calculate_sharpe_ratio, calculate_max_drawdown, calculate_cagr, calculate_sortino_ratio


class PerformanceAnalyzer:
    """
    Analyzes and calculates performance metrics for trading strategies.
    """

    def __init__(self):
        """
        Initialize the performance analyzer.
        """
        pass

    def calculate_returns(self, portfolio_values: list) -> list:
        """
        Calculate returns from portfolio values.
        
        Parameters:
        portfolio_values - List of portfolio values
        
        Returns:
        List of returns
        """
        if len(portfolio_values) < 2:
            return []
            
        returns = []
        for i in range(1, len(portfolio_values)):
            if portfolio_values[i-1] != 0:
                ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                returns.append(ret)
            else:
                returns.append(0.0)
        return returns

    def calculate_cumulative_returns(self, returns: list) -> list:
        """
        Calculate cumulative returns.
        
        Parameters:
        returns - List of returns
        
        Returns:
        List of cumulative returns
        """
        if not returns:
            return []
            
        cumulative_returns = [1.0]
        for ret in returns:
            cumulative_returns.append(cumulative_returns[-1] * (1 + ret))
        return cumulative_returns[1:]  # Remove initial 1.0

    def calculate_sharpe_ratio(self, returns: list, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio.
        
        Parameters:
        returns - List of returns
        risk_free_rate - Risk-free rate (annualized)
        
        Returns:
        Sharpe ratio
        """
        return calculate_sharpe_ratio(returns, risk_free_rate)

    def calculate_sortino_ratio(self, returns: list, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sortino ratio.
        
        Parameters:
        returns - List of returns
        risk_free_rate - Risk-free rate (annualized)
        
        Returns:
        Sortino ratio
        """
        return calculate_sortino_ratio(returns, risk_free_rate)

    def calculate_max_drawdown(self, portfolio_values: list) -> Tuple[float, list]:
        """
        Calculate maximum drawdown.
        
        Parameters:
        portfolio_values - List of portfolio values
        
        Returns:
        Tuple of (max_drawdown, drawdown_series)
        """
        if len(portfolio_values) == 0:
            return 0.0, []
            
        # Calculate cumulative returns
        if portfolio_values[0] != 0:
            cumulative = [val / portfolio_values[0] for val in portfolio_values]
        else:
            cumulative = [1.0] * len(portfolio_values)
        
        # Calculate running maximum
        running_max = [cumulative[0]]
        for i in range(1, len(cumulative)):
            running_max.append(max(running_max[-1], cumulative[i]))
        
        # Calculate drawdown
        drawdown = []
        for i in range(len(cumulative)):
            if running_max[i] != 0:
                dd = (cumulative[i] - running_max[i]) / running_max[i]
            else:
                dd = 0.0
            drawdown.append(dd)
        
        # Calculate maximum drawdown
        max_drawdown = min(drawdown) if drawdown else 0.0
        
        return max_drawdown, drawdown

    def calculate_calmar_ratio(self, returns: list, portfolio_values: list) -> float:
        """
        Calculate Calmar ratio.
        
        Parameters:
        returns - List of returns
        portfolio_values - List of portfolio values
        
        Returns:
        Calmar ratio
        """
        if len(returns) == 0:
            return 0.0
            
        # Calculate annualized return
        if len(returns) > 0:
            annualized_return = (1 + sum(returns) / len(returns)) ** 252 - 1
        else:
            annualized_return = 0.0
        
        # Calculate maximum drawdown
        max_drawdown, _ = self.calculate_max_drawdown(portfolio_values)
        
        # Calculate Calmar ratio
        if max_drawdown == 0:
            return 0.0
            
        calmar_ratio = annualized_return / abs(max_drawdown)
        return calmar_ratio

    def calculate_volatility(self, returns: list) -> float:
        """
        Calculate annualized volatility.
        
        Parameters:
        returns - List of returns
        
        Returns:
        Annualized volatility
        """
        if len(returns) < 2:
            return 0.0
            
        # Calculate standard deviation
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = variance ** 0.5
        
        volatility = std_dev * (252 ** 0.5)
        return volatility

    def calculate_var(self, returns: list, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Parameters:
        returns - List of returns
        confidence_level - Confidence level (default 95%)
        
        Returns:
        Value at Risk
        """
        if len(returns) == 0:
            return 0.0
            
        # Sort returns
        sorted_returns = sorted(returns)
        
        # Calculate VaR using historical method
        index = int((1 - confidence_level) * len(sorted_returns))
        if index >= len(sorted_returns):
            index = len(sorted_returns) - 1
        
        var = -sorted_returns[index]
        return var

    def calculate_cvar(self, returns: list, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Parameters:
        returns - List of returns
        confidence_level - Confidence level (default 95%)
        
        Returns:
        Conditional Value at Risk
        """
        if len(returns) == 0:
            return 0.0
            
        # Calculate VaR first
        var = self.calculate_var(returns, confidence_level)
        
        # Calculate CVaR as mean of returns below VaR
        tail_returns = [r for r in returns if -r >= var]
        
        if not tail_returns:
            return var
        
        cvar = -sum(tail_returns) / len(tail_returns)
        return cvar

    def calculate_omega_ratio(self, returns: list, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio.
        
        Parameters:
        returns - List of returns
        threshold - Threshold return (default 0)
        
        Returns:
        Omega ratio
        """
        if len(returns) == 0:
            return 0.0
            
        # Separate gains and losses
        gains = [r for r in returns if r > threshold]
        losses = [r for r in returns if r <= threshold]
        
        # Calculate expected gains and losses
        expected_gain = sum(gains) / len(gains) if gains else 0
        expected_loss = abs(sum(losses) / len(losses)) if losses else 0
        
        # Calculate Omega ratio
        if expected_loss == 0:
            return float('inf') if expected_gain > 0 else 0.0
            
        omega_ratio = expected_gain / expected_loss
        return omega_ratio

    def get_performance_summary(self, portfolio_values: list, risk_free_rate: float = 0.0) -> Dict:
        """
        Get comprehensive performance summary.
        
        Parameters:
        portfolio_values - List of portfolio values
        risk_free_rate - Risk-free rate (annualized)
        
        Returns:
        Dictionary with performance metrics
        """
        if len(portfolio_values) == 0:
            return {}
            
        # Calculate returns
        returns = self.calculate_returns(portfolio_values)
        
        # Calculate metrics
        if len(portfolio_values) > 0 and portfolio_values[0] != 0:
            total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        else:
            total_return = 0.0
            
        annualized_return = calculate_cagr(portfolio_values[0], portfolio_values[-1], len(portfolio_values) / 252) if portfolio_values and portfolio_values[0] != 0 else 0.0
        volatility = self.calculate_volatility(returns)
        sharpe_ratio = self.calculate_sharpe_ratio(returns, risk_free_rate)
        sortino_ratio = self.calculate_sortino_ratio(returns, risk_free_rate)
        max_drawdown, _ = self.calculate_max_drawdown(portfolio_values)
        calmar_ratio = self.calculate_calmar_ratio(returns, portfolio_values)
        var_95 = self.calculate_var(returns, 0.95)
        cvar_95 = self.calculate_cvar(returns, 0.95)
        omega_ratio = self.calculate_omega_ratio(returns)
        
        # Win/loss statistics
        winning_periods = [r for r in returns if r > 0]
        losing_periods = [r for r in returns if r < 0]
        
        win_rate = len(winning_periods) / len(returns) if len(returns) > 0 else 0
        avg_win = sum(winning_periods) / len(winning_periods) if winning_periods else 0
        avg_loss = sum(losing_periods) / len(losing_periods) if losing_periods else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'omega_ratio': omega_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        }


class ReportGenerator:
    """
    Generates comprehensive reports for trading strategies.
    """

    def __init__(self, analyzer: PerformanceAnalyzer):
        """
        Initialize the report generator.
        
        Parameters:
        analyzer - Performance analyzer
        """
        self.analyzer = analyzer

    def generate_html_report(self, portfolio_values: list, 
                            strategy_name: str = "Strategy",
                            benchmark_values: list = None) -> str:
        """
        Generate HTML report.
        
        Parameters:
        portfolio_values - List of portfolio values
        strategy_name - Name of the strategy
        benchmark_values - List of benchmark values (optional)
        
        Returns:
        HTML report as string
        """
        # Calculate performance metrics
        returns = self.analyzer.calculate_returns(portfolio_values)
        metrics = self.analyzer.get_performance_summary(portfolio_values)
        
        # Calculate benchmark metrics if provided
        benchmark_metrics = None
        if benchmark_values is not None:
            benchmark_returns = self.analyzer.calculate_returns(benchmark_values)
            benchmark_metrics = self.analyzer.get_performance_summary(benchmark_values)
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Strategy Report - {strategy_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                .plot {{ margin: 20px 0; text-align: center; }}
                .plot img {{ max-width: 100%; height: auto; }}
                .comparison {{ display: flex; justify-content: space-between; }}
                .comparison div {{ width: 48%; }}
            </style>
        </head>
        <body>
            <h1>Trading Strategy Report: {strategy_name}</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Performance Metrics</h2>
            <table class="metrics-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Return</td><td>{metrics.get('total_return', 0):.2%}</td></tr>
                <tr><td>Annualized Return</td><td>{metrics.get('annualized_return', 0):.2%}</td></tr>
                <tr><td>Volatility</td><td>{metrics.get('volatility', 0):.2%}</td></tr>
                <tr><td>Sharpe Ratio</td><td>{metrics.get('sharpe_ratio', 0):.2f}</td></tr>
                <tr><td>Sortino Ratio</td><td>{metrics.get('sortino_ratio', 0):.2f}</td></tr>
                <tr><td>Max Drawdown</td><td>{metrics.get('max_drawdown', 0):.2%}</td></tr>
                <tr><td>Calmar Ratio</td><td>{metrics.get('calmar_ratio', 0):.2f}</td></tr>
                <tr><td>Value at Risk (95%)</td><td>{metrics.get('var_95', 0):.2%}</td></tr>
                <tr><td>Conditional VaR (95%)</td><td>{metrics.get('cvar_95', 0):.2%}</td></tr>
                <tr><td>Omega Ratio</td><td>{metrics.get('omega_ratio', 0):.2f}</td></tr>
                <tr><td>Win Rate</td><td>{metrics.get('win_rate', 0):.2%}</td></tr>
                <tr><td>Avg Win</td><td>{metrics.get('avg_win', 0):.2%}</td></tr>
                <tr><td>Avg Loss</td><td>{metrics.get('avg_loss', 0):.2%}</td></tr>
                <tr><td>Profit Factor</td><td>{metrics.get('profit_factor', 0):.2f}</td></tr>
            </table>
        """
        
        # Add benchmark comparison if available
        if benchmark_metrics:
            html += f"""
            <h2>Benchmark Comparison</h2>
            <div class="comparison">
                <div>
                    <h3>Strategy Metrics</h3>
                    <table class="metrics-table">
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Total Return</td><td>{metrics.get('total_return', 0):.2%}</td></tr>
                        <tr><td>Sharpe Ratio</td><td>{metrics.get('sharpe_ratio', 0):.2f}</td></tr>
                        <tr><td>Max Drawdown</td><td>{metrics.get('max_drawdown', 0):.2%}</td></tr>
                    </table>
                </div>
                <div>
                    <h3>Benchmark Metrics</h3>
                    <table class="metrics-table">
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Total Return</td><td>{benchmark_metrics.get('total_return', 0):.2%}</td></tr>
                        <tr><td>Sharpe Ratio</td><td>{benchmark_metrics.get('sharpe_ratio', 0):.2f}</td></tr>
                        <tr><td>Max Drawdown</td><td>{benchmark_metrics.get('max_drawdown', 0):.2%}</td></tr>
                    </table>
                </div>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html

    def save_html_report(self, html_content: str, filename: str, 
                        directory: str = "reports"):
        """
        Save HTML report to file.
        
        Parameters:
        html_content - HTML content
        filename - Name of the file
        directory - Directory to save the file
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save file
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Report saved to {filepath}")

    def save_performance_metrics(self, metrics: Dict, filename: str,
                                directory: str = "reports"):
        """
        Save performance metrics to JSON file.
        
        Parameters:
        metrics - Dictionary with metrics
        filename - Name of the file
        directory - Directory to save the file
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save file
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Metrics saved to {filepath}")


class TradeAnalyzer:
    """
    Analyzes individual trades and trading behavior.
    """

    def __init__(self):
        """
        Initialize the trade analyzer.
        """
        pass

    def analyze_trades(self, trades: List[Dict]) -> Dict:
        """
        Analyze a list of trades.
        
        Parameters:
        trades - List of trade dictionaries
        
        Returns:
        Dictionary with trade analysis
        """
        if not trades:
            return {}
        
        # Basic statistics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        # PnL statistics
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        pnl_values = [t.get('pnl', 0) for t in trades]
        std_pnl = (sum((p - avg_pnl) ** 2 for p in pnl_values) / (len(pnl_values) - 1)) ** 0.5 if len(pnl_values) > 1 else 0
        
        # Win/loss statistics
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Drawdown analysis
        cumulative_pnl = [0]
        for trade in trades:
            cumulative_pnl.append(cumulative_pnl[-1] + trade.get('pnl', 0))
        
        running_max = [cumulative_pnl[0]]
        for i in range(1, len(cumulative_pnl)):
            running_max.append(max(running_max[-1], cumulative_pnl[i]))
        
        drawdown = [cumulative_pnl[i] - running_max[i] for i in range(len(cumulative_pnl))]
        max_drawdown = min(drawdown) if drawdown else 0
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'std_pnl': std_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'largest_win': max(pnl_values) if pnl_values else 0,
            'largest_loss': min(pnl_values) if pnl_values else 0
        }


class SensitivityAnalyzer:
    """
    Analyzes strategy sensitivity to parameters.
    """

    def __init__(self):
        """
        Initialize the sensitivity analyzer.
        """
        pass

    def analyze_parameter_sensitivity(self, parameter_results: Dict[str, List[Dict]]) -> Dict:
        """
        Analyze sensitivity to parameters.
        
        Parameters:
        parameter_results - Dictionary with parameter names as keys and lists of results as values
        
        Returns:
        Dictionary with sensitivity analysis
        """
        sensitivity_analysis = {}
        
        for param_name, results in parameter_results.items():
            if not results:
                continue
                
            # Calculate metrics
            returns = [r.get('return', 0) for r in results]
            sharpe_ratios = [r.get('sharpe', 0) for r in results]
            max_drawdowns = [r.get('max_drawdown', 0) for r in results]
            win_rates = [r.get('win_rate', 0) for r in results]
            
            metrics = {
                'mean_return': sum(returns) / len(returns) if returns else 0,
                'std_return': (sum((r - sum(returns) / len(returns)) ** 2 for r in returns) / (len(returns) - 1)) ** 0.5 if len(returns) > 1 else 0,
                'sharpe_ratio': sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0,
                'max_drawdown': sum(max_drawdowns) / len(max_drawdowns) if max_drawdowns else 0,
                'win_rate': sum(win_rates) / len(win_rates) if win_rates else 0
            }
            
            sensitivity_analysis[param_name] = {
                'values': [r.get('value', 0) for r in results],
                'returns': returns,
                'metrics': metrics
            }
        
        return sensitivity_analysis


def create_comprehensive_report(portfolio_values: list, 
                               strategy_name: str = "Strategy",
                               benchmark_values: list = None,
                               trades: List[Dict] = None) -> str:
    """
    Create a comprehensive trading strategy report.
    
    Parameters:
    portfolio_values - List of portfolio values
    strategy_name - Name of the strategy
    benchmark_values - List of benchmark values (optional)
    trades - List of trade dictionaries (optional)
    
    Returns:
    HTML report as string
    """
    # Initialize components
    analyzer = PerformanceAnalyzer()
    report_generator = ReportGenerator(analyzer)
    
    # Generate HTML report
    html_report = report_generator.generate_html_report(
        portfolio_values, strategy_name, benchmark_values
    )
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{strategy_name.replace(' ', '_')}_{timestamp}.html"
    report_generator.save_html_report(html_report, filename)
    
    # Save metrics
    if len(portfolio_values) > 0:
        returns = analyzer.calculate_returns(portfolio_values)
        metrics = analyzer.get_performance_summary(portfolio_values)
        metrics_filename = f"{strategy_name.replace(' ', '_')}_{timestamp}_metrics.json"
        report_generator.save_performance_metrics(metrics, metrics_filename)
    
    # Analyze trades if provided
    if trades:
        trade_analyzer = TradeAnalyzer()
        trade_metrics = trade_analyzer.analyze_trades(trades)
        trade_filename = f"{strategy_name.replace(' ', '_')}_{timestamp}_trade_metrics.json"
        report_generator.save_performance_metrics(trade_metrics, trade_filename)
    
    return html_report


def compare_strategies(strategy_results: Dict[str, list],
                      strategy_names: List[str] = None) -> str:
    """
    Compare multiple strategies.
    
    Parameters:
    strategy_results - Dictionary with strategy names as keys and portfolio value lists as values
    strategy_names - List of strategy names to compare (optional)
    
    Returns:
    HTML comparison report
    """
    if not strategy_results:
        return "<p>No strategies to compare</p>"
    
    # Filter strategies if names provided
    if strategy_names:
        filtered_results = {name: strategy_results[name] for name in strategy_names if name in strategy_results}
    else:
        filtered_results = strategy_results
    
    # Generate comparison HTML
    html = """
    <h2>Strategy Comparison</h2>
    <table class="metrics-table">
        <tr><th>Strategy</th><th>Total Return</th><th>Sharpe Ratio</th><th>Max Drawdown</th></tr>
    """
    
    analyzer = PerformanceAnalyzer()
    
    for name, values in filtered_results.items():
        if len(values) > 0:
            returns = analyzer.calculate_returns(values)
            metrics = analyzer.get_performance_summary(values)
            html += f"""
            <tr>
                <td>{name}</td>
                <td>{metrics.get('total_return', 0):.2%}</td>
                <td>{metrics.get('sharpe_ratio', 0):.2f}</td>
                <td>{metrics.get('max_drawdown', 0):.2%}</td>
            </tr>
            """
    
    html += "</table>"
    return html