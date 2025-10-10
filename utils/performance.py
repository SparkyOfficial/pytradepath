import math

def mean(values):
    """Calculate mean of values."""
    if not values:
        return 0.0
    return sum(values) / len(values)

def std(values):
    """Calculate standard deviation of values."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate the Sharpe ratio of a strategy.
    
    Parameters:
    returns - List of returns
    risk_free_rate - Risk-free rate (default 0%)
    
    Returns:
    Sharpe ratio
    """
    if not returns:
        return 0.0
        
    excess_returns = [r - risk_free_rate for r in returns]
    mean_excess = mean(excess_returns)
    std_excess = std(excess_returns)
    
    if std_excess == 0:
        return 0.0
        
    return mean_excess / std_excess * math.sqrt(252)  # Annualized

def calculate_max_drawdown(equity_curve):
    """
    Calculate the maximum drawdown of a strategy.
    
    Parameters:
    equity_curve - List of portfolio value over time
    
    Returns:
    Maximum drawdown as a percentage
    """
    if not equity_curve:
        return 0.0
        
    # Calculate the running maximum
    running_max = [equity_curve[0]]
    for i in range(1, len(equity_curve)):
        running_max.append(max(running_max[-1], equity_curve[i]))
    
    # Calculate the drawdown
    drawdowns = []
    for i in range(len(equity_curve)):
        if running_max[i] != 0:
            drawdown = (equity_curve[i] - running_max[i]) / running_max[i]
        else:
            drawdown = 0.0
        drawdowns.append(drawdown)
    
    # Return the maximum drawdown
    return min(drawdowns) if drawdowns else 0.0

def calculate_cagr(initial_value, final_value, years):
    """
    Calculate the Compound Annual Growth Rate.
    
    Parameters:
    initial_value - Initial portfolio value
    final_value - Final portfolio value
    years - Number of years
    
    Returns:
    CAGR as a percentage
    """
    if initial_value <= 0 or years <= 0:
        return 0.0
    return (final_value / initial_value) ** (1 / years) - 1

def calculate_sortino_ratio(returns, risk_free_rate=0.0, target_return=0.0):
    """
    Calculate the Sortino ratio of a strategy.
    
    Parameters:
    returns - List of returns
    risk_free_rate - Risk-free rate (default 0%)
    target_return - Target return (default 0%)
    
    Returns:
    Sortino ratio
    """
    if not returns:
        return 0.0
        
    # Calculate downside deviation
    downside_returns = [min(0, r - target_return) for r in returns]
    downside_deviation = math.sqrt(sum(r ** 2 for r in downside_returns) / len(returns)) * math.sqrt(252)
    
    # Calculate excess returns
    excess_returns = [r - risk_free_rate for r in returns]
    mean_excess = mean(excess_returns)
    
    if downside_deviation == 0:
        return 0.0
        
    return mean_excess / downside_deviation

def calculate_calmar_ratio(returns, years):
    """
    Calculate the Calmar ratio of a strategy.
    
    Parameters:
    returns - List of returns
    years - Number of years
    
    Returns:
    Calmar ratio
    """
    if not returns or years <= 0:
        return 0.0
        
    # Calculate CAGR
    initial_value = 1.0
    final_value = 1.0
    for r in returns:
        final_value *= (1 + r)
    cagr = calculate_cagr(initial_value, final_value, years)
    
    # Calculate max drawdown
    equity_curve = [1.0]
    for r in returns:
        equity_curve.append(equity_curve[-1] * (1 + r))
    max_dd = abs(calculate_max_drawdown(equity_curve))
    
    if max_dd == 0:
        return 0.0
        
    return cagr / max_dd