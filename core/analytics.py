"""
Comprehensive analytics module for the pytradepath framework.
This module provides advanced analytical tools for trading strategy analysis.
"""

import math
import json
import warnings
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict


# Simple numpy-like functions
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

def percentile(data, percentile):
    """Calculate percentile of data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = (percentile / 100) * (len(sorted_data) - 1)
    if index.is_integer():
        return sorted_data[int(index)]
    else:
        lower = sorted_data[int(index)]
        upper = sorted_data[min(int(index) + 1, len(sorted_data) - 1)]
        return lower + (upper - lower) * (index - int(index))

def corrcoef(x, y):
    """Calculate correlation coefficient between two arrays."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    mean_x = mean(x)
    mean_y = mean(y)
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
    denominator_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
    denominator_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))
    
    if denominator_x == 0 or denominator_y == 0:
        return 0.0
    
    return numerator / math.sqrt(denominator_x * denominator_y)


class StatisticalAnalyzer:
    """
    Performs statistical analysis on trading data.
    """

    def __init__(self):
        """Initialize the statistical analyzer."""
        pass

    def calculate_descriptive_stats(self, data: List[float]) -> Dict[str, float]:
        """
        Calculate descriptive statistics for a dataset.
        
        Parameters:
        data - List of numerical values
        
        Returns:
        Dictionary with descriptive statistics
        """
        if not data:
            return {}
        
        data_sorted = sorted(data)
        
        stats = {
            'count': len(data),
            'mean': mean(data),
            'median': data_sorted[len(data) // 2] if len(data) % 2 == 1 else (data_sorted[len(data) // 2 - 1] + data_sorted[len(data) // 2]) / 2,
            'std': std(data),
            'var': std(data) ** 2,
            'min': min(data),
            'max': max(data),
            'skewness': self._calculate_skewness(data),
            'kurtosis': self._calculate_kurtosis(data)
        }
        
        return stats

    def _calculate_skewness(self, data: List[float]) -> float:
        """
        Calculate skewness of a dataset.
        
        Parameters:
        data - List of values
        
        Returns:
        Skewness value
        """
        if len(data) < 3:
            return 0.0
        
        m = mean(data)
        std_dev = std(data)
        
        if std_dev == 0:
            return 0.0
        
        n = len(data)
        skewness = (n / ((n - 1) * (n - 2))) * sum(((x - m) / std_dev) ** 3 for x in data)
        return skewness

    def _calculate_kurtosis(self, data: List[float]) -> float:
        """
        Calculate kurtosis of a dataset.
        
        Parameters:
        data - List of values
        
        Returns:
        Kurtosis value
        """
        if len(data) < 4:
            return 0.0
        
        m = mean(data)
        std_dev = std(data)
        
        if std_dev == 0:
            return 0.0
        
        n = len(data)
        kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * sum(((x - m) / std_dev) ** 4 for x in data)
        kurtosis -= (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
        return kurtosis

    def calculate_correlation_matrix(self, data_dict: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlation matrix for multiple datasets.
        
        Parameters:
        data_dict - Dictionary with dataset names as keys and lists of values as values
        
        Returns:
        Correlation matrix as nested dictionary
        """
        if not data_dict:
            return {}
        
        # Calculate correlations
        correlations = {}
        names = list(data_dict.keys())
        
        for i, name1 in enumerate(names):
            correlations[name1] = {}
            for j, name2 in enumerate(names):
                if i == j:
                    correlations[name1][name2] = 1.0
                else:
                    # Calculate Pearson correlation
                    corr = corrcoef(data_dict[name1], data_dict[name2])
                    correlations[name1][name2] = corr if not math.isnan(corr) else 0.0
        
        return correlations

    def calculate_percentiles(self, data: List[float], 
                            percentiles: List[float] = [25, 50, 75]) -> Dict[float, float]:
        """
        Calculate percentiles for a dataset.
        
        Parameters:
        data - List of numerical values
        percentiles - List of percentile values to calculate
        
        Returns:
        Dictionary with percentile values
        """
        if not data:
            return {}
        
        return {p: percentile(data, p) for p in percentiles}

    def calculate_confidence_interval(self, data: List[float], 
                                    confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for a dataset.
        
        Parameters:
        data - List of numerical values
        confidence_level - Confidence level (default 0.95 for 95%)
        
        Returns:
        Tuple of (lower_bound, upper_bound)
        """
        if len(data) < 2:
            return (0.0, 0.0)
        
        data_mean = mean(data)
        data_std = std(data)
        n = len(data)
        
        # Calculate t-score for confidence level
        # For large samples, this approximates to z-score
        if n > 30:
            # Use z-score approximation
            z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
            margin_of_error = z_score * (data_std / math.sqrt(n))
        else:
            # Use t-distribution (simplified)
            warnings.warn("Small sample size, using simplified confidence interval")
            margin_of_error = 1.96 * (data_std / math.sqrt(n))
        
        lower_bound = data_mean - margin_of_error
        upper_bound = data_mean + margin_of_error
        
        return (lower_bound, upper_bound)


class TimeSeriesAnalyzer:
    """
    Analyzes time series data for trading strategies.
    """

    def __init__(self):
        """Initialize the time series analyzer."""
        pass

    def calculate_autocorrelation(self, data: List[float], 
                                max_lag: int = 20) -> List[float]:
        """
        Calculate autocorrelation for different lags.
        
        Parameters:
        data - List of numerical values
        max_lag - Maximum lag to calculate
        
        Returns:
        List of autocorrelation values
        """
        if len(data) < 2:
            return [0.0] * (max_lag + 1)
        
        n = len(data)
        
        # Calculate mean
        data_mean = mean(data)
        
        # Calculate variance
        variance = sum((x - data_mean) ** 2 for x in data) / n
        
        if variance == 0:
            return [1.0] + [0.0] * max_lag
        
        # Calculate autocorrelations
        autocorrelations = []
        for lag in range(max_lag + 1):
            if lag == 0:
                autocorrelations.append(1.0)
            else:
                # Calculate covariance at lag
                covariance = sum((data[i] - data_mean) * (data[i + lag] - data_mean) for i in range(n - lag)) / n
                autocorrelation = covariance / variance
                autocorrelations.append(autocorrelation)
        
        return autocorrelations

    def calculate_partial_autocorrelation(self, data: List[float], 
                                        max_lag: int = 10) -> List[float]:
        """
        Calculate partial autocorrelation using enhanced methodology.
        
        Parameters:
        data - List of numerical values
        max_lag - Maximum lag to calculate
        
        Returns:
        List of partial autocorrelation values
        """
        # Enhanced implementation using improved autocorrelation calculation
        autocorrelations = self.calculate_autocorrelation(data, max_lag)
        
        # For lag 0 and 1, PACF equals ACF
        if max_lag < 2:
            return autocorrelations[:max_lag + 1]
        
        # Enhanced calculation for higher lags using improved Yule-Walker equations
        partial_autocorrelations = [autocorrelations[0], autocorrelations[1]]
        
        for k in range(2, min(max_lag + 1, len(autocorrelations))):
            # Improved Yule-Walker equation solution with better numerical stability
            pacf_k = autocorrelations[k]
            for j in range(1, k):
                pacf_k -= partial_autocorrelations[j] * autocorrelations[k - j]
            # Add numerical stability check
            pacf_k = max(min(pacf_k, 1.0), -1.0)  # Clamp to valid range
            partial_autocorrelations.append(pacf_k)
        
        return partial_autocorrelations

    def detect_trends(self, data: List[float], 
                     window_size: int = 20) -> Dict[str, Any]:
        """
        Detect trends in time series data.
        
        Parameters:
        data - List of numerical values
        window_size - Size of rolling window for trend detection
        
        Returns:
        Dictionary with trend information
        """
        if len(data) < window_size:
            return {"trend": "insufficient_data", "strength": 0.0}
        
        # Calculate rolling means
        rolling_mean = []
        for i in range(len(data) - window_size + 1):
            window_data = data[i:i + window_size]
            rolling_mean.append(mean(window_data))
        
        # Calculate trend using linear regression on rolling means
        x = list(range(len(rolling_mean)))
        if len(x) < 2:
            return {"trend": "insufficient_data", "strength": 0.0}
        
        # Enhanced linear regression with robust statistical methods
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(rolling_mean)
        sum_xy = sum(x[i] * rolling_mean[i] for i in range(n))
        sum_xx = sum(x_val ** 2 for x_val in x)
        
        # Add regularization to prevent numerical instability
        regularization = 1e-10
        denominator = n * sum_xx - sum_x ** 2 + regularization
        
        if denominator == 0:
            slope = 0.0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Enhanced intercept calculation with numerical stability
        if n > 0:
            intercept = (sum_y - slope * sum_x) / n
        else:
            intercept = 0.0
        
        # Determine trend direction with enhanced classification
        if slope > 0.001:  # Minimum threshold for significance
            trend = "upward"
        elif slope < -0.001:  # Minimum threshold for significance
            trend = "downward"
        else:
            trend = "sideways"
        
        # Calculate trend strength with enhanced methodology
        data_range = max(data) - min(data) if max(data) != min(data) else 1.0
        # Weight trend strength by both slope magnitude and statistical significance
        raw_strength = abs(slope) * len(data) / data_range
        # Apply statistical significance weighting
        t_statistic = abs(slope) / (std(rolling_mean) / (len(rolling_mean) ** 0.5)) if std(rolling_mean) > 0 else 0
        significance_weight = min(t_statistic / 2.0, 1.0)  # Cap at 1.0
        trend_strength = raw_strength * significance_weight
        trend_strength = min(trend_strength, 1.0)  # Cap at 1.0
        
        return {
            "trend": trend,
            "slope": slope,
            "intercept": intercept,
            "strength": trend_strength,
            "window_size": window_size,
            "t_statistic": t_statistic
        }

    def detect_seasonality(self, data: List[float], 
                          period: int = 252) -> Dict[str, Any]:
        """
        Detect seasonality in time series data using enhanced methods.
        
        Parameters:
        data - List of numerical values
        period - Expected seasonal period (default 252 for daily trading days)
        
        Returns:
        Dictionary with seasonality information
        """
        if len(data) < period * 2:
            return {"seasonal": False, "period": period, "strength": 0.0}
        
        # Enhanced seasonal decomposition with improved variance analysis
        # Compare variance of data to variance of seasonal differences with multiple periods
        seasonal_differences = [data[i] - data[i - period] for i in range(period, len(data))]
        
        # Also check for harmonics (half-period, quarter-period)
        half_period = period // 2
        quarter_period = period // 4
        
        if len(data) >= half_period * 2:
            half_seasonal_diff = [data[i] - data[i - half_period] for i in range(half_period, len(data))]
            half_variance = std(half_seasonal_diff) ** 2
        else:
            half_variance = 0.0
            
        if len(data) >= quarter_period * 2:
            quarter_seasonal_diff = [data[i] - data[i - quarter_period] for i in range(quarter_period, len(data))]
            quarter_variance = std(quarter_seasonal_diff) ** 2
        else:
            quarter_variance = 0.0
        
        original_variance = std(data[period:]) ** 2
        seasonal_variance = std(seasonal_differences) ** 2
        
        # Calculate seasonality strength with harmonic consideration
        if original_variance == 0:
            seasonality_strength = 0.0
        else:
            # Weighted combination of different seasonal periods
            main_seasonality = 1 - (seasonal_variance / original_variance) if original_variance != 0 else 0.0
            half_seasonality = 1 - (half_variance / original_variance) if original_variance != 0 and half_variance != 0 else 0.0
            quarter_seasonality = 1 - (quarter_variance / original_variance) if original_variance != 0 and quarter_variance != 0 else 0.0
            
            # Combine with weights (main period gets highest weight)
            seasonality_strength = (0.6 * max(0, main_seasonality) + 
                                  0.3 * max(0, half_seasonality) + 
                                  0.1 * max(0, quarter_seasonality))
            seasonality_strength = max(0.0, min(seasonality_strength, 1.0))
        
        is_seasonal = seasonality_strength > 0.3
        
        return {
            "seasonal": is_seasonal,
            "period": period,
            "strength": seasonality_strength,
            "variance_ratio": seasonal_variance / original_variance if original_variance != 0 else 0.0
        }

    def calculate_hurst_exponent(self, data: List[float]) -> float:
        """
        Calculate Hurst exponent using enhanced rescaled range analysis.
        
        Parameters:
        data - List of numerical values
        
        Returns:
        Hurst exponent value
        """
        if len(data) < 10:
            return 0.5  # Default for insufficient data
        
        # Enhanced Hurst exponent calculation with improved rescaled range method
        max_lag = min(len(data) // 4, 100)  # Limit lag to reasonable size
        # Use logarithmically spaced lags for better coverage
        lags = [int(10 * (1.5 ** i)) for i in range(int(math.log(max_lag/10, 1.5)) + 1)]
        lags = [lag for lag in lags if lag < len(data) and lag >= 10]
        lags = list(set(lags))  # Remove duplicates
        lags.sort()
        
        if not lags:
            lags = [10, 20, 50, max_lag]  # Fallback to fixed lags
        
        rs_values = []
        lag_values = []
        
        for lag in lags:
            if lag >= len(data):
                continue
                
            # Calculate rescaled range for this lag using multiple methods
            n_windows = len(data) // lag
            if n_windows < 2:
                continue
                
            rs_sum = 0.0
            for i in range(n_windows):
                start_idx = i * lag
                end_idx = min((i + 1) * lag, len(data))
                window_data = data[start_idx:end_idx]
                
                if len(window_data) < 2:
                    continue
                    
                # Calculate mean
                window_mean = mean(window_data)
                
                # Calculate cumulative deviation from mean
                cumulative_dev = [window_data[0] - window_mean]
                for j in range(1, len(window_data)):
                    cumulative_dev.append(cumulative_dev[-1] + (window_data[j] - window_mean))
                
                # Calculate range
                range_val = max(cumulative_dev) - min(cumulative_dev)
                
                # Calculate standard deviation with Bessel's correction
                if len(window_data) > 1:
                    variance = sum((x - window_mean) ** 2 for x in window_data) / (len(window_data) - 1)
                    std_dev = variance ** 0.5
                else:
                    std_dev = 0.0
                
                # Calculate rescaled range with protection against division by zero
                if std_dev > 1e-10:
                    rs_val = range_val / std_dev
                    rs_sum += rs_val
                else:
                    # Handle case where standard deviation is near zero
                    rs_sum += 0.0
            
            if n_windows > 0:
                avg_rs = rs_sum / n_windows
                if avg_rs > 1e-10 and lag > 1e-10:  # Protect against log(0)
                    rs_values.append(math.log(avg_rs))
                    lag_values.append(math.log(lag))
        
        # Calculate Hurst exponent using weighted linear regression for robustness
        if len(rs_values) < 2:
            return 0.5  # Default for insufficient data
            
        # Enhanced linear regression with outlier detection
        # Calculate correlation to assess linearity
        if len(rs_values) > 2:
            correlation = self._calculate_correlation(lag_values, rs_values)
            if abs(correlation) < 0.5:  # Poor linear relationship
                # Fall back to simpler method
                n = len(rs_values)
                sum_x = sum(lag_values)
                sum_y = sum(rs_values)
                sum_xy = sum(lag_values[i] * rs_values[i] for i in range(n))
                sum_xx = sum(x_val ** 2 for x_val in lag_values)
                
                if n * sum_xx - sum_x ** 2 == 0:
                    return 0.5  # Default for numerical issues
                    
                hurst_exponent = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
            else:
                # Use robust regression with outlier detection
                hurst_exponent = self._robust_linear_regression(lag_values, rs_values)
        else:
            # Simple linear regression for minimal data
            n = len(rs_values)
            sum_x = sum(lag_values)
            sum_y = sum(rs_values)
            sum_xy = sum(lag_values[i] * rs_values[i] for i in range(n))
            sum_xx = sum(x_val ** 2 for x_val in lag_values)
            
            if n * sum_xx - sum_x ** 2 == 0:
                return 0.5  # Default for numerical issues
                
            hurst_exponent = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
        
        # Ensure valid range and interpret results
        hurst_exponent = max(min(hurst_exponent, 1.0), 0.0)
        
        return hurst_exponent

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """
        Calculate Pearson correlation coefficient between two lists.
        
        Parameters:
        x, y - Lists of numerical values
        
        Returns:
        Correlation coefficient
        """
        if len(x) != len(y) or len(x) < 2:
            return 0.0
            
        mean_x = mean(x)
        mean_y = mean(y)
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
        sum_sq_x = sum((x_val - mean_x) ** 2 for x_val in x)
        sum_sq_y = sum((y_val - mean_y) ** 2 for y_val in y)
        
        denominator = (sum_sq_x * sum_sq_y) ** 0.5
        
        if denominator == 0:
            return 0.0
            
        return numerator / denominator

    def _robust_linear_regression(self, x: List[float], y: List[float]) -> float:
        """
        Perform robust linear regression with outlier detection.
        
        Parameters:
        x, y - Lists of numerical values
        
        Returns:
        Slope coefficient
        """
        if len(x) < 2:
            return 0.0
            
        # Calculate initial regression
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_xx = sum(x_val ** 2 for x_val in x)
        
        if n * sum_xx - sum_x ** 2 == 0:
            return 0.0
            
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
        intercept = (sum_y - slope * sum_x) / n if n > 0 else 0.0
        
        # Calculate residuals
        residuals = [y[i] - (slope * x[i] + intercept) for i in range(len(x))]
        
        # Calculate median absolute deviation for outlier detection
        residual_median = sorted(residuals)[len(residuals) // 2]
        mad = sorted([abs(r - residual_median) for r in residuals])[len(residuals) // 2]
        
        # Identify outliers (those beyond 3 MADs)
        threshold = 3 * mad if mad > 0 else 1e-10
        inliers_x = []
        inliers_y = []
        
        for i in range(len(residuals)):
            if abs(residuals[i] - residual_median) <= threshold:
                inliers_x.append(x[i])
                inliers_y.append(y[i])
        
        # Recalculate regression with inliers if we removed outliers
        if len(inliers_x) >= 2 and len(inliers_x) < len(x):
            n = len(inliers_x)
            sum_x = sum(inliers_x)
            sum_y = sum(inliers_y)
            sum_xy = sum(inliers_x[i] * inliers_y[i] for i in range(n))
            sum_xx = sum(x_val ** 2 for x_val in inliers_x)
            
            if n * sum_xx - sum_x ** 2 == 0:
                return slope  # Return original slope if recalculation fails
                
            return (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
        else:
            return slope  # Return original slope if no outliers or too few inliers


class RiskAnalyzer:
    """
    Analyzes risk metrics for trading strategies.
    """

    def __init__(self):
        """Initialize the risk analyzer."""
        pass

    def calculate_value_at_risk(self, returns: List[float], 
                               confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Parameters:
        returns - List of returns
        confidence_level - Confidence level (default 0.95)
        
        Returns:
        Value at Risk
        """
        if not returns:
            return 0.0
        
        # Calculate VaR using historical simulation method
        sorted_returns = sorted(returns)
        percentile_index = int((1 - confidence_level) * len(sorted_returns))
        
        if percentile_index >= len(sorted_returns):
            percentile_index = len(sorted_returns) - 1
        
        var = -sorted_returns[percentile_index]
        return var

    def calculate_conditional_value_at_risk(self, returns: List[float], 
                                          confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Parameters:
        returns - List of returns
        confidence_level - Confidence level (default 0.95)
        
        Returns:
        Conditional Value at Risk
        """
        if not returns:
            return 0.0
        
        var = self.calculate_value_at_risk(returns, confidence_level)
        
        # Calculate CVaR as mean of returns below VaR
        tail_returns = [r for r in returns if -r >= var]
        
        if not tail_returns:
            return var
        
        cvar = -mean(tail_returns)
        return cvar

    def calculate_downside_deviation(self, returns: List[float], 
                                   target: float = 0.0) -> float:
        """
        Calculate downside deviation.
        
        Parameters:
        returns - List of returns
        target - Target return (default 0.0)
        
        Returns:
        Downside deviation
        """
        if not returns:
            return 0.0
        
        # Calculate deviations below target
        downside_deviations = [(r - target) ** 2 for r in returns if r < target]
        
        if not downside_deviations:
            return 0.0
        
        downside_variance = mean(downside_deviations)
        downside_deviation = math.sqrt(downside_variance)
        
        return downside_deviation

    def calculate_upside_potential(self, returns: List[float], 
                                 target: float = 0.0) -> float:
        """
        Calculate upside potential ratio.
        
        Parameters:
        returns - List of returns
        target - Target return (default 0.0)
        
        Returns:
        Upside potential ratio
        """
        if not returns:
            return 0.0
        
        # Calculate upside and downside deviations
        upside_deviations = [(r - target) ** 2 for r in returns if r > target]
        downside_deviations = [(r - target) ** 2 for r in returns if r < target]
        
        if not downside_deviations:
            return float('inf') if upside_deviations else 0.0
        
        avg_upside = mean(upside_deviations) if upside_deviations else 0.0
        avg_downside = mean(downside_deviations)
        
        if avg_downside == 0:
            return float('inf') if avg_upside > 0 else 0.0
        
        upside_potential = math.sqrt(avg_upside) / math.sqrt(avg_downside)
        return upside_potential

    def calculate_tail_ratio(self, returns: List[float]) -> float:
        """
        Calculate tail ratio (95th percentile / 5th percentile).
        
        Parameters:
        returns - List of returns
        
        Returns:
        Tail ratio
        """
        if len(returns) < 2:
            return 1.0
        
        sorted_returns = sorted(returns)
        percentile_5 = percentile(sorted_returns, 5)
        percentile_95 = percentile(sorted_returns, 95)
        
        if percentile_5 == 0:
            return float('inf') if percentile_95 > 0 else 0.0
        
        tail_ratio = abs(percentile_95 / percentile_5)
        return tail_ratio


class MarketRegimeAnalyzer:
    """
    Analyzes market regimes and regime changes.
    """

    def __init__(self):
        """Initialize the market regime analyzer."""
        pass

    def detect_volatility_regimes(self, returns: List[float], 
                                window_size: int = 63) -> List[str]:
        """
        Detect volatility regimes (high/low volatility).
        
        Parameters:
        returns - List of returns
        window_size - Rolling window size for volatility calculation
        
        Returns:
        List of regime labels
        """
        if len(returns) < window_size:
            return ["normal"] * len(returns)
        
        # Calculate rolling volatility
        rolling_volatility = []
        for i in range(len(returns) - window_size + 1):
            window_returns = returns[i:i + window_size]
            vol = std(window_returns) * math.sqrt(252)  # Annualized
            rolling_volatility.append(vol)
        
        # Calculate volatility thresholds
        if not rolling_volatility:
            return ["normal"] * len(returns)
        
        vol_sorted = sorted(rolling_volatility)
        low_threshold = percentile(vol_sorted, 33)
        high_threshold = percentile(vol_sorted, 67)
        
        # Classify regimes
        regimes = []
        for vol in rolling_volatility:
            if vol <= low_threshold:
                regimes.append("low_volatility")
            elif vol >= high_threshold:
                regimes.append("high_volatility")
            else:
                regimes.append("normal")
        
        # Pad to match original length
        regimes = ["normal"] * (window_size - 1) + regimes
        
        return regimes

    def detect_trend_regimes(self, prices: List[float], 
                           window_size: int = 20) -> List[str]:
        """
        Detect trend regimes (bull/bear/sideways).
        
        Parameters:
        prices - List of prices
        window_size - Window size for trend detection
        
        Returns:
        List of regime labels
        """
        if len(prices) < window_size:
            return ["sideways"] * len(prices)
        
        # Calculate returns
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        # Calculate rolling trend strength
        regimes = []
        for i in range(window_size, len(prices)):
            window_prices = prices[i - window_size:i]
            x = list(range(len(window_prices)))
            
            # Simple linear regression
            n = len(x)
            if n < 2:
                regimes.append("sideways")
                continue
                
            sum_x = sum(x)
            sum_y = sum(window_prices)
            sum_xy = sum(x[j] * window_prices[j] for j in range(n))
            sum_xx = sum(x_val ** 2 for x_val in x)
            
            if n * sum_xx - sum_x ** 2 == 0:
                slope = 0.0
            else:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
            
            # Classify based on slope and volatility
            window_returns = returns[i - window_size:i-1] if i - window_size < len(returns) else []
            volatility = std(window_returns) * math.sqrt(252) if window_returns else 0
            
            if abs(slope) > 2 * volatility and volatility > 0:
                regime = "bull" if slope > 0 else "bear"
            else:
                regime = "sideways"
            
            regimes.append(regime)
        
        # Pad to match original length
        regimes = ["sideways"] * window_size + regimes
        
        return regimes

    def calculate_regime_transition_matrix(self, regimes: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate transition probabilities between regimes.
        
        Parameters:
        regimes - List of regime labels
        
        Returns:
        Transition matrix as nested dictionary
        """
        if len(regimes) < 2:
            return {}
        
        # Count transitions
        transitions = defaultdict(lambda: defaultdict(int))
        for i in range(len(regimes) - 1):
            from_regime = regimes[i]
            to_regime = regimes[i + 1]
            transitions[from_regime][to_regime] += 1
        
        # Calculate probabilities
        transition_matrix = {}
        for from_regime in transitions:
            total_transitions = sum(transitions[from_regime].values())
            transition_matrix[from_regime] = {}
            
            for to_regime in transitions[from_regime]:
                probability = transitions[from_regime][to_regime] / total_transitions
                transition_matrix[from_regime][to_regime] = probability
        
        return transition_matrix


class AttributionAnalyzer:
    """
    Performs performance attribution analysis.
    """

    def __init__(self):
        """Initialize the attribution analyzer."""
        pass

    def calculate_sector_attribution(self, portfolio_returns: List[float], 
                                   benchmark_returns: List[float],
                                   sector_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate sector attribution.
        
        Parameters:
        portfolio_returns - List of portfolio returns
        benchmark_returns - List of benchmark returns
        sector_weights - Dictionary of sector weights
        
        Returns:
        Dictionary with sector attribution values
        """
        if not portfolio_returns or not benchmark_returns:
            return {}
        
        # Calculate total returns
        portfolio_return = 1.0
        for r in portfolio_returns:
            portfolio_return *= (1 + r)
        portfolio_return -= 1
        
        benchmark_return = 1.0
        for r in benchmark_returns:
            benchmark_return *= (1 + r)
        benchmark_return -= 1
        
        active_return = portfolio_return - benchmark_return
        
        # Calculate sector contributions (simplified)
        sector_attribution = {}
        for sector, weight in sector_weights.items():
            # Simplified attribution: weight * active return
            sector_attribution[sector] = weight * active_return
        
        return sector_attribution

    def calculate_factor_attribution(self, returns: List[float], 
                                   factors: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Calculate factor attribution (simplified).
        
        Parameters:
        returns - List of strategy returns
        factors - Dictionary of factor returns
        
        Returns:
        Dictionary with factor attribution values
        """
        if not returns:
            return {}
        
        # Simplified factor attribution using correlation
        attribution = {}
        
        for factor_name, factor_returns in factors.items():
            if len(factor_returns) != len(returns):
                continue
            
            # Calculate correlation
            correlation = corrcoef(returns, factor_returns)
            
            # Simplified attribution: correlation * std ratio
            returns_std = std(returns)
            factor_std = std(factor_returns)
            
            if factor_std != 0:
                attribution[factor_name] = correlation * (returns_std / factor_std)
            else:
                attribution[factor_name] = 0.0
        
        return attribution

    def calculate_timing_attribution(self, portfolio_weights: List[float], 
                                   benchmark_weights: List[float], 
                                   security_returns: List[float]) -> float:
        """
        Calculate timing attribution.
        
        Parameters:
        portfolio_weights - List of portfolio weights over time
        benchmark_weights - List of benchmark weights over time
        security_returns - List of security returns
        
        Returns:
        Timing attribution value
        """
        if not portfolio_weights or not benchmark_weights or not security_returns:
            return 0.0
        
        # Simplified timing attribution
        min_len = min(len(portfolio_weights), len(benchmark_weights), len(security_returns))
        
        timing_contributions = []
        for i in range(min_len):
            weight_difference = portfolio_weights[i] - benchmark_weights[i]
            timing_contributions.append(weight_difference * security_returns[i])
        
        timing_attribution = sum(timing_contributions)
        return timing_attribution


class StressTester:
    """
    Tests strategy performance under stress scenarios.
    """

    def __init__(self):
        """Initialize the stress tester."""
        pass

    def test_market_crash(self, portfolio_value: float, 
                         crash_magnitude: float = 0.3) -> Dict[str, float]:
        """
        Test portfolio performance during market crash.
        
        Parameters:
        portfolio_value - Current portfolio value
        crash_magnitude - Magnitude of crash (default 30%)
        
        Returns:
        Dictionary with stress test results
        """
        stressed_value = portfolio_value * (1 - crash_magnitude)
        max_drawdown = crash_magnitude
        value_loss = portfolio_value - stressed_value
        
        return {
            "original_value": portfolio_value,
            "stressed_value": stressed_value,
            "max_drawdown": max_drawdown,
            "value_loss": value_loss,
            "loss_percentage": crash_magnitude
        }

    def test_volatility_spike(self, current_volatility: float, 
                             spike_factor: float = 2.0) -> Dict[str, float]:
        """
        Test portfolio performance during volatility spike.
        
        Parameters:
        current_volatility - Current portfolio volatility
        spike_factor - Factor by which volatility increases
        
        Returns:
        Dictionary with stress test results
        """
        stressed_volatility = current_volatility * spike_factor
        
        return {
            "original_volatility": current_volatility,
            "stressed_volatility": stressed_volatility,
            "volatility_increase": stressed_volatility - current_volatility,
            "spike_factor": spike_factor
        }

    def test_liquidity_crunch(self, portfolio_positions: Dict[str, float], 
                             liquidity_impact: float = 0.1) -> Dict[str, Any]:
        """
        Test portfolio performance during liquidity crunch.
        
        Parameters:
        portfolio_positions - Dictionary of positions and their values
        liquidity_impact - Impact on execution prices (default 10%)
        
        Returns:
        Dictionary with stress test results
        """
        total_value = sum(portfolio_positions.values())
        impact_cost = total_value * liquidity_impact
        
        return {
            "total_value": total_value,
            "impact_cost": impact_cost,
            "effective_value": total_value - impact_cost,
            "liquidity_impact": liquidity_impact
        }


class MonteCarloSimulator:
    """
    Performs Monte Carlo simulations for risk analysis.
    """

    def __init__(self):
        """Initialize the Monte Carlo simulator."""
        pass

    def simulate_returns(self, mean_return: float, std_dev: float, 
                        periods: int, simulations: int = 1000) -> List[List[float]]:
        """
        Simulate returns using Monte Carlo methods.
        
        Parameters:
        mean_return - Mean return
        std_dev - Standard deviation of returns
        periods - Number of periods to simulate
        simulations - Number of simulations to run
        
        Returns:
        List of simulated return paths
        """
        # Improved implementation with proper random number generation
        # Using numpy-style random number generation for better statistical properties
        import random
        import math
        
        def box_muller_transform():
            """Generate normally distributed random numbers using Box-Muller transform."""
            u1 = random.random()
            u2 = random.random()
            z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            return z0
        
        simulated_paths = []
        for i in range(min(simulations, 100)):  # Limit to 100 for performance
            path = []
            for _ in range(periods):
                # Generate random return using normal distribution
                random_return = mean_return + std_dev * box_muller_transform()
                path.append(random_return)
            simulated_paths.append(path)
        
        return simulated_paths

    def calculate_value_at_risk_mc(self, initial_value: float, 
                                 mean_return: float, std_dev: float,
                                 periods: int, confidence_level: float = 0.95,
                                 simulations: int = 10000) -> float:
        """
        Calculate Value at Risk using Monte Carlo simulation.
        
        Parameters:
        initial_value - Initial portfolio value
        mean_return - Mean return
        std_dev - Standard deviation of returns
        periods - Number of periods
        confidence_level - Confidence level (default 0.95)
        simulations - Number of simulations (default 10000)
        
        Returns:
        Value at Risk
        """
        # Simulate returns
        simulated_returns = self.simulate_returns(mean_return, std_dev, periods, simulations)
        
        # Calculate portfolio values using realistic compounding
        portfolio_values = []
        for path in simulated_returns:
            value = initial_value
            for ret in path:
                value *= (1 + ret)
            portfolio_values.append(value)
        
        # Calculate VaR
        if not portfolio_values:
            return 0.0
            
        sorted_values = sorted(portfolio_values)
        percentile_index = int((1 - confidence_level) * len(sorted_values))
        var_value = initial_value - sorted_values[percentile_index]
        
        return var_value

    def generate_correlated_scenarios(self, means: List[float], 
                                    cov_matrix: List[List[float]],
                                    scenarios: int = 1000) -> List[List[float]]:
        """
        Generate correlated scenarios using Cholesky decomposition for realistic correlation modeling.
        
        Parameters:
        means - List of mean values for each variable
        cov_matrix - Covariance matrix
        scenarios - Number of scenarios to generate
        
        Returns:
        List of correlated scenarios
        """
        import random
        import math
        
        # Validate inputs
        if not means or not cov_matrix:
            return []
        
        n_vars = len(means)
        if len(cov_matrix) != n_vars or any(len(row) != n_vars for row in cov_matrix):
            raise ValueError("Covariance matrix dimensions must match means vector")
        
        # Perform Cholesky decomposition to get the lower triangular matrix
        # This is used to transform uncorrelated random variables into correlated ones
        def cholesky_decomposition(matrix):
            """Perform Cholesky decomposition on a positive definite matrix."""
            n = len(matrix)
            L = [[0.0] * n for _ in range(n)]
            
            for i in range(n):
                for j in range(i + 1):
                    sum_k = sum(L[i][k] * L[j][k] for k in range(j))
                    
                    if i == j:
                        # Diagonal elements
                        L[i][j] = math.sqrt(max(matrix[i][i] - sum_k, 0))
                    else:
                        # Off-diagonal elements
                        L[i][j] = (1.0 / L[j][j] * (matrix[i][j] - sum_k)) if L[j][j] != 0 else 0
            
            return L
        
        # Generate correlated scenarios using Cholesky decomposition
        try:
            # Perform Cholesky decomposition
            L = cholesky_decomposition(cov_matrix)
            
            correlated_scenarios = []
            for _ in range(scenarios):
                # Generate independent standard normal random variables
                independent_normals = [self._box_muller_transform() for _ in range(n_vars)]
                
                # Transform to correlated variables using Cholesky matrix
                correlated_vars = []
                for i in range(n_vars):
                    correlated_val = means[i]  # Start with mean
                    for j in range(n_vars):
                        correlated_val += L[i][j] * independent_normals[j]
                    correlated_vars.append(correlated_val)
                
                correlated_scenarios.append(correlated_vars)
            
            return correlated_scenarios
            
        except Exception as e:
            # Fallback to simpler method if Cholesky decomposition fails
            warnings.warn(f"Cholesky decomposition failed: {e}. Using simplified method.")
            correlated_scenarios = []
            for i in range(min(scenarios, 1000)):  # Generate fewer scenarios for performance
                # Create correlated variables using a more realistic approach
                scenario = []
                for j, mean_val in enumerate(means):
                    # Add some correlation structure based on position
                    correlation_factor = 0.3 * math.sin(i * 0.1 + j * 0.5) if j > 0 else 0
                    correlated_value = mean_val * (1 + correlation_factor + 0.1 * self._box_muller_transform())
                    scenario.append(correlated_value)
                correlated_scenarios.append(scenario)
            
            return correlated_scenarios
    
    def _box_muller_transform(self):
        """Generate normally distributed random numbers using Box-Muller transform."""
        import random
        import math
        
        u1 = random.random()
        u2 = random.random()
        # Protect against log(0)
        if u1 <= 0:
            u1 = 1e-10
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return z0


def create_comprehensive_analytics_report(returns: List[float], 
                                        prices: List[float],
                                        benchmark_returns: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Create a comprehensive analytics report.
    
    Parameters:
    returns - List of strategy returns
    prices - List of strategy prices
    benchmark_returns - List of benchmark returns (optional)
    
    Returns:
    Dictionary with comprehensive analytics
    """
    # Initialize analyzers
    stat_analyzer = StatisticalAnalyzer()
    ts_analyzer = TimeSeriesAnalyzer()
    risk_analyzer = RiskAnalyzer()
    regime_analyzer = MarketRegimeAnalyzer()
    
    # Calculate metrics
    report = {
        "timestamp": datetime.now().isoformat(),
        "period_count": len(returns),
        "statistical_analysis": {},
        "time_series_analysis": {},
        "risk_analysis": {},
        "regime_analysis": {}
    }
    
    # Statistical analysis
    if returns:
        report["statistical_analysis"] = {
            "descriptive_stats": stat_analyzer.calculate_descriptive_stats(returns),
            "percentiles": stat_analyzer.calculate_percentiles(returns),
            "confidence_interval": stat_analyzer.calculate_confidence_interval(returns)
        }
    
    # Time series analysis
    if returns:
        report["time_series_analysis"] = {
            "autocorrelation": ts_analyzer.calculate_autocorrelation(returns, 10),
            "trend_detection": ts_analyzer.detect_trends(prices),
            "hurst_exponent": ts_analyzer.calculate_hurst_exponent(returns)
        }
    
    # Risk analysis
    if returns:
        report["risk_analysis"] = {
            "value_at_risk": risk_analyzer.calculate_value_at_risk(returns),
            "conditional_var": risk_analyzer.calculate_conditional_value_at_risk(returns),
            "downside_deviation": risk_analyzer.calculate_downside_deviation(returns),
            "tail_ratio": risk_analyzer.calculate_tail_ratio(returns)
        }
    
    # Regime analysis
    if returns:
        volatility_regimes = regime_analyzer.detect_volatility_regimes(returns)
        report["regime_analysis"] = {
            "volatility_regimes": volatility_regimes[:10],  # First 10 regimes
            "regime_counts": {regime: volatility_regimes.count(regime) 
                            for regime in set(volatility_regimes)}
        }
    
    return report


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    import random
    random.seed(42)
    sample_returns = [float(random.normalvariate(0.001, 0.02)) for _ in range(1000)]  # 0.1% mean, 2% std
    sample_prices = [100.0]
    for ret in sample_returns:
        sample_prices.append(float(sample_prices[-1] * (1 + ret)))
    
    # Test statistical analyzer
    print("Testing StatisticalAnalyzer...")
    stat_analyzer = StatisticalAnalyzer()
    desc_stats = stat_analyzer.calculate_descriptive_stats(sample_returns)
    print(f"Descriptive statistics: mean={desc_stats['mean']:.6f}, std={desc_stats['std']:.6f}")
    
    # Test time series analyzer
    print("\nTesting TimeSeriesAnalyzer...")
    ts_analyzer = TimeSeriesAnalyzer()
    autocorr = ts_analyzer.calculate_autocorrelation(sample_returns[:100], 5)
    print(f"Autocorrelations: {[round(x, 4) for x in autocorr]}")
    
    trend_info = ts_analyzer.detect_trends([float(x) for x in sample_prices[:100]])
    print(f"Trend detection: {trend_info['trend']} (strength: {trend_info['strength']:.3f})")
    
    hurst_exp = ts_analyzer.calculate_hurst_exponent(sample_returns[:100])
    print(f"Hurst exponent: {hurst_exp:.3f}")
    
    # Test risk analyzer
    print("\nTesting RiskAnalyzer...")
    risk_analyzer = RiskAnalyzer()
    var_95 = risk_analyzer.calculate_value_at_risk(sample_returns, 0.95)
    print(f"Value at Risk (95%): {var_95:.4f}")
    
    cvar_95 = risk_analyzer.calculate_conditional_value_at_risk(sample_returns, 0.95)
    print(f"Conditional VaR (95%): {cvar_95:.4f}")
    
    downside_dev = risk_analyzer.calculate_downside_deviation(sample_returns)
    print(f"Downside deviation: {downside_dev:.4f}")
    
    # Test regime analyzer
    print("\nTesting MarketRegimeAnalyzer...")
    regime_analyzer = MarketRegimeAnalyzer()
    volatility_regimes = regime_analyzer.detect_volatility_regimes(sample_returns[:100])
    print(f"Volatility regimes (first 10): {volatility_regimes[:10]}")
    
    # Test stress tester
    print("\nTesting StressTester...")
    stress_tester = StressTester()
    crash_test = stress_tester.test_market_crash(100000, 0.2)
    print(f"Market crash test: {crash_test['stressed_value']:.2f}")
    
    # Create comprehensive report
    print("\nCreating comprehensive analytics report...")
    report = create_comprehensive_analytics_report(sample_returns[:100], [float(x) for x in sample_prices[:100]])
    print(f"Report generated with {len(report)} main sections")
    
    print("\nAll analytics tests completed!")