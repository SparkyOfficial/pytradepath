"""
Advanced risk management module for the pytradepath framework.
"""

import math
from typing import Dict, List, Tuple, Optional
from .event import SignalEvent, OrderEvent
from .portfolio import Portfolio
from .risk import RiskManager, PositionSizer


class AdvancedRiskManager(RiskManager):
    """
    An advanced risk manager that combines multiple risk management techniques.
    """

    def __init__(self, portfolio: Portfolio, 
                 max_positions: int = 10,
                 max_percent_per_position: float = 0.1,
                 stop_loss_percent: float = 0.05,
                 max_sector_exposure: float = 0.3,
                 max_correlation: float = 0.8,
                 min_liquidity: float = 1000000):
        """
        Initialize the advanced risk manager.
        
        Parameters:
        portfolio - The portfolio to manage risk for
        max_positions - Maximum number of positions allowed
        max_percent_per_position - Maximum percent of portfolio per position
        stop_loss_percent - Stop loss percentage
        max_sector_exposure - Maximum exposure to any single sector
        max_correlation - Maximum allowed correlation between assets
        min_liquidity - Minimum required daily liquidity per position
        """
        super().__init__(portfolio)
        self.max_positions = max_positions
        self.max_percent_per_position = max_percent_per_position
        self.stop_loss_percent = stop_loss_percent
        self.max_sector_exposure = max_sector_exposure
        self.max_correlation = max_correlation
        self.min_liquidity = min_liquidity
        
        # Initialize risk components
        self.stop_losses = {}
        self.sector_data = {}
        self.liquidity_data = {}
        self.correlations = {}

    def set_sector_data(self, symbol: str, sector: str):
        """
        Set sector data for a symbol.
        
        Parameters:
        symbol - The symbol
        sector - The sector
        """
        self.sector_data[symbol] = sector

    def set_liquidity_data(self, symbol: str, daily_liquidity: float):
        """
        Set liquidity data for a symbol.
        
        Parameters:
        symbol - The symbol
        daily_liquidity - Daily liquidity in dollars
        """
        self.liquidity_data[symbol] = daily_liquidity

    def set_correlations(self, correlations: Dict):
        """
        Set correlation data.
        
        Parameters:
        correlations - Dictionary of correlations
        """
        self.correlations = correlations

    def modify_signals(self, signals: List[SignalEvent]) -> List[SignalEvent]:
        """
        Modify signals based on comprehensive risk management rules.
        """
        # Check position limits
        current_positions = sum(1 for pos in self.portfolio.current_positions.values() if pos != 0)
        
        if current_positions >= self.max_positions:
            print("Position limit reached, not allowing new positions")
            return []
        
        # Check sector exposure
        risky_sectors = self._check_sector_risk()
        if risky_sectors:
            print(f"High sector exposure in: {risky_sectors}")
            # Filter out signals in risky sectors
            filtered_signals = []
            for signal in signals:
                sector = self.sector_data.get(signal.symbol, 'Unknown')
                if sector not in risky_sectors:
                    filtered_signals.append(signal)
            signals = filtered_signals
        
        # Check correlation risk
        high_correlations = self._check_correlation_risk()
        if high_correlations:
            print(f"High correlations detected: {high_correlations}")
            # Filter out signals that would increase correlation risk
            filtered_signals = []
            for signal in signals:
                # Simplified check - in practice, you would do more sophisticated analysis
                filtered_signals.append(signal)
            signals = filtered_signals
        
        return signals

    def modify_orders(self, orders: List[OrderEvent]) -> List[OrderEvent]:
        """
        Modify orders based on risk management rules.
        """
        modified_orders = []
        
        for order in orders:
            # Check position size limits
            if not self._check_position_size_limit(order):
                print(f"Order for {order.symbol} exceeds position size limit")
                continue
                
            # Check liquidity risk
            if not self._check_liquidity_risk(order):
                print(f"Order for {order.symbol} has liquidity risk")
                continue
                
            # Set stop loss for new positions
            if self._is_new_position(order):
                entry_price = 100.0  # Simplified - in practice, get real price
                self._set_stop_loss(order.symbol, entry_price)
            
            modified_orders.append(order)
        
        return modified_orders

    def _check_position_size_limit(self, order: OrderEvent) -> bool:
        """
        Check if order exceeds position size limits.
        
        Parameters:
        order - Order to check
        
        Returns:
        True if order is within limits, False otherwise
        """
        portfolio_value = self.portfolio.current_holdings['total']
        max_position_value = portfolio_value * self.max_percent_per_position
        
        order_value = order.quantity * 100  # Simplified - use real price in practice
        
        return order_value <= max_position_value

    def _check_liquidity_risk(self, order: OrderEvent) -> bool:
        """
        Check if order has liquidity risk.
        
        Parameters:
        order - Order to check
        
        Returns:
        True if order is within liquidity limits, False otherwise
        """
        daily_liquidity = self.liquidity_data.get(order.symbol, 0)
        order_value = order.quantity * 100  # Simplified - use real price in practice
        
        # Check if order value exceeds 10% of daily liquidity
        return order_value <= daily_liquidity * 0.1

    def _is_new_position(self, order: OrderEvent) -> bool:
        """
        Check if order represents a new position.
        
        Parameters:
        order - Order to check
        
        Returns:
        True if order is for a new position, False otherwise
        """
        current_position = self.portfolio.current_positions.get(order.symbol, 0)
        
        if order.direction == 'BUY':
            return current_position <= 0
        elif order.direction == 'SELL':
            return current_position >= 0
        return False

    def _set_stop_loss(self, symbol: str, entry_price: float):
        """
        Set a stop-loss for a position.
        
        Parameters:
        symbol - The symbol to set stop-loss for
        entry_price - The entry price of the position
        """
        if order.direction == 'BUY':
            stop_loss_price = entry_price * (1 - self.stop_loss_percent)
        else:  # SELL
            stop_loss_price = entry_price * (1 + self.stop_loss_percent)
            
        self.stop_losses[symbol] = stop_loss_price

    def _check_sector_risk(self) -> List[str]:
        """
        Check for excessive sector exposure.
        
        Returns:
        List of sectors with excessive exposure
        """
        sector_exposures = self._calculate_sector_exposures()
        risky_sectors = []
        
        for sector, exposure in sector_exposures.items():
            if exposure > self.max_sector_exposure:
                risky_sectors.append(sector)
                
        return risky_sectors

    def _calculate_sector_exposures(self) -> Dict[str, float]:
        """
        Calculate portfolio exposure by sector.
        
        Returns:
        Dictionary of sector exposures
        """
        sector_exposures = {}
        total_value = self.portfolio.current_holdings['total']
        
        if total_value == 0:
            return sector_exposures
            
        # Calculate exposure by sector
        for symbol, position in self.portfolio.current_positions.items():
            sector = self.sector_data.get(symbol, 'Unknown')
            position_value = position * 100  # Simplified - use real price in practice
            
            if sector in sector_exposures:
                sector_exposures[sector] += position_value
            else:
                sector_exposures[sector] = position_value
        
        # Convert to percentages
        for sector in sector_exposures:
            sector_exposures[sector] = sector_exposures[sector] / total_value
            
        return sector_exposures

    def _check_correlation_risk(self) -> List[Tuple[str, str]]:
        """
        Check for high correlations between assets.
        
        Returns:
        List of highly correlated symbol pairs
        """
        high_correlations = []
        
        # Check all pairs
        for symbol1 in self.correlations:
            for symbol2 in self.correlations[symbol1]:
                if symbol1 != symbol2:  # Avoid duplicates
                    correlation = self.correlations[symbol1][symbol2]
                    if abs(correlation) > self.max_correlation:
                        high_correlations.append((symbol1, symbol2, correlation))
        
        return high_correlations


class AdaptivePositionSizer(PositionSizer):
    """
    A position sizer that adapts position size based on multiple factors.
    """

    def __init__(self, portfolio: Portfolio, 
                 base_percent: float = 0.02,
                 volatility_adjustment: bool = True,
                 risk_adjustment: bool = True):
        """
        Initialize the adaptive position sizer.
        
        Parameters:
        portfolio - The portfolio to size positions for
        base_percent - Base percentage of portfolio for each position
        volatility_adjustment - Whether to adjust for volatility
        risk_adjustment - Whether to adjust for risk factors
        """
        super().__init__(portfolio)
        self.base_percent = base_percent
        self.volatility_adjustment = volatility_adjustment
        self.risk_adjustment = risk_adjustment
        
        # Volatility data (simplified)
        self.volatility_data = {}

    def set_volatility(self, symbol: str, volatility: float):
        """
        Set volatility data for a symbol.
        
        Parameters:
        symbol - The symbol
        volatility - Annualized volatility
        """
        self.volatility_data[symbol] = volatility

    def size_order(self, signal: SignalEvent, market_price: float) -> OrderEvent:
        """
        Size an order adaptively based on multiple factors.
        
        Parameters:
        signal - SignalEvent to size
        market_price - Current market price
        
        Returns:
        Sized OrderEvent
        """
        # Start with base position size
        portfolio_value = self.portfolio.current_holdings['total']
        base_position_value = portfolio_value * self.base_percent
        quantity = int(base_position_value / market_price) if market_price > 0 else 0
        
        # Adjust for volatility if enabled
        if self.volatility_adjustment:
            quantity = self._adjust_for_volatility(signal.symbol, quantity)
        
        # Adjust for risk factors if enabled
        if self.risk_adjustment:
            quantity = self._adjust_for_risk(signal, quantity)
        
        # Ensure minimum quantity of 1
        quantity = max(quantity, 1)
        
        return OrderEvent(
            symbol=signal.symbol,
            order_type='MARKET',
            quantity=quantity,
            direction=signal.signal_type
        )

    def _adjust_for_volatility(self, symbol: str, quantity: int) -> int:
        """
        Adjust position size based on asset volatility.
        
        Parameters:
        symbol - The symbol
        quantity - Base quantity
        
        Returns:
        Adjusted quantity
        """
        volatility = self.volatility_data.get(symbol, 0.15)  # Default 15% annualized
        target_volatility = 0.15  # Target 15% annualized
        
        # Adjust inversely proportional to volatility
        volatility_factor = target_volatility / volatility if volatility > 0 else 1
        adjusted_quantity = int(quantity * volatility_factor)
        
        # Limit adjustment to reasonable bounds
        return max(int(quantity * 0.5), min(adjusted_quantity, int(quantity * 2)))

    def _adjust_for_risk(self, signal: SignalEvent, quantity: int) -> int:
        """
        Adjust position size based on risk factors.
        
        Parameters:
        signal - Signal event
        quantity - Base quantity
        
        Returns:
        Adjusted quantity
        """
        # Reduce position size for weaker signals
        strength_factor = min(signal.strength, 1.0)
        adjusted_quantity = int(quantity * strength_factor)
        
        # Ensure minimum quantity
        return max(adjusted_quantity, 1)