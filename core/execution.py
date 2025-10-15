import datetime
from abc import ABC, abstractmethod
import random
from .event import FillEvent, OrderEvent


class ExecutionHandler(ABC):
    """
    Abstract base class for handling order execution.
    """

    @abstractmethod
    def execute_order(self, event: OrderEvent):
        """
        Takes an OrderEvent and executes it, producing
        a FillEvent that gets placed onto the Events queue.
        """
        raise NotImplementedError("Should implement execute_order()")


class SimulatedExecutionHandler(ExecutionHandler):
    """
    Simulates execution of orders in a backtesting environment.
    """

    def __init__(self, events, commission_rate=0.001, slippage_factor=0.0001):
        """
        Initializes the simulated execution handler.
        
        Parameters:
        events - The Queue of Event objects
        commission_rate - Commission rate per trade (default 0.1%)
        slippage_factor - Slippage factor (default 0.01%)
        """
        self.events = events
        self.commission_rate = commission_rate
        self.slippage_factor = slippage_factor
        # Initialize market data simulator
        self.market_data_simulator = MarketDataSimulator()

    def execute_order(self, event: OrderEvent):
        """
        Converts OrderEvents into FillEvents with realistic market conditions,
        including latency, slippage, and fill ratio problems.
        """
        if event.type.name == 'ORDER':
            # Get market price from a realistic market data simulator
            # In a sophisticated implementation, this would come from live market data feeds
            # For backtesting, we simulate realistic market conditions
            # Check if market price is provided in the event attributes
            if event.market_price is not None:
                base_price = event.market_price
            else:
                # Get price from our market data simulator with realistic market dynamics
                base_price = self.market_data_simulator.get_latest_price(event.symbol)
            
            # Apply realistic slippage based on order size, market volatility, and liquidity
            # Larger orders have more slippage due to market impact
            order_size_factor = min(event.quantity / 1000.0, 1.0)  # Normalize order size
            volatility_factor = self.market_data_simulator.get_volatility(event.symbol)
            liquidity_factor = self.market_data_simulator.get_liquidity(event.symbol)
            
            # Dynamic slippage model incorporating multiple market factors
            dynamic_slippage = self.slippage_factor * (
                1 + order_size_factor + 
                volatility_factor + 
                (1 - liquidity_factor)  # Lower liquidity increases slippage
            )
            
            # Apply slippage based on order direction and market conditions
            if event.direction == 'BUY':
                fill_price = base_price * (1 + dynamic_slippage)
            else:
                fill_price = base_price * (1 - dynamic_slippage)
            
            # Calculate commission with tiered pricing model
            # Real brokers often have tiered commission structures
            commission_tier = self._get_commission_tier(int(event.quantity))
            effective_commission_rate = self.commission_rate * commission_tier
            
            # Calculate commission
            commission = effective_commission_rate * event.quantity * fill_price
            
            # Simulate partial fills based on market liquidity
            fill_ratio = self._calculate_fill_probability(int(event.quantity), liquidity_factor)
            actual_quantity = int(event.quantity * fill_ratio)
            
            # Only create fill event if we have a meaningful fill
            if actual_quantity > 0:
                # Create the FillEvent with realistic parameters
                fill_event = FillEvent(
                    symbol=event.symbol,
                    quantity=actual_quantity,
                    direction=event.direction,
                    fill_price=fill_price,
                    commission=commission
                )
                
                # Put the FillEvent onto the events queue
                self.events.put(fill_event)

    def _get_commission_tier(self, quantity: int) -> float:
        """
        Determine commission tier based on order size.
        
        Parameters:
        quantity - Number of shares/contracts
        
        Returns:
        Commission multiplier
        """
        if quantity < 100:
            return 1.2  # Higher commission for small orders
        elif quantity < 1000:
            return 1.0  # Standard commission
        elif quantity < 10000:
            return 0.8  # Volume discount
        else:
            return 0.6  # Large order discount

    def _calculate_fill_probability(self, quantity: int, liquidity_factor: float) -> float:
        """
        Calculate the probability of getting a full fill based on order size and liquidity.
        
        Parameters:
        quantity - Number of shares/contracts
        liquidity_factor - Market liquidity measure (0.0 to 1.0)
        
        Returns:
        Fill probability (0.0 to 1.0)
        """
        # Larger orders are less likely to get fully filled in illiquid markets
        size_impact = min(quantity / 10000.0, 0.8)  # Cap at 80% impact
        fill_prob = liquidity_factor * (1 - size_impact)
        return max(0.1, min(fill_prob, 1.0))  # Minimum 10% fill probability


class MarketDataSimulator:
    """
    Simulates realistic market data for backtesting purposes.
    """
    
    def __init__(self):
        """Initialize the market data simulator."""
        self.price_memory = {}
        self.volatility_memory = {}
        self.liquidity_memory = {}
        
    def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest simulated market price for a symbol.
        
        Parameters:
        symbol - Trading symbol
        
        Returns:
        Latest market price
        """
        # Initialize symbol data if not present
        if symbol not in self.price_memory:
            self.price_memory[symbol] = {
                'price': 100.0 + random.uniform(-20, 20),  # Base price between 80-120
                'timestamp': datetime.datetime.now()
            }
        
        # Simulate realistic price movements
        current_data = self.price_memory[symbol]
        time_elapsed = (datetime.datetime.now() - current_data['timestamp']).total_seconds()
        
        # Apply random walk with mean reversion
        drift = 0.0001  # Small upward drift
        volatility = self.get_volatility(symbol)
        random_shock = random.normalvariate(0, volatility)
        
        # Mean reversion toward $100
        mean_reversion = 0.001 * (100 - current_data['price'])
        
        # Calculate new price
        price_change = drift + random_shock + mean_reversion
        new_price = current_data['price'] * (1 + price_change)
        
        # Update memory
        self.price_memory[symbol] = {
            'price': max(new_price, 0.01),  # Ensure positive price
            'timestamp': datetime.datetime.now()
        }
        
        return self.price_memory[symbol]['price']
    
    def get_volatility(self, symbol: str) -> float:
        """
        Get the current volatility for a symbol.
        
        Parameters:
        symbol - Trading symbol
        
        Returns:
        Volatility measure (0.0 to 1.0)
        """
        if symbol not in self.volatility_memory:
            # Initialize with realistic volatility (0.5% to 3% daily)
            self.volatility_memory[symbol] = random.uniform(0.005, 0.03)
        
        # Randomly fluctuate volatility
        vol_change = random.normalvariate(0, 0.001)
        self.volatility_memory[symbol] = max(0.001, min(
            self.volatility_memory[symbol] + vol_change, 0.1))
        
        return self.volatility_memory[symbol]
    
    def get_liquidity(self, symbol: str) -> float:
        """
        Get the current liquidity for a symbol.
        
        Parameters:
        symbol - Trading symbol
        
        Returns:
        Liquidity measure (0.0 to 1.0)
        """
        if symbol not in self.liquidity_memory:
            # Initialize with realistic liquidity (20% to 90%)
            self.liquidity_memory[symbol] = random.uniform(0.2, 0.9)
        
        # Randomly fluctuate liquidity
        liq_change = random.normalvariate(0, 0.01)
        self.liquidity_memory[symbol] = max(0.1, min(
            self.liquidity_memory[symbol] + liq_change, 1.0))
        
        return self.liquidity_memory[symbol]