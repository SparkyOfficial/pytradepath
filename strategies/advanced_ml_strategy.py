from core.event import SignalEvent
from core.strategy import Strategy
from core.ml import SimpleLinearRegression, FeatureEngineer
from core.risk import PositionSizer, KellyCriterionPositionSizer
from typing import List, Dict
import math


class AdvancedMLStrategy(Strategy):
    """
    Advanced machine learning-based trading strategy that combines
    technical indicators, risk management, and position sizing.
    """

    def __init__(self, symbols: List[str], 
                 short_window: int = 10, 
                 long_window: int = 50,
                 use_ml: bool = True):
        """
        Initializes the advanced ML strategy.
        
        Parameters:
        symbols - List of ticker symbols to trade
        short_window - Short moving average window
        long_window - Long moving average window
        use_ml - Whether to use ML predictions
        """
        super().__init__(symbols)
        self.short_window = short_window
        self.long_window = long_window
        self.use_ml = use_ml
        
        # Initialize position tracking
        self.bought = self._calculate_initial_bought()
        
        # Initialize ML components
        if self.use_ml:
            self.ml_model = SimpleLinearRegression("AdvancedML")
            self.feature_engineer = FeatureEngineer()
            self.is_model_trained = False
        
        # Technical indicators
        self.price_history = {symbol: [] for symbol in symbols}
        self.ma_short = {symbol: [] for symbol in symbols}
        self.ma_long = {symbol: [] for symbol in symbols}
        self.rsi = {symbol: [] for symbol in symbols}
        
        # Risk management
        self.position_sizer = None

    def _calculate_initial_bought(self):
        """
        Adds keys to the bought dictionary for all symbols
        and sets them to 'OUT'.
        """
        bought = {}
        for s in self.symbols:
            bought[s] = 'OUT'
        return bought

    def _calculate_moving_average(self, prices: List[float], window: int) -> float:
        """
        Calculate moving average.
        
        Parameters:
        prices - List of prices
        window - Window size
        
        Returns:
        Moving average
        """
        if len(prices) < window:
            return 0.0
        return sum(prices[-window:]) / window

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """
        Calculate Relative Strength Index.
        
        Parameters:
        prices - List of prices
        period - RSI period
        
        Returns:
        RSI value
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral value
            
        # Calculate price changes
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Separate gains and losses
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        # Calculate average gains and losses
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0  # No losses, RSI is 100
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def _prepare_ml_features(self, symbol: str) -> List[List[float]]:
        """
        Prepare features for ML model.
        
        Parameters:
        symbol - Symbol to prepare features for
        
        Returns:
        Feature matrix
        """
        if len(self.price_history[symbol]) < self.long_window:
            return []
            
        features = []
        prices = self.price_history[symbol]
        
        # Create features for the last data point
        if len(prices) >= self.long_window:
            # Price-based features
            current_price = prices[-1]
            ma_short_val = self._calculate_moving_average(prices, self.short_window)
            ma_long_val = self._calculate_moving_average(prices, self.long_window)
            rsi_val = self._calculate_rsi(prices, 14)
            
            # Technical features
            price_change = (current_price - prices[-2]) / prices[-2] if len(prices) > 1 and prices[-2] != 0 else 0
            ma_diff = (ma_short_val - ma_long_val) / ma_long_val if ma_long_val != 0 else 0
            
            # Volume features (using price volatility as proxy)
            volatility = 0.0
            if len(prices) >= 20:
                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(max(1, len(prices)-20), len(prices)) if prices[i-1] != 0]
                if returns:
                    volatility = math.sqrt(sum(r**2 for r in returns) / len(returns)) if returns else 0
            
            # Create feature vector
            feature_vector = [
                price_change,
                ma_diff,
                rsi_val / 100,  # Normalize RSI
                volatility,
                current_price / 100  # Normalize price
            ]
            
            features.append(feature_vector)
        
        return features

    def _train_ml_model(self, symbol: str):
        """
        Train the ML model with historical data.
        
        Parameters:
        symbol - Symbol to train model for
        """
        if len(self.price_history[symbol]) < self.long_window + 10:
            return
            
        # Prepare training data
        prices = self.price_history[symbol]
        X_train = []
        y_train = []
        
        # Use last 100 data points for training
        start_idx = max(0, len(prices) - 100)
        for i in range(start_idx + self.long_window, len(prices) - 1):
            # Features
            feature_prices = prices[start_idx:i]
            features = self._prepare_ml_features(symbol)
            if features:
                X_train.append(features[0])
                
                # Target: next period return
                target_return = (prices[i+1] - prices[i]) / prices[i] if prices[i] != 0 else 0
                y_train.append(target_return)
        
        # Train model if we have data
        if X_train and y_train:
            try:
                self.ml_model.train(X_train, y_train)
                self.is_model_trained = True
            except Exception as e:
                print(f"Error training ML model for {symbol}: {e}")

    def _get_ml_prediction(self, symbol: str) -> float:
        """
        Get ML prediction for a symbol.
        
        Parameters:
        symbol - Symbol to predict
        """
        if not self.use_ml or not self.is_model_trained:
            return 0.0
            
        try:
            features = self._prepare_ml_features(symbol)
            if features:
                prediction = self.ml_model.predict(features)
                return prediction[0] if prediction else 0.0
        except Exception as e:
            print(f"Error getting ML prediction for {symbol}: {e}")
            
        return 0.0

    def set_position_sizer(self, position_sizer: PositionSizer):
        """
        Set the position sizer for the strategy.
        
        Parameters:
        position_sizer - Position sizer to use
        """
        self.position_sizer = position_sizer

    def calculate_signals(self, event):
        """
        Generates signals based on advanced ML and technical analysis.
        """
        if event.type.name == 'MARKET':
            for symbol in self.symbols:
                # Get the latest bars for the symbol
                bars = event.data.get(symbol, [])
                
                if not bars:
                    continue
                    
                # Update price history
                for bar in bars:
                    self.price_history[symbol].append(bar['close'])
                
                # Keep only recent history
                if len(self.price_history[symbol]) > 200:
                    self.price_history[symbol] = self.price_history[symbol][-200:]
                
                # Check if we have enough data
                if len(self.price_history[symbol]) >= self.long_window:
                    current_price = self.price_history[symbol][-1]
                    prices = self.price_history[symbol]
                    
                    # Calculate technical indicators
                    ma_short_val = self._calculate_moving_average(prices, self.short_window)
                    ma_long_val = self._calculate_moving_average(prices, self.long_window)
                    rsi_val = self._calculate_rsi(prices, 14)
                    
                    # Update indicator history
                    self.ma_short[symbol].append(ma_short_val)
                    self.ma_long[symbol].append(ma_long_val)
                    self.rsi[symbol].append(rsi_val)
                    
                    # Keep only recent history
                    if len(self.ma_short[symbol]) > 50:
                        self.ma_short[symbol] = self.ma_short[symbol][-50:]
                        self.ma_long[symbol] = self.ma_long[symbol][-50:]
                        self.rsi[symbol] = self.rsi[symbol][-50:]
                    
                    # Train ML model periodically
                    if self.use_ml and len(self.price_history[symbol]) % 20 == 0:
                        self._train_ml_model(symbol)
                    
                    # Get ML prediction
                    ml_prediction = self._get_ml_prediction(symbol)
                    
                    # Generate signals based on multiple factors
                    signal_strength = 0.0
                    signal_type = None
                    
                    # Technical analysis signals
                    tech_signal = 0.0
                    if ma_short_val > ma_long_val:
                        tech_signal = 1.0  # Bullish
                    elif ma_short_val < ma_long_val:
                        tech_signal = -1.0  # Bearish
                    
                    # RSI signals
                    rsi_signal = 0.0
                    if rsi_val < 30:
                        rsi_signal = 1.0  # Oversold
                    elif rsi_val > 70:
                        rsi_signal = -1.0  # Overbought
                    
                    # Combine signals
                    combined_signal = (tech_signal + rsi_signal) / 2
                    
                    # Incorporate ML prediction
                    if self.use_ml:
                        final_signal = (combined_signal + ml_prediction) / 2
                    else:
                        final_signal = combined_signal
                    
                    # Determine action based on final signal
                    if final_signal > 0.3 and self.bought[symbol] == 'OUT':
                        # Strong buy signal
                        signal_type = 'BUY'
                        signal_strength = min(1.0, abs(final_signal))
                    elif final_signal < -0.3 and self.bought[symbol] == 'OUT':
                        # Strong sell signal
                        signal_type = 'SELL'
                        signal_strength = min(1.0, abs(final_signal))
                    elif abs(final_signal) < 0.1 and self.bought[symbol] != 'OUT':
                        # Weak signal, exit position
                        signal_type = 'EXIT'
                        signal_strength = 1.0
                    
                    # Generate signal event
                    if signal_type:
                        signal = SignalEvent(symbol, signal_type, signal_strength)
                        self.events_queue.put(signal)
                        
                        # Update position tracking
                        if signal_type == 'BUY':
                            self.bought[symbol] = 'LONG'
                        elif signal_type == 'SELL':
                            self.bought[symbol] = 'SHORT'
                        elif signal_type == 'EXIT':
                            self.bought[symbol] = 'OUT'