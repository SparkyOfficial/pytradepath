import math
import random
from typing import List, Dict, Tuple, Optional, Union
from abc import ABC, abstractmethod
import warnings


# Simple math functions
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

def sqrt(x):
    """Calculate square root."""
    return math.sqrt(x)

def exp(x):
    """Calculate exponential."""
    return math.exp(x)

def log(x):
    """Calculate natural logarithm."""
    return math.log(x)


class MLModel(ABC):
    """
    Abstract base class for machine learning models.
    """

    def __init__(self, name: str):
        """
        Initialize the ML model.
        
        Parameters:
        name - Name of the model
        """
        self.name = name
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(self, X: List[List[float]], y: List[float]):
        """
        Train the model.
        
        Parameters:
        X - Feature matrix
        y - Target vector
        """
        raise NotImplementedError("Should implement train()")

    @abstractmethod
    def predict(self, X: List[List[float]]) -> List[float]:
        """
        Make predictions.
        
        Parameters:
        X - Feature matrix
        
        Returns:
        Predictions
        """
        raise NotImplementedError("Should implement predict()")

    @abstractmethod
    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        """
        Predict class probabilities.
        
        Parameters:
        X - Feature matrix
        
        Returns:
        Class probabilities
        """
        raise NotImplementedError("Should implement predict_proba()")


class SimpleLinearRegression(MLModel):
    """
    Simple linear regression model.
    """

    def __init__(self, name: str = "LinearRegression"):
        """
        Initialize the linear regression model.
        
        Parameters:
        name - Name of the model
        """
        super().__init__(name)
        self.coefficients = []
        self.intercept = 0.0

    def train(self, X: List[List[float]], y: List[float]):
        """
        Train the model using ordinary least squares.
        
        Parameters:
        X - Feature matrix
        y - Target vector
        """
        if not X or not y or len(X) != len(y):
            raise ValueError("Invalid training data")
        
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        
        # Calculate means
        y_mean = mean(y)
        X_means = [mean([X[i][j] for i in range(n_samples)]) for j in range(n_features)]
        
        # Calculate coefficients using normal equation
        # This is a simplified implementation
        self.coefficients = [0.0] * n_features
        self.intercept = y_mean
        
        # Simple heuristic: correlation-based coefficients
        for j in range(n_features):
            x_vals = [X[i][j] for i in range(n_samples)]
            x_mean = X_means[j]
            
            # Calculate correlation-like coefficient
            if std(x_vals) > 0 and std(y) > 0:
                numerator = sum((x_vals[i] - x_mean) * (y[i] - y_mean) for i in range(n_samples))
                denominator = sum((x_vals[i] - x_mean) ** 2 for i in range(n_samples))
                if denominator > 0:
                    self.coefficients[j] = numerator / denominator
        
        self.is_trained = True

    def predict(self, X: List[List[float]]) -> List[float]:
        """
        Make predictions.
        
        Parameters:
        X - Feature matrix
        
        Returns:
        Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        for row in X:
            pred = self.intercept + sum(self.coefficients[j] * row[j] for j in range(len(self.coefficients)))
            predictions.append(pred)
        
        return predictions

    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        """
        Predict class probabilities (not applicable for regression).
        
        Parameters:
        X - Feature matrix
        
        Returns:
        Class probabilities
        """
        raise ValueError("Probability predictions not supported for regression")


class SimpleDecisionTree(MLModel):
    """
    Simple decision tree classifier.
    """

    def __init__(self, name: str = "DecisionTree", max_depth: int = 3):
        """
        Initialize the decision tree.
        
        Parameters:
        name - Name of the model
        max_depth - Maximum depth of the tree
        """
        super().__init__(name)
        self.max_depth = max_depth
        self.tree = None

    def train(self, X: List[List[float]], y: List[float]):
        """
        Train the decision tree.
        
        Parameters:
        X - Feature matrix
        y - Target vector
        """
        if not X or not y or len(X) != len(y):
            raise ValueError("Invalid training data")
        
        self.tree = self._build_tree(X, y, depth=0)
        self.is_trained = True

    def _build_tree(self, X: List[List[float]], y: List[float], depth: int):
        """
        Build the decision tree recursively.
        
        Parameters:
        X - Feature matrix
        y - Target vector
        depth - Current depth
        
        Returns:
        Tree node
        """
        # Base cases
        if depth >= self.max_depth or len(set(y)) == 1 or not X:
            # Return leaf node with majority class
            if not y:
                return {"prediction": 0}
            
            # Count classes
            class_counts = {}
            for label in y:
                class_counts[label] = class_counts.get(label, 0) + 1
            
            # Find majority class
            majority_class = max(class_counts, key=class_counts.get)
            return {"prediction": majority_class}
        
        # Find best split (simplified)
        best_feature = 0
        best_threshold = mean([row[0] for row in X]) if X else 0
        
        # Split data
        left_X, left_y, right_X, right_y = [], [], [], []
        for i, row in enumerate(X):
            if row[best_feature] <= best_threshold:
                left_X.append(row)
                left_y.append(y[i])
            else:
                right_X.append(row)
                right_y.append(y[i])
        
        # Create node
        node = {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(left_X, left_y, depth + 1),
            "right": self._build_tree(right_X, right_y, depth + 1)
        }
        
        return node

    def predict(self, X: List[List[float]]) -> List[float]:
        """
        Make predictions.
        
        Parameters:
        X - Feature matrix
        
        Returns:
        Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        for row in X:
            pred = self._predict_single(row, self.tree)
            predictions.append(pred)
        
        return predictions

    def _predict_single(self, row: List[float], node: Dict) -> float:
        """
        Predict for a single sample.
        
        Parameters:
        row - Feature vector
        node - Current tree node
        
        Returns:
        Prediction
        """
        if "prediction" in node:
            return node["prediction"]
        
        if row[node["feature"]] <= node["threshold"]:
            return self._predict_single(row, node["left"])
        else:
            return self._predict_single(row, node["right"])

    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        """
        Predict class probabilities.
        
        Parameters:
        X - Feature matrix
        
        Returns:
        Class probabilities
        """
        # Simplified probability prediction
        predictions = self.predict(X)
        probabilities = []
        
        for pred in predictions:
            # Binary classification probabilities
            if pred == 0:
                probabilities.append([1.0, 0.0])
            else:
                probabilities.append([0.0, 1.0])
        
        return probabilities


class FeatureEngineer:
    """
    Engineer features for machine learning models.
    """

    def __init__(self):
        """
        Initialize the feature engineer.
        """
        pass

    def create_features(self, data: List[Dict]) -> List[Dict]:
        """
        Create features from raw market data.
        
        Parameters:
        data - List with OHLCV data
        
        Returns:
        List with engineered features
        """
        if not data:
            return data
        
        features_data = []
        
        # Price-based features
        for i, row in enumerate(data):
            new_row = row.copy()
            
            # Calculate returns if close prices are available
            if i > 0 and 'close' in row and 'close' in data[i-1]:
                try:
                    prev_close = float(data[i-1]['close'])
                    curr_close = float(row['close'])
                    if prev_close != 0:
                        new_row['returns'] = (curr_close - prev_close) / prev_close
                    else:
                        new_row['returns'] = 0
                except (ValueError, TypeError):
                    new_row['returns'] = 0
            else:
                new_row['returns'] = 0
            
            # Moving averages (simplified)
            if i >= 4:
                try:
                    ma_5 = sum(float(data[j]['close']) for j in range(i-4, i+1)) / 5
                    new_row['ma_5'] = ma_5
                except (ValueError, TypeError):
                    new_row['ma_5'] = float(row.get('close', 0))
            else:
                new_row['ma_5'] = float(row.get('close', 0))
            
            if i >= 19:
                try:
                    ma_20 = sum(float(data[j]['close']) for j in range(i-19, i+1)) / 20
                    new_row['ma_20'] = ma_20
                except (ValueError, TypeError):
                    new_row['ma_20'] = float(row.get('close', 0))
            else:
                new_row['ma_20'] = float(row.get('close', 0))
            
            # RSI-like feature (simplified)
            if i >= 14:
                gains = 0
                losses = 0
                for j in range(i-13, i+1):
                    if j > 0:
                        try:
                            curr_close = float(data[j]['close'])
                            prev_close = float(data[j-1]['close'])
                            change = curr_close - prev_close
                            if change > 0:
                                gains += change
                            else:
                                losses -= change
                        except (ValueError, TypeError):
                            pass
                
                if gains + losses > 0:
                    rs = gains / losses if losses > 0 else 100
                    new_row['rsi_like'] = 100 - (100 / (1 + rs))
                else:
                    new_row['rsi_like'] = 50
            else:
                new_row['rsi_like'] = 50
            
            # Volume features
            try:
                volume = float(row.get('volume', 0))
                if i >= 19:
                    vol_ma = sum(float(data[j].get('volume', 0)) for j in range(i-19, i+1)) / 20
                    if vol_ma > 0:
                        new_row['volume_ratio'] = volume / vol_ma
                    else:
                        new_row['volume_ratio'] = 1
                else:
                    new_row['volume_ratio'] = 1
            except (ValueError, TypeError):
                new_row['volume_ratio'] = 1
            
            features_data.append(new_row)
        
        return features_data

    def prepare_data_for_training(self, data: List[Dict], 
                                target_column: str = 'returns',
                                lookback_window: int = 10) -> Tuple[List[List[float]], List[float]]:
        """
        Prepare data for training machine learning models.
        
        Parameters:
        data - List with features
        target_column - Column to predict
        lookback_window - Number of previous time steps to use as features
        
        Returns:
        Feature matrix and target vector
        """
        if not data:
            return [], []
        
        # Select numeric columns for features
        feature_columns = [col for col in data[0].keys() 
                          if col not in ['timestamp', 'symbol', target_column] 
                          and isinstance(data[0][col], (int, float, str)) 
                          and col != target_column]
        
        # Convert to numeric and prepare sequences
        X, y = [], []
        
        for i in range(lookback_window, len(data)):
            # Features: lookback_window previous time steps
            features = []
            for j in range(lookback_window):
                for col in feature_columns:
                    try:
                        features.append(float(data[i-lookback_window+j].get(col, 0)))
                    except (ValueError, TypeError):
                        features.append(0.0)
            
            X.append(features)
            
            # Target: current value
            try:
                target_val = float(data[i].get(target_column, 0))
                y.append(target_val)
            except (ValueError, TypeError):
                y.append(0.0)
        
        return X, y

    def create_classification_target(self, returns: List[float], 
                                   threshold: float = 0.0) -> List[int]:
        """
        Create classification target from returns.
        
        Parameters:
        returns - List of returns
        threshold - Threshold for classification (default: 0 for positive/negative)
        
        Returns:
        Classification target (0 for negative, 1 for positive)
        """
        return [1 if ret > threshold else 0 for ret in returns]


class ModelEvaluator:
    """
    Evaluate machine learning models.
    """

    def __init__(self):
        """
        Initialize the model evaluator.
        """
        pass

    def evaluate_model(self, model: MLModel, X_test: List[List[float]], 
                      y_test: List[float]) -> Dict:
        """
        Evaluate a model.
        
        Parameters:
        model - Trained model
        X_test - Test features
        y_test - Test targets
        
        Returns:
        Dictionary with evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if not y_test or not y_pred or len(y_test) != len(y_pred):
            return {'error': 'Invalid test data'}
        
        # Mean squared error (for regression)
        mse = mean([(y_test[i] - y_pred[i]) ** 2 for i in range(len(y_test))])
        
        # Mean absolute error
        mae = mean([abs(y_test[i] - y_pred[i]) for i in range(len(y_test))])
        
        # R-squared (simplified)
        y_mean = mean(y_test)
        if std(y_test) > 0:
            ss_res = sum((y_test[i] - y_pred[i]) ** 2 for i in range(len(y_test)))
            ss_tot = sum((y_test[i] - y_mean) ** 2 for i in range(len(y_test)))
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            r2 = 0
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
        
        # If model supports probability predictions, calculate additional metrics
        try:
            y_proba = model.predict_proba(X_test)
            # Add more metrics if needed
        except:
            pass
        
        return metrics

    def evaluate_trading_performance(self, predictions: List[float], 
                                   actual_returns: List[float],
                                   transaction_cost: float = 0.001) -> Dict:
        """
        Evaluate trading performance based on predictions.
        
        Parameters:
        predictions - Model predictions
        actual_returns - Actual returns
        transaction_cost - Transaction cost per trade
        
        Returns:
        Dictionary with trading performance metrics
        """
        if len(predictions) != len(actual_returns):
            raise ValueError("Predictions and actual returns must have the same length")
        
        # Calculate strategy returns
        strategy_returns = []
        positions = []
        current_position = 0
        
        for i in range(len(predictions)):
            # Determine new position based on prediction
            # Simplified: positive prediction = long, negative = short
            new_position = 1 if predictions[i] > 0 else -1 if predictions[i] < 0 else 0
            
            # Calculate transaction cost if position changes
            if new_position != current_position:
                cost = transaction_cost * abs(new_position - current_position)
            else:
                cost = 0
            
            # Calculate return
            return_val = new_position * actual_returns[i] - cost
            strategy_returns.append(return_val)
            positions.append(new_position)
            current_position = new_position
        
        # Calculate performance metrics
        if not strategy_returns:
            return {}
        
        total_return = 1.0
        for ret in strategy_returns:
            total_return *= (1 + ret)
        total_return -= 1
        
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1 if len(strategy_returns) > 0 else 0
        volatility = std(strategy_returns) * sqrt(252) if len(strategy_returns) > 1 else 0
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        cumulative = [1.0]
        for ret in strategy_returns:
            cumulative.append(cumulative[-1] * (1 + ret))
        
        running_max = [cumulative[0]]
        for i in range(1, len(cumulative)):
            running_max.append(max(running_max[-1], cumulative[i]))
        
        drawdown = [(cumulative[i] - running_max[i]) / running_max[i] for i in range(len(cumulative))]
        max_drawdown = min(drawdown) if drawdown else 0
        
        win_rate = sum(1 for ret in strategy_returns if ret > 0) / len(strategy_returns) if strategy_returns else 0
        
        positive_returns = [ret for ret in strategy_returns if ret > 0]
        negative_returns = [ret for ret in strategy_returns if ret < 0]
        
        avg_win = mean(positive_returns) if positive_returns else 0
        avg_loss = mean(negative_returns) if negative_returns else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }


class EnsembleModel(MLModel):
    """
    Ensemble of multiple models.
    """

    def __init__(self, name: str, models: List[MLModel], 
                 voting: str = 'average'):
        """
        Initialize the ensemble model.
        
        Parameters:
        name - Name of the ensemble
        models - List of models to ensemble
        voting - Voting method ('average' or 'majority')
        """
        super().__init__(name)
        self.models = models
        self.voting = voting

    def train(self, X: List[List[float]], y: List[float]):
        """
        Train all models in the ensemble.
        
        Parameters:
        X - Feature matrix
        y - Target vector
        """
        for model in self.models:
            model.train(X, y)
        self.is_trained = True

    def predict(self, X: List[List[float]]) -> List[float]:
        """
        Make ensemble predictions.
        
        Parameters:
        X - Feature matrix
        
        Returns:
        Ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.voting == 'majority':
            # Majority voting (for classification)
            predictions = [model.predict(X) for model in self.models]
            # Transpose and vote
            ensemble_preds = []
            for i in range(len(X)):
                votes = [pred[i] for pred in predictions]
                # Simple majority
                ensemble_preds.append(1 if sum(1 for v in votes if v > 0) > len(votes) / 2 else 0)
            return ensemble_preds
        else:
            # Average predictions (for regression)
            predictions = [model.predict(X) for model in self.models]
            # Calculate average for each sample
            ensemble_preds = []
            for i in range(len(X)):
                avg_pred = mean([pred[i] for pred in predictions])
                ensemble_preds.append(avg_pred)
            return ensemble_preds

    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        """
        Predict ensemble class probabilities.
        
        Parameters:
        X - Feature matrix
        
        Returns:
        Ensemble class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        probabilities = [model.predict_proba(X) for model in self.models]
        # Average probabilities
        ensemble_proba = []
        for i in range(len(X)):
            avg_proba = [mean([proba[i][j] for proba in probabilities]) 
                        for j in range(len(probabilities[0][i]))]
            ensemble_proba.append(avg_proba)
        return ensemble_proba


class MLStrategy:
    """
    Machine learning-based trading strategy.
    """

    def __init__(self, model: MLModel, feature_engineer: FeatureEngineer):
        """
        Initialize the ML strategy.
        
        Parameters:
        model - Trained ML model
        feature_engineer - Feature engineer
        """
        self.model = model
        self.feature_engineer = feature_engineer

    def generate_signals(self, data: List[Dict]) -> List[Dict]:
        """
        Generate trading signals based on ML predictions.
        
        Parameters:
        data - List with market data
        
        Returns:
        List with signals
        """
        # Create features
        features_data = self.feature_engineer.create_features(data)
        
        # Prepare data for prediction
        X, _ = self.feature_engineer.prepare_data_for_training(features_data)
        
        # Make predictions
        if not X:
            return []
            
        predictions = self.model.predict(X)
        probabilities = [[0.5, 0.5]] * len(predictions)  # Simplified
        
        # Create signals
        signals = []
        for i, pred in enumerate(predictions):
            signal = {
                'prediction': 1 if pred > 0 else -1 if pred < 0 else 0,
                'probability': max(probabilities[i]) if probabilities[i] else 0.5,
                'timestamp': features_data[i + 10]['timestamp'] if i + 10 < len(features_data) else ''  # Lookback window
            }
            signals.append(signal)
        
        return signals

    def train(self, data: List[Dict], target_column: str = 'returns'):
        """
        Train the ML model.
        
        Parameters:
        data - List with training data
        target_column - Column to predict
        """
        # Create features
        features_data = self.feature_engineer.create_features(data)
        
        # Prepare data for training
        X, y = self.feature_engineer.prepare_data_for_training(features_data, target_column)
        
        # Train model
        if X and y:
            self.model.train(X, y)


def create_standard_models() -> List[MLModel]:
    """
    Create a standard set of ML models for comparison.
    
    Returns:
    List of ML models
    """
    models = [
        SimpleLinearRegression("LinearRegression"),
        SimpleDecisionTree("DecisionTree", max_depth=3)
    ]
    
    return models


def train_and_evaluate_models(models: List[MLModel], 
                             X_train: List[List[float]], y_train: List[float],
                             X_test: List[List[float]], y_test: List[float]) -> Dict:
    """
    Train and evaluate a list of models.
    
    Parameters:
    models - List of models to train and evaluate
    X_train - Training features
    y_train - Training targets
    X_test - Test features
    y_test - Test targets
    
    Returns:
    Dictionary with results
    """
    results = {}
    
    for model in models:
        try:
            print(f"Training {model.name}...")
            model.train(X_train, y_train)
            
            print(f"Evaluating {model.name}...")
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate_model(model, X_test, y_test)
            results[model.name] = metrics
            
        except Exception as e:
            print(f"Error with {model.name}: {e}")
            results[model.name] = {'error': str(e)}
    
    return results