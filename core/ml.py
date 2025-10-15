import math
import random
from typing import List, Dict, Tuple, Optional, Union
from abc import ABC, abstractmethod
import warnings


# Enhanced math functions with numerical stability
def mean(values):
    """Calculate mean of values with improved numerical stability."""
    if not values:
        return 0.0
    # Use Kahan summation for better numerical accuracy
    sum_val = 0.0
    compensation = 0.0
    for val in values:
        y = val - compensation
        t = sum_val + y
        compensation = (t - sum_val) - y
        sum_val = t
    return sum_val / len(values)

def std(values):
    """Calculate standard deviation of values with Bessel's correction."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    # Use numerically stable one-pass algorithm
    sum_sq_diff = 0.0
    sum_diff = 0.0
    for x in values:
        diff = x - m
        sum_diff += diff
        sum_sq_diff += diff * diff
    # Apply Bessel's correction for sample standard deviation
    variance = (sum_sq_diff - sum_diff * sum_diff / len(values)) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))

def sqrt(x):
    """Calculate square root with domain validation."""
    if x < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return math.sqrt(x)

def exp(x):
    """Calculate exponential with overflow protection."""
    if x > 709:  # Prevent overflow
        return float('inf')
    return math.exp(x)

def log(x):
    """Calculate natural logarithm with domain validation."""
    if x <= 0:
        raise ValueError("Cannot calculate logarithm of non-positive number")
    return math.log(x)


class MLModel(ABC):
    """
    Abstract base class for machine learning models with enhanced functionality.
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
        self.feature_names = []
        self.training_history = []

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


class EnhancedLinearRegression(MLModel):
    """
    Enhanced linear regression model with regularization and diagnostics.
    """

    def __init__(self, name: str = "EnhancedLinearRegression", 
                 regularization: float = 0.01, fit_intercept: bool = True):
        """
        Initialize the enhanced linear regression model.
        
        Parameters:
        name - Name of the model
        regularization - L2 regularization parameter (ridge regression)
        fit_intercept - Whether to fit an intercept term
        """
        super().__init__(name)
        self.regularization = regularization
        self.fit_intercept = fit_intercept
        self.coefficients = []
        self.intercept = 0.0
        self.feature_importance = []
        self.r_squared = 0.0
        self.rmse = 0.0

    def train(self, X: List[List[float]], y: List[float]):
        """
        Train the model using regularized least squares with enhanced diagnostics.
        
        Parameters:
        X - Feature matrix
        y - Target vector
        """
        if not X or not y or len(X) != len(y):
            raise ValueError("Invalid training data")
        
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        
        # Validate input data
        if n_samples < n_features + int(self.fit_intercept):
            raise ValueError("Insufficient samples for the number of features")
        
        # Handle intercept term
        if self.fit_intercept:
            # Add column of ones for intercept (manually)
            X_with_intercept = []
            for i in range(n_samples):
                row = [1.0] + X[i]
                X_with_intercept.append(row)
            n_features_with_intercept = n_features + 1
        else:
            X_with_intercept = X
            n_features_with_intercept = n_features
            self.intercept = 0.0
        
        # Apply ridge regression with regularization
        # Create regularization matrix (don't regularize intercept)
        # Solve regularized normal equation using manual matrix operations
        try:
            # Calculate X^T * X
            XtX = self._matrix_multiply_transpose(X_with_intercept, X_with_intercept)
            
            # Add regularization (don't regularize intercept)
            for i in range(n_features_with_intercept):
                for j in range(n_features_with_intercept):
                    if i == j and (not self.fit_intercept or i > 0):
                        XtX[i][j] += self.regularization
            
            # Calculate X^T * y
            Xty = self._matrix_vector_multiply_transpose(X_with_intercept, y)
            
            # Solve linear system using Gaussian elimination
            coefficients = self._solve_linear_system(XtX, Xty)
            
            if self.fit_intercept:
                self.intercept = coefficients[0]
                self.coefficients = coefficients[1:]
            else:
                self.coefficients = coefficients
            
            self.is_trained = True
            
            # Calculate model diagnostics
            self._calculate_diagnostics(X, y)
            
        except Exception as e:
            raise ValueError(f"Failed to solve linear system: {e}")
        
    def _matrix_multiply_transpose(self, A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """Multiply A^T * B manually."""
        rows_A = len(A)
        cols_A = len(A[0]) if A else 0
        cols_B = len(B[0]) if B else 0
        
        result = [[0.0 for _ in range(cols_B)] for _ in range(cols_A)]
        
        for i in range(cols_A):
            for j in range(cols_B):
                for k in range(rows_A):
                    result[i][j] += A[k][i] * B[k][j]
        
        return result

    def _matrix_vector_multiply_transpose(self, A: List[List[float]], b: List[float]) -> List[float]:
        """Multiply A^T * b manually."""
        rows_A = len(A)
        cols_A = len(A[0]) if A else 0
        
        result = [0.0 for _ in range(cols_A)]
        
        for i in range(cols_A):
            for k in range(rows_A):
                result[i] += A[k][i] * b[k]
        
        return result

    def _solve_linear_system(self, A: List[List[float]], b: List[float]) -> List[float]:
        """Solve Ax = b using Gaussian elimination with partial pivoting."""
        n = len(A)
        if n != len(b):
            raise ValueError("Matrix and vector dimensions don't match")
        
        # Create augmented matrix
        augmented = []
        for i in range(n):
            row = A[i][:] + [b[i]]
            augmented.append(row)
        
        # Forward elimination with partial pivoting
        for i in range(n):
            # Find pivot
            max_row = i
            for k in range(i + 1, n):
                if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                    max_row = k
            
            # Swap rows
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
            
            # Make all rows below this one 0 in current column
            for k in range(i + 1, n):
                if augmented[i][i] != 0:
                    factor = augmented[k][i] / augmented[i][i]
                    for j in range(i, n + 1):
                        augmented[k][j] -= factor * augmented[i][j]
        
        # Back substitution
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = augmented[i][n]
            for j in range(i + 1, n):
                x[i] -= augmented[i][j] * x[j]
            if augmented[i][i] != 0:
                x[i] /= augmented[i][i]
        
        return x

    def _calculate_diagnostics(self, X: List[List[float]], y: List[float]):
        """
        Calculate model diagnostics.
        
        Parameters:
        X - Feature matrix
        y - Target vector
        """
        # Only calculate diagnostics if model is trained
        if not self.is_trained:
            return
            
        # Make predictions
        predictions = self.predict(X)
        
        # Calculate R-squared
        y_mean = mean(y)
        ss_res = sum((y[i] - predictions[i]) ** 2 for i in range(len(y)))
        ss_tot = sum((y[i] - y_mean) ** 2 for i in range(len(y)))
        self.r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate RMSE
        mse_values = [(y[i] - predictions[i]) ** 2 for i in range(len(y))]
        self.rmse = sqrt(mean(mse_values))
        
        # Calculate feature importance (absolute coefficients normalized)
        if self.coefficients:
            abs_coeffs = [abs(c) for c in self.coefficients]
            total_abs = sum(abs_coeffs)
            if total_abs > 0:
                self.feature_importance = [c / total_abs for c in abs_coeffs]
            else:
                self.feature_importance = [1.0 / len(self.coefficients)] * len(self.coefficients)

    def predict(self, X: List[List[float]]) -> List[float]:
        """
        Make predictions with enhanced error handling.
        
        Parameters:
        X - Feature matrix
        
        Returns:
        Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if not X:
            return []
        
        # Validate feature dimensions
        if len(X[0]) != len(self.coefficients):
            raise ValueError(f"Expected {len(self.coefficients)} features, got {len(X[0])}")
        
        predictions = []
        for row in X:
            try:
                pred = self.intercept + sum(self.coefficients[j] * row[j] for j in range(len(self.coefficients)))
                predictions.append(pred)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid feature values in row: {e}")
        
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


class EnhancedDecisionTree(MLModel):
    """
    Enhanced decision tree classifier with improved splitting criteria and pruning.
    """

    def __init__(self, name: str = "EnhancedDecisionTree", 
                 max_depth: int = 5, min_samples_split: int = 10, 
                 min_samples_leaf: int = 5, max_features: Optional[int] = None):
        """
        Initialize the enhanced decision tree.
        
        Parameters:
        name - Name of the model
        max_depth - Maximum depth of the tree
        min_samples_split - Minimum samples required to split an internal node
        min_samples_leaf - Minimum samples required to be at a leaf node
        max_features - Number of features to consider when looking for the best split
        """
        super().__init__(name)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree = None
        self.n_features = 0
        self.feature_importances = []

    def train(self, X: List[List[float]], y: List[float]):
        """
        Train the enhanced decision tree with improved splitting.
        
        Parameters:
        X - Feature matrix
        y - Target vector
        """
        if not X or not y or len(X) != len(y):
            raise ValueError("Invalid training data")
        
        self.n_features = len(X[0]) if X else 0
        if self.max_features is None:
            self.max_features = self.n_features
        
        # Convert to appropriate data types
        try:
            X_array = [list(map(float, row)) for row in X]
            y_array = list(map(float, y))
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid data format: {e}")
        
        # Build tree with enhanced splitting criteria
        self.tree = self._build_tree(X_array, y_array, depth=0)
        self.is_trained = True
        
        # Calculate feature importances
        self._calculate_feature_importances()

    def _build_tree(self, X: List[List[float]], y: List[float], depth: int):
        """
        Build the decision tree recursively with enhanced criteria.
        
        Parameters:
        X - Feature matrix
        y - Target vector
        depth - Current depth
        
        Returns:
        Tree node
        """
        n_samples = len(y)
        
        # Base cases
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(set(y)) == 1 or 
            not X):
            # Return leaf node with majority class and additional statistics
            if not y:
                return {"prediction": 0, "samples": 0, "class_distribution": {}}
            
            # Count classes with probabilities
            class_counts = {}
            for label in y:
                class_counts[label] = class_counts.get(label, 0) + 1
            
            # Find majority class and calculate probability
            majority_class = self._get_majority_class(class_counts)
            total_samples = sum(class_counts.values())
            class_probabilities = {cls: count/total_samples for cls, count in class_counts.items()}
            
            return {
                "prediction": majority_class,
                "samples": total_samples,
                "class_distribution": class_probabilities,
                "is_leaf": True
            }
        
        # Find best split using information gain
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        # If no good split found, make leaf
        if best_gain <= 0:
            class_counts = {}
            for label in y:
                class_counts[label] = class_counts.get(label, 0) + 1
            majority_class = self._get_majority_class(class_counts)
            total_samples = sum(class_counts.values())
            class_probabilities = {cls: count/total_samples for cls, count in class_counts.items()}
            
            return {
                "prediction": majority_class,
                "samples": total_samples,
                "class_distribution": class_probabilities,
                "is_leaf": True
            }
        
        # Split data
        left_indices, right_indices = self._split_data(X, best_feature, best_threshold)
        
        # Check minimum samples in leaves
        if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
            class_counts = {}
            for label in y:
                class_counts[label] = class_counts.get(label, 0) + 1
            majority_class = self._get_majority_class(class_counts)
            total_samples = sum(class_counts.values())
            class_probabilities = {cls: count/total_samples for cls, count in class_counts.items()}
            
            return {
                "prediction": majority_class,
                "samples": total_samples,
                "class_distribution": class_probabilities,
                "is_leaf": True
            }
        
        # Create split data
        left_X = [X[i] for i in left_indices]
        left_y = [y[i] for i in left_indices]
        right_X = [X[i] for i in right_indices]
        right_y = [y[i] for i in right_indices]
        
        # Create node
        node = {
            "feature": best_feature,
            "threshold": best_threshold,
            "information_gain": best_gain,
            "samples": n_samples,
            "left": self._build_tree(left_X, left_y, depth + 1),
            "right": self._build_tree(right_X, right_y, depth + 1),
            "is_leaf": False
        }
        
        return node

    def _get_majority_class(self, class_counts: Dict[float, int]) -> float:
        """
        Get the majority class from class counts.
        
        Parameters:
        class_counts - Dictionary of class counts
        
        Returns:
        Majority class
        """
        if not class_counts:
            return 0.0
        
        # Find class with maximum count
        max_count = -1
        majority_class = 0.0
        for cls, count in class_counts.items():
            if count > max_count:
                max_count = count
                majority_class = cls
        
        return majority_class

    def _find_best_split(self, X: List[List[float]], y: List[float]) -> Tuple[int, float, float]:
        """
        Find the best split using information gain.
        
        Parameters:
        X - Feature matrix
        y - Target vector
        
        Returns:
        Best feature index, threshold, and information gain
        """
        n_samples = len(y)
        n_features = len(X[0]) if X else 0
        
        if n_samples <= 1 or n_features == 0:
            return 0, 0, 0
        
        # Calculate parent entropy
        parent_entropy = self._calculate_entropy(y)
        
        best_feature = 0
        best_threshold = 0
        best_gain = -1
        
        # Determine features to consider
        features_to_consider = list(range(n_features))
        if self.max_features is not None and self.max_features < n_features:
            features_to_consider = random.sample(features_to_consider, self.max_features)
        
        # Try all features
        for feature_idx in features_to_consider:
            # Get unique values for this feature
            feature_values = [row[feature_idx] for row in X]
            unique_values = sorted(list(set(feature_values)))
            
            # Try split points between consecutive unique values
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                # Split data
                left_indices, right_indices = self._split_data(X, feature_idx, threshold)
                
                # Skip if split doesn't create valid partitions
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                # Calculate information gain
                left_y = [y[i] for i in left_indices]
                right_y = [y[i] for i in right_indices]
                
                left_entropy = self._calculate_entropy(left_y)
                right_entropy = self._calculate_entropy(right_y)
                
                # Weighted average entropy
                n_left = len(left_y)
                n_right = len(right_y)
                weighted_entropy = (n_left / n_samples) * left_entropy + (n_right / n_samples) * right_entropy
                
                # Information gain
                gain = parent_entropy - weighted_entropy
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain

    def _calculate_entropy(self, y: List[float]) -> float:
        """
        Calculate entropy of a target vector.
        
        Parameters:
        y - Target vector
        
        Returns:
        Entropy
        """
        if not y:
            return 0
        
        # Count class frequencies
        class_counts = {}
        for label in y:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Calculate entropy
        n_samples = len(y)
        entropy = 0
        for count in class_counts.values():
            if count > 0:
                p = count / n_samples
                entropy -= p * log(p) if p > 0 else 0
        
        return entropy

    def _split_data(self, X: List[List[float]], feature_idx: int, threshold: float) -> Tuple[List[int], List[int]]:
        """
        Split data based on feature and threshold.
        
        Parameters:
        X - Feature matrix
        feature_idx - Feature index
        threshold - Split threshold
        
        Returns:
        Left and right indices
        """
        left_indices = []
        right_indices = []
        
        for i, row in enumerate(X):
            if row[feature_idx] <= threshold:
                left_indices.append(i)
            else:
                right_indices.append(i)
        
        return left_indices, right_indices

    def _calculate_feature_importances(self):
        """
        Calculate feature importances based on information gain.
        """
        if not self.tree:
            self.feature_importances = [0.0] * self.n_features
            return
        
        importances = [0.0] * self.n_features
        self._accumulate_importances(self.tree, importances)
        
        # Normalize importances
        total_importance = sum(importances)
        if total_importance > 0:
            self.feature_importances = [imp / total_importance for imp in importances]
        else:
            self.feature_importances = [1.0 / self.n_features] * self.n_features

    def _accumulate_importances(self, node: Dict, importances: List[float]):
        """
        Accumulate feature importances recursively.
        
        Parameters:
        node - Tree node
        importances - Accumulated importances
        """
        if node.get("is_leaf", False):
            return
        
        # Add importance for this split
        feature = node.get("feature", 0)
        gain = node.get("information_gain", 0)
        samples = node.get("samples", 0)
        
        # Weight importance by number of samples and information gain
        if 0 <= feature < len(importances):
            importances[feature] += gain * samples
        
        # Recurse
        left_child = node.get("left")
        right_child = node.get("right")
        
        if left_child:
            self._accumulate_importances(left_child, importances)
        if right_child:
            self._accumulate_importances(right_child, importances)

    def predict(self, X: List[List[float]]) -> List[float]:
        """
        Make predictions with enhanced error handling.
        
        Parameters:
        X - Feature matrix
        
        Returns:
        Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if not X:
            return []
        
        predictions = []
        for row in X:
            if self.tree is not None:
                pred = self._predict_single(row, self.tree)
                predictions.append(pred)
            else:
                predictions.append(0.0)
        
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
        if node.get("is_leaf", False):
            return node["prediction"]
        
        feature_idx = node.get("feature", 0)
        threshold = node.get("threshold", 0)
        
        if feature_idx < len(row) and row[feature_idx] <= threshold:
            left_child = node.get("left")
            if left_child:
                return self._predict_single(row, left_child)
            else:
                return node.get("prediction", 0)
        else:
            right_child = node.get("right")
            if right_child:
                return self._predict_single(row, right_child)
            else:
                return node.get("prediction", 0)

    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        """
        Predict class probabilities with enhanced implementation.
        
        Parameters:
        X - Feature matrix
        
        Returns:
        Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if not X:
            return []
        
        probabilities = []
        for row in X:
            if self.tree is not None:
                proba = self._predict_proba_single(row, self.tree)
                probabilities.append(proba)
            else:
                probabilities.append([0.5, 0.5])
        
        return probabilities

    def _predict_proba_single(self, row: List[float], node: Dict) -> List[float]:
        """
        Predict probabilities for a single sample.
        
        Parameters:
        row - Feature vector
        node - Current tree node
        
        Returns:
        Class probabilities
        """
        if node.get("is_leaf", False):
            # Return class distribution from leaf
            class_dist = node.get("class_distribution", {})
            # Convert to ordered list of probabilities
            # Assuming binary classification for simplicity
            proba = [class_dist.get(0, 0.0), class_dist.get(1, 0.0)]
            return proba
        
        feature_idx = node.get("feature", 0)
        threshold = node.get("threshold", 0)
        
        if feature_idx < len(row) and row[feature_idx] <= threshold:
            left_child = node.get("left")
            if left_child:
                return self._predict_proba_single(row, left_child)
            else:
                # Fallback to node prediction
                class_dist = node.get("class_distribution", {0: 0.5, 1: 0.5})
                return [class_dist.get(0, 0.0), class_dist.get(1, 0.0)]
        else:
            right_child = node.get("right")
            if right_child:
                return self._predict_proba_single(row, right_child)
            else:
                # Fallback to node prediction
                class_dist = node.get("class_distribution", {0: 0.5, 1: 0.5})
                return [class_dist.get(0, 0.0), class_dist.get(1, 0.0)]


class FeatureEngineer:
    """
    Engineer features for machine learning models with enhanced functionality.
    """

    def __init__(self):
        """
        Initialize the feature engineer.
        """
        pass

    def create_features(self, data: List[Dict]) -> List[Dict]:
        """
        Create enhanced features from raw market data.
        
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
            
            # Moving averages with enhanced calculation
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
            
            # Enhanced RSI calculation
            if i >= 14:
                gains = 0
                losses = 0
                valid_periods = 0
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
                            valid_periods += 1
                        except (ValueError, TypeError):
                            pass
                
                if valid_periods > 0 and gains + losses > 0:
                    avg_gain = gains / valid_periods
                    avg_loss = losses / valid_periods
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        new_row['rsi'] = 100 - (100 / (1 + rs))
                    else:
                        new_row['rsi'] = 100
                else:
                    new_row['rsi'] = 50
            else:
                new_row['rsi'] = 50
            
            # Volume features with enhanced calculation
            try:
                volume = float(row.get('volume', 0))
                if i >= 19:
                    vol_sum = 0
                    valid_volumes = 0
                    for j in range(i-19, i+1):
                        try:
                            vol_sum += float(data[j].get('volume', 0))
                            valid_volumes += 1
                        except (ValueError, TypeError):
                            pass
                    if valid_volumes > 0:
                        vol_ma = vol_sum / valid_volumes
                        if vol_ma > 0:
                            new_row['volume_ratio'] = volume / vol_ma
                        else:
                            new_row['volume_ratio'] = 1
                    else:
                        new_row['volume_ratio'] = 1
                else:
                    new_row['volume_ratio'] = 1
            except (ValueError, TypeError):
                new_row['volume_ratio'] = 1
            
            # Price volatility feature
            if i >= 19:
                try:
                    prices = [float(data[j]['close']) for j in range(i-19, i+1)]
                    price_volatility = std(prices)
                    new_row['volatility'] = price_volatility
                except (ValueError, TypeError):
                    new_row['volatility'] = 0
            else:
                new_row['volatility'] = 0
            
            features_data.append(new_row)
        
        return features_data

    def prepare_data_for_training(self, data: List[Dict], 
                                target_column: str = 'returns',
                                lookback_window: int = 10) -> Tuple[List[List[float]], List[float]]:
        """
        Prepare data for training machine learning models with enhanced features.
        
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
        Create classification target from returns with enhanced thresholding.
        
        Parameters:
        returns - List of returns
        threshold - Threshold for classification (default: 0 for positive/negative)
        
        Returns:
        Classification target (0 for negative, 1 for positive)
        """
        return [1 if ret > threshold else 0 for ret in returns]


class ModelEvaluator:
    """
    Evaluate machine learning models with enhanced metrics.
    """

    def __init__(self):
        """
        Initialize the model evaluator.
        """
        pass

    def evaluate_model(self, model: MLModel, X_test: List[List[float]], 
                      y_test: List[float]) -> Dict:
        """
        Evaluate a model with enhanced metrics.
        
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
        mse_values = [(y_test[i] - y_pred[i]) ** 2 for i in range(len(y_test))]
        mse = mean(mse_values)
        
        # Root mean squared error
        rmse = sqrt(mse)
        
        # Mean absolute error
        mae_values = [abs(y_test[i] - y_pred[i]) for i in range(len(y_test))]
        mae = mean(mae_values)
        
        # Mean absolute percentage error
        mape_values = []
        for i in range(len(y_test)):
            if y_test[i] != 0:
                mape_values.append(abs((y_test[i] - y_pred[i]) / y_test[i]))
        mape = mean(mape_values) * 100 if mape_values else 0
        
        # R-squared with enhanced calculation
        y_mean = mean(y_test)
        if std(y_test) > 0:
            ss_res = sum((y_test[i] - y_pred[i]) ** 2 for i in range(len(y_test)))
            ss_tot = sum((y_test[i] - y_mean) ** 2 for i in range(len(y_test)))
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            r2 = 0
        
        # Enhanced metrics for financial applications
        # Direction accuracy (for financial predictions)
        direction_correct = 0
        for i in range(len(y_test)):
            if (y_test[i] > 0 and y_pred[i] > 0) or (y_test[i] < 0 and y_pred[i] < 0):
                direction_correct += 1
        direction_accuracy = direction_correct / len(y_test) if len(y_test) > 0 else 0
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'direction_accuracy': direction_accuracy
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
        Evaluate trading performance based on predictions with enhanced metrics.
        
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
        trades = 0
        
        for i in range(len(predictions)):
            # Determine new position based on prediction
            # Enhanced: positive prediction = long, negative = short, near zero = flat
            if predictions[i] > 0.1:
                new_position = 1  # Long
            elif predictions[i] < -0.1:
                new_position = -1  # Short
            else:
                new_position = 0  # Flat
            
            # Calculate transaction cost if position changes
            if new_position != current_position:
                cost = transaction_cost * abs(new_position - current_position)
                trades += 1
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
        
        # Win rate and profit factor
        winning_trades = [ret for ret in strategy_returns if ret > 0]
        losing_trades = [ret for ret in strategy_returns if ret < 0]
        
        win_rate = len(winning_trades) / len(strategy_returns) if strategy_returns else 0
        profit_factor = (sum(winning_trades) / abs(sum(losing_trades))) if losing_trades and sum(losing_trades) != 0 else float('inf')
        
        avg_win = mean(winning_trades) if winning_trades else 0
        avg_loss = mean(losing_trades) if losing_trades else 0
        
        # Sharpe ratio on returns (alternative calculation)
        sharpe_ratio_alt = (mean(strategy_returns) / std(strategy_returns) * sqrt(252)) if len(strategy_returns) > 1 and std(strategy_returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sharpe_ratio_alt': sharpe_ratio_alt,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'trades': trades,
            'calmar_ratio': abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0
        }


class MLStrategy:
    """
    Machine learning-based trading strategy with enhanced functionality.
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
                'confidence': abs(pred),  # Use absolute value as confidence measure
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
    Create a standard set of enhanced ML models for comparison.
    
    Returns:
    List of ML models
    """
    models = [
        EnhancedLinearRegression("EnhancedLinearRegression", regularization=0.01),
        EnhancedDecisionTree("EnhancedDecisionTree", max_depth=5, min_samples_split=10)
    ]
    
    return models


def train_and_evaluate_models(models: List[MLModel], 
                             X_train: List[List[float]], y_train: List[float],
                             X_test: List[List[float]], y_test: List[float]) -> Dict:
    """
    Train and evaluate a list of enhanced models.
    
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

