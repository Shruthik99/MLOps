# model.py - Credit Card Fraud Detection Model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
import os

class FraudDetectionModel:
    """
    Credit Card Fraud Detection Model using XGBoost
    """
    
    def __init__(self, model_path='models/fraud_model.pkl', scaler_path='models/scaler.pkl'):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.is_trained = False
        
    def preprocess_data(self, X):
        """
        Preprocess the input features
        
        Args:
            X: Input features (DataFrame or numpy array)
        
        Returns:
            Preprocessed features
        """
        if not self.is_trained:
            # Fit scaler during training
            X_scaled = self.scaler.fit_transform(X)
        else:
            # Use fitted scaler during inference
            X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the XGBoost model for fraud detection
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        
        Returns:
            Dictionary containing training metrics
        """
        # Preprocess data
        X_train_scaled = self.preprocess_data(X_train)
        
        # XGBoost parameters optimized for fraud detection
        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1]),  # Handle imbalanced data
            'random_state': 42
        }
        
        # Initialize and train model
        self.model = xgb.XGBClassifier(**params)
        
        # Prepare evaluation set if validation data provided
        eval_set = [(X_train_scaled, y_train)]
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            eval_set.append((X_val_scaled, y_val))
        
        # Train model
        self.model.fit(
            X_train_scaled, 
            y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        self.is_trained = True
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train_scaled)
        metrics = self.calculate_metrics(y_train, train_pred)
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Input features
        
        Returns:
            Predicted labels (0: legitimate, 1: fraud)
        """
        if not self.is_trained and not self.load_model():
            raise ValueError("Model must be trained or loaded before making predictions")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X: Input features
        
        Returns:
            Probability of fraud for each sample
        """
        if not self.is_trained and not self.load_model():
            raise ValueError("Model must be trained or loaded before making predictions")
        
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        return probabilities[:, 1]  # Return fraud probability
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Dictionary containing metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        return metrics
    
    def save_model(self):
        """
        Save trained model and scaler to disk
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save model and scaler
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        return True
    
    def load_model(self):
        """
        Load trained model and scaler from disk
        
        Returns:
            Boolean indicating success
        """
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_trained = True
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_feature_importance(self):
        """
        Get feature importance scores
        
        Returns:
            Dictionary of feature importances
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance = self.model.feature_importances_
        return importance


# Helper functions for demo/testing
def generate_synthetic_data(n_samples=1000, fraud_rate=0.1):
    """
    Generate synthetic fraud detection data for testing
    
    Args:
        n_samples: Number of samples to generate
        fraud_rate: Proportion of fraudulent transactions
    
    Returns:
        X: Feature matrix
        y: Labels (0: legitimate, 1: fraud)
    """
    np.random.seed(42)
    
    n_features = 10
    n_frauds = int(n_samples * fraud_rate)
    n_legitimate = n_samples - n_frauds
    
    # Generate legitimate transactions (normal distribution)
    legitimate = np.random.randn(n_legitimate, n_features)
    
    # Generate fraudulent transactions (different distribution)
    fraud = np.random.randn(n_frauds, n_features) * 2 + 1
    
    # Combine data
    X = np.vstack([legitimate, fraud])
    y = np.array([0] * n_legitimate + [1] * n_frauds)
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def train_and_evaluate_model():
    """
    Demo function to train and evaluate the model
    
    Returns:
        Tuple of (model, metrics)
    """
    # Generate synthetic data
    X, y = generate_synthetic_data(n_samples=2000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train model
    model = FraudDetectionModel()
    train_metrics = model.train(X_train, y_train)
    
    # Evaluate on test set
    test_pred = model.predict(X_test)
    test_metrics = model.calculate_metrics(y_test, test_pred)
    
    return model, {'train': train_metrics, 'test': test_metrics}