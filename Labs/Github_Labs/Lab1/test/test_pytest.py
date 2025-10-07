# test_pytest.py - Pytest tests for Fraud Detection Model
import sys
import os
import pytest
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import FraudDetectionModel, generate_synthetic_data


class TestFraudDetectionModel:
    """Test suite for FraudDetectionModel using pytest"""
    
    @pytest.fixture
    def model(self):
        """Create a fresh model instance for each test"""
        return FraudDetectionModel()
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        X, y = generate_synthetic_data(n_samples=100)
        return X, y
    
    def test_model_initialization(self, model):
        """Test that model initializes correctly"""
        assert model.model is None
        assert model.is_trained is False
        assert model.model_path == 'models/fraud_model.pkl'
    
    def test_preprocess_data(self, model, sample_data):
        """Test data preprocessing"""
        X, _ = sample_data
        X_scaled = model.preprocess_data(X)
        
        # Check that scaling was applied
        assert X_scaled.shape == X.shape
        # Scaled data should have approximately zero mean and unit variance
        assert np.abs(np.mean(X_scaled)) < 0.1
        assert np.abs(np.std(X_scaled) - 1.0) < 0.1
    
    def test_train_model(self, model, sample_data):
        """Test model training"""
        X, y = sample_data
        
        # Train model
        metrics = model.train(X, y)
        
        # Check that model was trained
        assert model.is_trained is True
        assert model.model is not None
        
        # Check metrics are returned
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_predict(self, model, sample_data):
        """Test model predictions"""
        X, y = sample_data
        
        # Train model first
        model.train(X, y)
        
        # Make predictions
        predictions = model.predict(X[:10])
        
        # Check predictions
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_proba(self, model, sample_data):
        """Test probability predictions"""
        X, y = sample_data
        
        # Train model first
        model.train(X, y)
        
        # Get probabilities
        probabilities = model.predict_proba(X[:10])
        
        # Check probabilities
        assert len(probabilities) == 10
        assert all(0 <= p <= 1 for p in probabilities)
    
    def test_calculate_metrics(self, model):
        """Test metric calculation"""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 0, 1])
        
        metrics = model.calculate_metrics(y_true, y_pred)
        
        # Check all metrics are present
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # Check specific metric values
        assert metrics['accuracy'] == 0.875  # 7/8 correct
        assert metrics['precision'] == 1.0  # No false positives
        assert metrics['recall'] == 0.75  # 3/4 true positives detected
    
    def test_untrained_model_prediction_error(self, model):
        """Test that untrained model raises error on prediction"""
        X_test = np.random.randn(5, 10)
        
        with pytest.raises(ValueError, match="Model must be trained"):
            model.predict(X_test)
    
    def test_feature_importance(self, model, sample_data):
        """Test feature importance extraction"""
        X, y = sample_data
        
        # Train model
        model.train(X, y)
        
        # Get feature importance
        importance = model.get_feature_importance()
        
        # Check importance scores
        assert len(importance) == X.shape[1]
        assert all(score >= 0 for score in importance)
        assert np.sum(importance) > 0  # At least some features should be important


# Parameterized tests for different data sizes
@pytest.mark.parametrize("n_samples,fraud_rate", [
    (100, 0.1),
    (500, 0.2),
    (1000, 0.05)
])
def test_different_data_sizes(n_samples, fraud_rate):
    """Test model with different data sizes and fraud rates"""
    X, y = generate_synthetic_data(n_samples=n_samples, fraud_rate=fraud_rate)
    
    model = FraudDetectionModel()
    metrics = model.train(X, y)
    
    # Model should achieve reasonable performance
    assert metrics['accuracy'] > 0.5  # Better than random
    assert model.is_trained is True
