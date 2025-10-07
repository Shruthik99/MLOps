# test_unittest.py - Unittest tests for Fraud Detection Model
import sys
import os
import unittest
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import FraudDetectionModel, generate_synthetic_data


class TestFraudDetectionModel(unittest.TestCase):
    """Test suite for FraudDetectionModel using unittest"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.model = FraudDetectionModel()
        self.X, self.y = generate_synthetic_data(n_samples=100)
    
    def tearDown(self):
        """Clean up after each test method"""
        # Clean up any saved models if they exist
        if os.path.exists('models/fraud_model.pkl'):
            os.remove('models/fraud_model.pkl')
        if os.path.exists('models/scaler.pkl'):
            os.remove('models/scaler.pkl')
    
    def test_model_initialization(self):
        """Test that model initializes correctly"""
        self.assertIsNone(self.model.model)
        self.assertFalse(self.model.is_trained)
        self.assertEqual(self.model.model_path, 'models/fraud_model.pkl')
    
    def test_preprocess_data(self):
        """Test data preprocessing"""
        X_scaled = self.model.preprocess_data(self.X)
        
        # Check that scaling was applied
        self.assertEqual(X_scaled.shape, self.X.shape)
        
        # Scaled data should have approximately zero mean and unit variance
        mean = np.mean(X_scaled)
        std = np.std(X_scaled)
        self.assertAlmostEqual(mean, 0, places=1)
        self.assertAlmostEqual(std, 1, places=1)
    
    def test_train_model(self):
        """Test model training"""
        # Train model
        metrics = self.model.train(self.X, self.y)
        
        # Check that model was trained
        self.assertTrue(self.model.is_trained)
        self.assertIsNotNone(self.model.model)
        
        # Check metrics are returned
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        
        # Check metric ranges
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
        self.assertGreaterEqual(metrics['precision'], 0)
        self.assertLessEqual(metrics['precision'], 1)
        self.assertGreaterEqual(metrics['recall'], 0)
        self.assertLessEqual(metrics['recall'], 1)
        self.assertGreaterEqual(metrics['f1_score'], 0)
        self.assertLessEqual(metrics['f1_score'], 1)
    
    def test_predict(self):
        """Test model predictions"""
        # Train model first
        self.model.train(self.X, self.y)
        
        # Make predictions
        predictions = self.model.predict(self.X[:10])
        
        # Check predictions
        self.assertEqual(len(predictions), 10)
        for pred in predictions:
            self.assertIn(pred, [0, 1])
    
    def test_predict_proba(self):
        """Test probability predictions"""
        # Train model first
        self.model.train(self.X, self.y)
        
        # Get probabilities
        probabilities = self.model.predict_proba(self.X[:10])
        
        # Check probabilities
        self.assertEqual(len(probabilities), 10)
        for prob in probabilities:
            self.assertGreaterEqual(prob, 0)
            self.assertLessEqual(prob, 1)
    
    def test_calculate_metrics(self):
        """Test metric calculation"""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 0, 1])
        
        metrics = self.model.calculate_metrics(y_true, y_pred)
        
        # Check all metrics are present
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        
        # Check specific metric values
        self.assertEqual(metrics['accuracy'], 0.875)  # 7/8 correct
        self.assertEqual(metrics['precision'], 1.0)  # No false positives
        self.assertEqual(metrics['recall'], 0.75)  # 3/4 true positives detected
    
    def test_untrained_model_prediction_error(self):
        """Test that untrained model raises error on prediction"""
        X_test = np.random.randn(5, 10)
        
        with self.assertRaises(ValueError) as context:
            self.model.predict(X_test)
        
        self.assertIn("Model must be trained", str(context.exception))
    
    def test_save_and_load_model(self):
        """Test model saving and loading"""
        # Train model
        self.model.train(self.X, self.y)
        
        # Save model
        save_success = self.model.save_model()
        self.assertTrue(save_success)
        
        # Create new model instance and load
        new_model = FraudDetectionModel()
        load_success = new_model.load_model()
        self.assertTrue(load_success)
        self.assertTrue(new_model.is_trained)
        
        # Make predictions with loaded model
        predictions = new_model.predict(self.X[:5])
        self.assertEqual(len(predictions), 5)
    
    def test_feature_importance(self):
        """Test feature importance extraction"""
        # Train model
        self.model.train(self.X, self.y)
        
        # Get feature importance
        importance = self.model.get_feature_importance()
        
        # Check importance scores
        self.assertEqual(len(importance), self.X.shape[1])
        for score in importance:
            self.assertGreaterEqual(score, 0)
        
        # At least some features should be important
        self.assertGreater(np.sum(importance), 0)
    
    def test_different_fraud_rates(self):
        """Test model with different fraud rates"""
        fraud_rates = [0.05, 0.1, 0.2]
        
        for rate in fraud_rates:
            with self.subTest(fraud_rate=rate):
                X, y = generate_synthetic_data(n_samples=200, fraud_rate=rate)
                model = FraudDetectionModel()
                metrics = model.train(X, y)
                
                # Model should achieve reasonable performance
                self.assertGreater(metrics['accuracy'], 0.5)  # Better than random


if __name__ == '__main__':
    unittest.main()
