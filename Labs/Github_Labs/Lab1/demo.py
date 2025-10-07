"""
Interactive Demo Script for Fraud Detection Model
Run this script to see the model in action!
"""

import numpy as np
from src.model import FraudDetectionModel, generate_synthetic_data, train_and_evaluate_model
import warnings
warnings.filterwarnings('ignore')

def print_separator():
    print("="*60)

def main():
    print_separator()
    print(" CREDIT CARD FRAUD DETECTION MODEL - DEMO ".center(60))
    print_separator()
    
    # 1. Quick Demo with Pre-built Function
    print("\n📊 QUICK DEMO: Training and Evaluating Model")
    print("-"*40)
    
    model, metrics = train_and_evaluate_model()
    
    print("\n✅ Model Training Complete!")
    print(f"\nTraining Metrics:")
    for metric, value in metrics['train'].items():
        print(f"  • {metric.capitalize()}: {value:.4f}")
    
    print(f"\nTest Metrics:")
    for metric, value in metrics['test'].items():
        print(f"  • {metric.capitalize()}: {value:.4f}")
    
    print_separator()
    
    # 2. Interactive Prediction Demo
    print("\n🔍 INTERACTIVE PREDICTION DEMO")
    print("-"*40)
    print("\nLet's simulate some transactions and detect fraud...")
    
    # Generate some test samples
    n_samples = 10
    X_demo, y_demo = generate_synthetic_data(n_samples=n_samples, fraud_rate=0.3)
    
    # Make predictions
    predictions = model.predict(X_demo)
    probabilities = model.predict_proba(X_demo)
    
    print(f"\n{'Transaction':<15} {'Actual':<10} {'Predicted':<12} {'Fraud Prob':<12} {'Status':<10}")
    print("-"*70)
    
    for i in range(n_samples):
        actual = "Fraud" if y_demo[i] == 1 else "Legitimate"
        predicted = "Fraud" if predictions[i] == 1 else "Legitimate"
        status = "✅ Correct" if predictions[i] == y_demo[i] else "❌ Wrong"
        
        # Color coding for terminal (optional)
        if predictions[i] == 1:
            predicted = f"🚨 {predicted}"
        
        print(f"Transaction {i+1:<3} {actual:<10} {predicted:<12} {probabilities[i]:<12.2%} {status:<10}")
    
    # Calculate demo accuracy
    demo_accuracy = np.mean(predictions == y_demo)
    print(f"\nDemo Accuracy: {demo_accuracy:.2%}")
    
    print_separator()
    
    # 3. Feature Importance Demo
    print("\n📈 FEATURE IMPORTANCE ANALYSIS")
    print("-"*40)
    
    importance = model.get_feature_importance()
    
    print("\nTop 5 Most Important Features:")
    # Get indices of top 5 features
    top_features = np.argsort(importance)[-5:][::-1]
    
    for rank, idx in enumerate(top_features, 1):
        print(f"  {rank}. Feature {idx+1}: {importance[idx]:.4f}")
    
    print_separator()
    
    # 4. Model Persistence Demo
    print("\n💾 MODEL PERSISTENCE DEMO")
    print("-"*40)
    
    print("\nSaving model to disk...")
    model.save_model()
    print("✅ Model saved successfully!")
    
    print("\nLoading model from disk...")
    new_model = FraudDetectionModel()
    success = new_model.load_model()
    
    if success:
        print("✅ Model loaded successfully!")
        
        # Test loaded model
        test_data = np.random.randn(3, 10)
        loaded_predictions = new_model.predict(test_data)
        print(f"\nLoaded model predictions on 3 random samples: {loaded_predictions}")
    else:
        print("ℹ️  Model files not found (this is normal for first run)")
    
    print_separator()
    
    # 5. Performance Summary
    print("\n📋 PERFORMANCE SUMMARY")
    print("-"*40)
    
    print("\n🎯 Key Insights:")
    print("  • The model achieves good accuracy on synthetic data")
    print("  • XGBoost handles class imbalance well")
    print("  • Feature importance shows which variables matter most")
    print("  • Model can be saved and loaded for production use")
    
    print("\n🚀 Next Steps:")
    print("  1. Try with real credit card fraud dataset")
    print("  2. Add MLflow for experiment tracking")
    print("  3. Build Streamlit dashboard for monitoring")
    print("  4. Deploy with FastAPI for real-time predictions")
    
    print_separator()
    print("\n✨ Demo Complete! Check the test/ folder to run automated tests.")
    print_separator()


if __name__ == "__main__":
    main()
