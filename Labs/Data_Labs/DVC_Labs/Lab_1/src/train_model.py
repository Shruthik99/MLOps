import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import json
from datetime import datetime

def load_data():
    """Load preprocessed data"""
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/y_test.csv').values.ravel()
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, n_estimators=100, max_depth=10):
    """Train Random Forest model"""
    
    print("Training Random Forest model...")
    print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("✓ Model trained successfully")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Bad Wine', 'Good Wine']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy, y_pred

def save_model(model, accuracy):
    """Save trained model and metrics"""
    
    # Save model
    model_path = 'models/wine_quality_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ Model saved to {model_path}")
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'model_type': 'RandomForestClassifier',
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✓ Metrics saved to models/metrics.json")

def main():
    """Main training pipeline"""
    
    # Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy, y_pred = evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, accuracy)
    
    print("\n✓ Training pipeline complete!")

if __name__ == "__main__":
    main()