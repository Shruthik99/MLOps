import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

def download_and_prepare_data():
    """Download Wine Quality dataset and prepare it"""
    
    # Download red wine quality dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    print("Downloading Wine Quality dataset...")
    df = pd.read_csv(url, sep=';')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Save raw data
    df.to_csv('data/wine_quality_raw.csv', index=False)
    print("✓ Raw data saved to data/wine_quality_raw.csv")
    
    return df

def preprocess_data(df):
    """Preprocess the wine quality data"""
    
    # Create binary classification: good wine (quality >= 6) vs bad wine
    df['quality_label'] = (df['quality'] >= 6).astype(int)
    
    # Features and target
    X = df.drop(['quality', 'quality_label'], axis=1)
    y = df['quality_label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Save processed data
    X_train_scaled.to_csv('data/X_train.csv', index=False)
    X_test_scaled.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    
    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("✓ Preprocessed data saved")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Features: {X.columns.tolist()}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Download and prepare
    df = download_and_prepare_data()
    
    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    print("\n✓ Data preparation complete!")