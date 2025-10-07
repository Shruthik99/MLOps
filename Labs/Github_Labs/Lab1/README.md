# Credit Card Fraud Detection Model - MLOps github Lab 1 

## Overview
This project aims to implement a **Credit Card Fraud Detection System** using XGBoost. This modification demonstrates practical MLOps practices in a financial risk management context, featuring automated testing, CI/CD pipelines, and model deployment readiness.

## 🎯 Key Modifications from Original Lab

| Original Lab | This Modification |
|-------------|-------------------|
| Simple calculator with arithmetic functions | XGBoost-based fraud detection model |
| Basic function testing | Comprehensive ML model testing |
| Simple CI/CD | Enhanced CI/CD with coverage reports and multi-version testing |
| No data handling | Data preprocessing, scaling, and synthetic data generation |
| No model persistence | Model serialization and loading capabilities |

## 📁 Project Structure

```
Lab1/
├── .github/
│   └── workflows/
│       ├── pytest_action.yml      # GitHub Actions for pytest
│       └── unittest_action.yml    # GitHub Actions for unittest
├── src/
│   ├── __init__.py
│   └── model.py                   # Fraud detection model
├── test/
│   ├── test_pytest.py             # Pytest tests
│   └── test_unittest.py           # Unittest tests
├── demo.py                        # Demo script
├── README.md
├── requirements.txt
└── .gitignore.txt
```

## 🚀 Features

### Model Capabilities
- **Binary Classification**: Detects fraudulent vs legitimate transactions
- **XGBoost Algorithm**: Gradient boosting for high accuracy
- **Imbalanced Data Handling**: Automatic class weight balancing
- **Feature Scaling**: StandardScaler for preprocessing
- **Model Persistence**: Save/load trained models
- **Feature Importance**: Extract important features for interpretability

### MLOps Features
- **Automated Testing**: Comprehensive test suites with pytest and unittest
- **CI/CD Pipeline**: GitHub Actions for automated testing on push/PR
- **Code Coverage**: Track test coverage with pytest-cov
- **Multi-Python Version Testing**: Test against Python 3.8, 3.9, 3.10
- **Artifact Management**: Automatic test report generation and storage

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <your-repository-url>
cd Lab1_NEW
```

2. **Create virtual environment**
```bash
python -m venv fraud_detection_env

# Windows
fraud_detection_env\Scripts\activate

# Linux/Mac
source fraud_detection_env/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Model Training
```python
from src.model import FraudDetectionModel, generate_synthetic_data

# Generate synthetic data
X, y = generate_synthetic_data(n_samples=1000, fraud_rate=0.1)

# Initialize and train model
model = FraudDetectionModel()
metrics = model.train(X, y)

print(f"Training Metrics: {metrics}")

# Make predictions
predictions = model.predict(X[:10])
probabilities = model.predict_proba(X[:10])

# Save model
model.save_model()
```

### Loading and Using Saved Model
```python
from src.model import FraudDetectionModel
import numpy as np

# Load pre-trained model
model = FraudDetectionModel()
model.load_model()

# Make predictions on new data
new_data = np.random.randn(5, 10)  # 5 samples, 10 features
predictions = model.predict(new_data)
fraud_probabilities = model.predict_proba(new_data)
```

## 🧪 Testing

### Run Pytest Tests
```bash
# Run all pytest tests
pytest test/test_pytest.py -v

# Run with coverage report
pytest test/test_pytest.py --cov=src --cov-report=html

# Run specific test
pytest test/test_pytest.py::TestFraudDetectionModel::test_train_model
```

### Run Unittest Tests
```bash
# Run all unittest tests
python -m unittest test.test_unittest

# Run with verbose output
python -m unittest test.test_unittest -v
```

## 📊 Model Performance Metrics

The model evaluates performance using:
- **Accuracy**: Overall correct predictions
- **Precision**: Ratio of true frauds among predicted frauds
- **Recall**: Ratio of detected frauds among all actual frauds
- **F1-Score**: Harmonic mean of precision and recall

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

1. **Pytest Workflow** (`pytest_action.yml`)
   - Triggers on push to main or PR
   - Tests on multiple Python versions (3.8, 3.9, 3.10)
   - Generates coverage reports
   - Uploads test artifacts

2. **Unittest Workflow** (`unittest_action.yml`)
   - Runs unittest suite
   - Generates XML test reports
   - Provides success/failure notifications

