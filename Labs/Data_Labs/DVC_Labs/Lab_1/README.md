# Lab 1: Wine Quality Classification with DVC and Google Cloud Storage

**Lab**: Data Version Control (DVC) Implementation  
**Dataset**: Wine Quality Dataset  
**Cloud Provider**: Google Cloud Platform (GCS)

---

## 📋 Overview

This lab demonstrates the implementation of Data Version Control (DVC) for machine learning projects. We use DVC to:
- Version control datasets
- Track ML models
- Store data remotely on Google Cloud Storage
- Maintain reproducibility across experiments

## 🎯 Objectives

1. Set up DVC with Google Cloud Storage as remote storage
2. Track and version a wine quality dataset
3. Build a simple classification model
4. Demonstrate data versioning capabilities
5. Show how to revert to previous data versions

## 📊 Dataset

**Wine Quality Dataset** from UCI Machine Learning Repository
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Type**: Red Wine Quality
- **Samples**: 1,599 wines
- **Features**: 11 physicochemical features
- **Target**: Wine quality classification (Good/Bad wine based on quality score)

### Features:
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

## 🏗️ Project Structure
```
Lab_1/
│
├── .dvc/                              # DVC configuration directory
│   ├── cache/                        # DVC cache
│   ├── tmp/                          # Temporary DVC files
│   ├── .gitignore              
│   ├── .gitkeep                      
│   └── config                        # DVC remote configuration 
│
├── data/                              # Data directory 
│   ├── .gitignore                    
│   ├── .gitkeep                      
│   ├── wine_quality_raw.csv          # Raw dataset (ignored by Git)
│   ├── wine_quality_raw.csv.dvc      # DVC metadata for raw data
│   ├── X_train.csv                   # Processed training features (ignored by Git)
│   ├── X_train.csv.dvc               # DVC metadata for training features
│   ├── X_test.csv                    # Processed test features (ignored by Git)
│   ├── X_test.csv.dvc                # DVC metadata for test features
│   ├── y_train.csv                   # Training labels (ignored by Git)
│   ├── y_train.csv.dvc               # DVC metadata for training labels
│   ├── y_test.csv                    # Test labels (ignored by Git)
│   ├── y_test.csv.dvc                # DVC metadata for test labels
│   ├── data.txt                      # Additional data file (ignored by Git)
│   └── data.txt.dvc                  # DVC metadata for data.txt
│
├── models/                            # Model directory (tracked by DVC)
│   ├── .gitignore                    # Ignores actual model files from Git
│   ├── .gitkeep                      # Keeps directory structure in Git
│   ├── wine_quality_model.pkl        # Trained model (ignored by Git)
│   ├── wine_quality_model.pkl.dvc    # DVC metadata for model
│   ├── scaler.pkl                    # Feature scaler (ignored by Git)
│   ├── scaler.pkl.dvc                # DVC metadata for scaler
│   └── metrics.json                  # Model performance metrics
│
├── src/                               # Source code
│   ├── __init__.py                   # Makes src a Python package
│   ├── data_preprocessing.py         # Data preprocessing pipeline
│   ├── train_model.py                # Model training script
│   └── update_dataset.py             # Dataset modification script
│
├── .dvcignore                         # DVC ignore patterns
├── .gitignore                         # Git ignore file
├── README.md                          # Project documentation
└── requirements.txt                   # Python dependencies

```

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- Git
- Google Cloud Platform account
- VSCode (recommended)

### Installation

1. **Configure GCP credentials**:
   - Place your `gcp-credentials.json` in the project root
   - Update `.dvc/config` with your bucket name

2. **Initialize DVC** :
```bash
   dvc init
   dvc remote add -d myremote gs://your-bucket-name
   dvc remote modify myremote credentialpath gcp-credentials.json
```

## 🚀 Usage

### 1. Download and Preprocess Data
```bash
python src/data_preprocessing.py
```

### 2. Train Model
```bash
python src/train_model.py
```

### 3. Track Data with DVC
```bash
dvc add data/wine_quality_raw.csv
dvc add models/wine_quality_model.pkl
```

### 4. Commit to Git
```bash
git add data/*.dvc models/*.dvc .dvc/
git commit -m "Add dataset and model"
```

### 5. Push to DVC Remote
```bash
dvc push
```

### 6. Pull Data (on new machine or after checkout)
```bash
dvc pull
```

## 🔄 Versioning Workflow

### Making Changes
```bash
# 1. Modify data or retrain model
python src/update_dataset.py
python src/train_model.py

# 2. Track changes
dvc add data/wine_quality_raw.csv
dvc add models/wine_quality_model.pkl

# 3. Commit
git add data/*.dvc models/*.dvc
git commit -m "Update: description of changes"

# 4. Push
dvc push
```

### Reverting to Previous Version
```bash
# 1. Find commit hash
git log --oneline

# 2. Checkout commit
git checkout <commit-hash>

# 3. Restore data
dvc checkout
```

## 📈 Model Performance

**Initial Model Results**:
- Algorithm: Random Forest Classifier
- Accuracy: ~XX% (will vary based on your run)
- Features: 11 physicochemical properties
- Task: Binary classification (Good/Bad wine)

Metrics are saved in `models/metrics.json` after each training run.

## 🔑 Key DVC Commands

| Command | Description |
|---------|-------------|
| `dvc init` | Initialize DVC in project |
| `dvc add <file>` | Start tracking a file |
| `dvc push` | Upload data to remote storage |
| `dvc pull` | Download data from remote storage |
| `dvc checkout` | Restore files to match current Git commit |
| `dvc status` | Check status of tracked files |
| `dvc remote list` | List configured remotes |

## 📝 Key Learnings

1. **Data Versioning**: DVC creates hash-based identifiers for datasets
2. **Remote Storage**: Large files stored in GCS, not Git
3. **Git Integration**: DVC metadata (.dvc files) tracked in Git
4. **Reproducibility**: Exact dataset versions tied to code versions
5. **Collaboration**: Team members can sync data via `dvc pull`

## 🔒 Security Notes

- ⚠️ **NEVER commit** `gcp-credentials.json` to Git
- ⚠️ Credentials file is in `.gitignore`
- ⚠️ Rotate service account keys regularly
- ⚠️ Use minimal IAM permissions (Storage Admin, not Owner)


### DVC push fails
```bash
# Check credentials
dvc remote modify myremote credentialpath gcp-credentials.json

# Verify bucket access
gsutil ls gs://your-bucket-name
```

### Git tracking data files
```bash
# Ensure data files are in .gitignore
git rm --cached data/*.csv
git add data/.gitignore
```

## 📚 References

- [DVC Documentation](https://dvc.org/doc)
- [Google Cloud Storage](https://cloud.google.com/storage/docs)
- [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)



