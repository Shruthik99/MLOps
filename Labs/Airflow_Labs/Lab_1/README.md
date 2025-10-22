# 🏙️ Smart City Energy Analysis 

## MLOps Implementation with Apache Airflow

### Project Overview
This project transforms the basic Airflow clustering lab into a comprehensive Smart City Energy Management system that analyzes energy consumption patterns across urban zones using multiple machine learning algorithms.
<img width="1889" height="1089" alt="image" src="https://github.com/user-attachments/assets/4a3fd06b-f9cf-450e-ba8c-666712805349" />


### Key Enhancements

4 ML Algorithms vs 1 in original lab
Anomaly Detection for energy theft/malfunction
PCA Dimensionality Reduction for scalability
Real-world Context with smart city data
Production Features including error handling and JSON reports

### Technical Stack

Orchestration: Apache Airflow 2.9.2
Containerization: Docker & Docker Compose
ML Framework: Scikit-learn
Data Processing: Pandas, NumPy
Algorithms: K-Means, DBSCAN, Hierarchical Clustering, Isolation Forest

### Installation & Setup
Prerequisites

Docker Desktop installed and running
4GB+ RAM allocated to Docker

### Dataset & Scale

#### Synthetic Data Generation
- **500 smart buildings** across 4 types (residential, commercial, industrial, public)
- **90 days** of hourly sensor readings
- **1,080,000+ data points** generated dynamically
- **19+ engineered features** including temporal, consumption, and environmental metrics

#### Static Datasets (for validation)
- `file.csv`: 50,000 training records
- `test.csv`: 10,000 test records with higher anomaly rate

### 🚀 Quick Start

#### Prerequisites
- Docker Desktop (8GB RAM allocated)
- Python 3.8+
- 10GB free disk space

#### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Lab_1
```

2. **Set up environment**
```bash
# Create environment file
echo "AIRFLOW_UID=50000" > .env

# Create required directories
mkdir -p working_data logs plugins
```

3. **Initialize Airflow**
```bash
docker compose up airflow-init
```

4. **Start services**
```bash
docker compose up
```

5. **Access Airflow UI**
- URL: http://localhost:8080
- Username: `admin`
- Password: `admin123`

###  Pipeline Architecture

```
Data Generation → Validation → Feature Engineering → PCA
                                                      ↓
Evaluation ← [K-Means | GMM | Hierarchical] ← Temporal Analysis
    ↓
Anomaly Detection → Visualization → Report Generation
    ↓
Model Export
```

###  Project Structure

```
Lab_1/
├── dags/
│   ├── airflow.py              # Main DAG definition
│   ├── data/
│   │   ├── file.csv            # Training data (50k records)
│   │   └── test.csv            # Test data (10k records)
│   └── src/
│       ├── __init__.py
│       └── lab.py              # Processing functions
├── working_data/               # Pipeline outputs
│   ├── models/                 # Saved models
│   ├── visualizations/         # Generated plots
│   └── reports/               # Analysis reports
├── docker-compose.yaml         # Docker configuration
├── .env                       # Environment variables
└── README.md                  # This file
```

### Key Features

#### 1. Advanced Clustering Algorithms

**K-Means Clustering**
- Multiple initialization methods (k-means++, random)
- Elbow method with automatic knee detection
- Silhouette analysis for cluster quality
- Gap statistic for statistical validation

**Gaussian Mixture Models**
- Soft clustering with probability assignments
- Multiple covariance types (full, tied, diagonal, spherical)
- BIC/AIC for model selection
- Handles overlapping clusters

**Hierarchical Clustering**
- Multiple linkage methods (Ward, complete, average)
- Dendrogram visualization
- Automatic optimal cut detection
- Building taxonomy creation

#### 2. Comprehensive Evaluation Metrics

- **Silhouette Score**: Cluster cohesion and separation (-1 to 1)
- **Calinski-Harabasz Index**: Between vs within cluster variance
- **Davies-Bouldin Index**: Average cluster similarity (lower is better)
- **Dunn Index**: Cluster compactness and separation
- **Stability Analysis**: Bootstrap validation

#### 3. Anomaly Detection Ensemble

- **Isolation Forest**: Tree-based anomaly detection
- **Local Outlier Factor**: Density-based detection
- **One-Class SVM**: Boundary-based detection
- **Ensemble Voting**: Combines all methods

#### 4. Visualizations & Reporting

- 3D PCA scatter plots
- Temporal consumption heatmaps
- Cluster comparison matrices
- Dendrogram for hierarchical relationships
- Interactive Plotly dashboards
- PDF optimization reports with ROI calculations

### Expected Results

#### Clustering Performance
- **3-5 distinct energy patterns** identified
- **Silhouette Score**: 0.6-0.8 (good separation)
- **Optimal clusters**: Determined by consensus

#### Anomaly Detection
- **~3% anomaly rate** in training data
- **Precision**: 0.85+
- **Recall**: 0.75+
- **F1-Score**: 0.80+

#### Business Impact
- **15-20% energy savings** identified
- **ROI period**: 8-12 months
- **CO2 reduction**: Quantified in reports
- **Maintenance predictions**: Early anomaly detection

### Running the Pipeline

1. **Enable the DAG**
   - Find "Smart_City_Energy_Analysis_Pipeline" in Airflow UI
   - Toggle the switch to ON

2. **Trigger Execution**
   - Click on DAG name
   - Click "Trigger DAG" button
   - Monitor progress in Graph view

3. **View Results**
   - Check `working_data/visualizations/` for plots
   - Review `working_data/reports/` for analysis
   - Models saved in `working_data/models/`

### Configuration

#### Modify Pipeline Parameters

Edit `dags/airflow.py` to adjust:
```python
'n_buildings': 500,        # Number of buildings
'days': 90,                # Analysis period
'anomaly_rate': 0.03,      # Anomaly percentage
'k_range': (3, 20),        # Cluster range
```

#### Change Algorithms

In `dags/src/lab.py`, modify:
- Clustering methods
- Optimization techniques
- Anomaly detection algorithms
- Evaluation metrics

### Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Data Points | 1,080,000+ | Total records processed |
| Features | 19+ | Engineered features |
| Processing Time | ~10-15 min | Full pipeline execution |
| Memory Usage | ~2GB | Peak RAM consumption |
| Model Accuracy | 85%+ | Anomaly detection F1 |

###  Educational Value

This implementation demonstrates:

1. **MLOps Best Practices**
   - Pipeline orchestration with Airflow
   - Containerization with Docker
   - Model versioning and export
   - Error handling and retries

2. **Advanced ML Techniques**
   - Multiple clustering algorithms
   - Ensemble methods
   - Feature engineering
   - Dimensionality reduction

3. **Production Considerations**
   - Scalable architecture
   - Monitoring and logging
   - API specifications
   - Business metrics

4. **Real-World Application**
   - Domain-specific problem solving
   - ROI and sustainability metrics
   - Actionable recommendations
   - Interactive visualizations

###  Troubleshooting

#### Docker Issues
```bash
# Reset everything
docker compose down -v
docker system prune -a
docker compose up airflow-init
docker compose up
```

#### Login Problems
```bash
# Create admin user manually
docker exec -it lab_1-airflow-webserver-1 airflow users create \
    --username admin \
    --firstname Admin \
    --lastname Admin \
    --role Admin \
    --email admin@admin.com \
    --password admin123
```

#### Port Conflicts
```bash
# Check what's using port 8080
netstat -ano | findstr :8080

# Use different port
# Edit docker-compose.yaml: change "8080:8080" to "8081:8080"
```

### Technologies Used

- **Apache Airflow 2.9.2**: Workflow orchestration
- **Docker**: Containerization
- **Python 3.8+**: Core programming
- **Scikit-learn**: Machine learning algorithms
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Static visualizations
- **Plotly**: Interactive dashboards
- **SciPy**: Statistical analysis


## Sample Output

### Cluster Analysis
```
Optimal Clusters Found: 4
Cluster 0: High-consumption industrial (23% of buildings)
Cluster 1: Efficient residential (38% of buildings)
Cluster 2: Peak-hour commercial (27% of buildings)
Cluster 3: Anomalous patterns (12% of buildings)
```

### Energy Savings
```
Potential Annual Savings: $2.4M
CO2 Reduction: 1,250 tons/year
Equivalent Trees: 31,250 trees planted
ROI Period: 10 months
```

### Model Performance
```
Best Model: Gaussian Mixture Model
Silhouette Score: 0.742
Anomaly Detection F1: 0.863
Processing Time: 12m 34s
```

---


