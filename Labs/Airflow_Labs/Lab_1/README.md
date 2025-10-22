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
#### Prerequisites

- Docker Desktop installed and running
- 4GB+ RAM allocated to Docker

### Data Pipeline
DAG Tasks (6 total)

load_data_task: Loads/generates smart city energy data
data_quality_check: Validates data existence
data_preprocessing_task: Feature engineering & standardization
build_save_model_task: Trains 3 clustering algorithms
detect_anomalies (parallel): Isolation Forest anomaly detection
evaluate_and_generate_report: Creates comparison reports
generate_summary: Final summary output

### Data Generation

Training: 8000 records (smart_city_energy.csv)
Test: 2000 records (test_energy.csv)
Features: 12+ including energy consumption, temperature, occupancy
Zones: 6 city zones (Residential, Commercial, Industrial, etc.)
Anomalies: 5% embedded for testing



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

### Results & Business Value
### Expected Outcomes

4-5 optimal clusters identifying consumption patterns
~500 anomalies detected (5% of 10k records)
K-Means typically performs best for this data
Actionable insights for city energy management

### Applications

Cost Optimization: Identify inefficient buildings
Security: Detect energy theft through anomalies
Planning: Zone-based infrastructure decisions
Sustainability: Optimize renewable integration



## Project Structure

```
C:\Users\shrut\Desktop\MLOps-1\Labs\Airflow_Labs\Lab_1\
│
├── docker-compose.yaml        # Docker configuration
├── .env                      # Environment variables 
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── dags/
│   ├── airflow.py           # Main DAG with 6 tasks
│   ├── data/
│   │   ├── generate_data.py        # Data generator script
│   │   ├── smart_city_energy.csv   # Generated training data
│   │   └── test_energy.csv         # Generated test data
│   ├── model/               # Saved models (generated at runtime)
│   ├── results/             # JSON reports (generated at runtime)
│   └── src/
│       ├── __init__.py      # Package init
│       ├── lab.py           # ML algorithms implementation
│       └── airflow.py       # Copy of main DAG (backup)
│
├── logs/                    # Airflow execution logs
├── plugins/                 # Custom plugins
└── working_data/            # Output files
```

## Data Pipeline

### DAG Tasks (6 total)
1. **load_data_task**: Loads/generates smart city energy data
2. **data_quality_check**: Validates data existence
3. **data_preprocessing_task**: Feature engineering & standardization
4. **build_save_model_task**: Trains 3 clustering algorithms
5. **detect_anomalies** (parallel): Isolation Forest anomaly detection
6. **evaluate_and_generate_report**: Creates comparison reports
7. **generate_summary**: Final summary output

### Data Generation
- **Training**: 8000 records (`file.csv`)
- **Test**: 2000 records (`test.csv`)
- **Features**: 12+ including energy consumption, temperature, occupancy
- **Zones**: 6 city zones (Residential, Commercial, Industrial, etc.)
- **Anomalies**: 5% embedded for testing

## Machine Learning Components

### Clustering Algorithms

| Algorithm | Purpose | Key Metrics |
|-----------|---------|-------------|
| K-Means | Pattern identification | Silhouette Score, SSE |
| DBSCAN | Density-based clusters | Noise point detection |
| Hierarchical | Relationship analysis | Dendrogram potential |


## Comparison with Original Lab

| Aspect | Original Lab | This Project |
|--------|-------------|--------------|
| Algorithms | 1 (K-Means) | 4 (K-Means, DBSCAN, Hierarchical, Isolation Forest) |
| Evaluation | Elbow method | 4 comprehensive metrics |
| Context | Academic exercise | Real-world smart city application |
| Data | Simple CSV | Auto-generated energy patterns |
| Output | Single model | Multiple models + JSON reports |
| Error Handling | Basic | Comprehensive try-except blocks |

## Troubleshooting

### Common Issues

**DAG not appearing**
```bash
python dags\airflow.py  # Check syntax
docker compose restart  # Restart services
```

**Model folder missing**
```bash
mkdir dags\model
```

**Import errors**
- Verify `__init__.py` exists in `src/`
- Check docker-compose.yaml includes all packages

**Memory issues**
- Increase Docker RAM allocation
- Reduce data size in generator

## Commands Reference

```bash
# Check Docker status
docker ps

# View logs
docker compose logs airflow-webserver

# Enter container
docker exec -it lab_1-airflow-webserver-1 bash

# Clean restart
docker compose down
docker compose up airflow-init
docker compose up

# Stop services
docker compose down
```
