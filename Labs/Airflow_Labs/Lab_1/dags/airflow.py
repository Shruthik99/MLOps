# dags/airflow.py
"""
Smart City Energy Consumption Pattern Analysis Pipeline
Author: Your Name
Description: Advanced MLOps pipeline for analyzing energy consumption patterns
from IoT sensors in smart buildings, identifying usage patterns and anomalies
for optimization and predictive maintenance.
"""

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta
from airflow import configuration as conf
import json

# Import our enhanced lab functions
from src.lab import (
    generate_energy_data,
    load_validate_energy_data,
    feature_engineering,
    temporal_analysis,
    perform_pca_analysis,
    build_kmeans_model,
    build_gaussian_mixture,
    build_hierarchical_clustering,
    evaluate_clustering_models,
    detect_anomalies,
    generate_dashboards,
    create_optimization_report,
    export_model_artifacts
)

# Enable pickle support for XCom
conf.set('core', 'enable_xcom_pickling', 'True')

# Enhanced default arguments
default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['your.email@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
    'execution_timeout': timedelta(minutes=30),
}

# Create the enhanced DAG
dag = DAG(
    'Smart_City_Energy_Analysis_Pipeline',
    default_args=default_args,
    description='Multi-model clustering for energy consumption pattern analysis with anomaly detection',
    schedule_interval='0 2 * * *',  # Run at 2 AM daily
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'clustering', 'energy', 'iot', 'smart_city', 'sustainability'],
)

# Task 1: Generate or Load Energy Data
generate_data_task = PythonOperator(
    task_id='generate_energy_sensor_data',
    python_callable=generate_energy_data,
    op_kwargs={
        'n_buildings': 500,
        'days': 90,
        'sensor_frequency': 'hourly',
        'building_types': ['residential', 'commercial', 'industrial', 'public'],
        'include_weather': True,
        'include_events': True,
        'anomaly_rate': 0.03  # 3% anomalous patterns
    },
    dag=dag,
)

# Task 2: Load and Validate Data with Quality Checks
load_validate_task = PythonOperator(
    task_id='load_validate_energy_data',
    python_callable=load_validate_energy_data,
    op_args=[generate_data_task.output],
    op_kwargs={
        'check_missing': True,
        'check_outliers': True,
        'validate_ranges': True
    },
    dag=dag,
)

# Task 3: Advanced Feature Engineering
feature_engineering_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering,
    op_args=[load_validate_task.output],
    op_kwargs={
        'create_lag_features': True,
        'create_rolling_stats': True,
        'create_cyclic_features': True,
        'create_interaction_terms': True,
        'window_sizes': [24, 168, 720]  # Daily, weekly, monthly
    },
    dag=dag,
)

# Task 4: Temporal Pattern Analysis
temporal_task = PythonOperator(
    task_id='temporal_pattern_analysis',
    python_callable=temporal_analysis,
    op_args=[feature_engineering_task.output],
    op_kwargs={
        'decomposition_method': 'STL',  # Seasonal-Trend decomposition
        'analyze_seasonality': True,
        'detect_changepoints': True
    },
    dag=dag,
)

# Task 5: PCA for Dimensionality Reduction
pca_task = PythonOperator(
    task_id='perform_pca_analysis',
    python_callable=perform_pca_analysis,
    op_args=[temporal_task.output],
    op_kwargs={
        'variance_threshold': 0.95,
        'n_components': None,  # Auto-select
        'visualize': True,
        'use_kernel_pca': False
    },
    dag=dag,
)

# Task 6a: K-Means with Multiple Optimization Methods
kmeans_task = PythonOperator(
    task_id='build_kmeans_model',
    python_callable=build_kmeans_model,
    op_args=[pca_task.output],
    op_kwargs={
        'k_range': (3, 20),
        'optimization_methods': ['elbow', 'silhouette', 'gap_statistic', 'davies_bouldin'],
        'init_methods': ['k-means++', 'random'],
        'n_init': 20,
        'max_iter': 500
    },
    dag=dag,
)

# Task 6b: Gaussian Mixture Model (More flexible than K-Means)
gmm_task = PythonOperator(
    task_id='build_gaussian_mixture_model',
    python_callable=build_gaussian_mixture,
    op_args=[pca_task.output],
    op_kwargs={
        'n_components_range': (3, 20),
        'covariance_types': ['full', 'tied', 'diag', 'spherical'],
        'selection_criterion': 'bic',  # Bayesian Information Criterion
        'n_init': 10
    },
    dag=dag,
)

# Task 6c: Hierarchical Clustering (For Building Taxonomy)
hierarchical_task = PythonOperator(
    task_id='build_hierarchical_clustering',
    python_callable=build_hierarchical_clustering,
    op_args=[pca_task.output],
    op_kwargs={
        'linkage_methods': ['ward', 'complete', 'average'],
        'distance_threshold': None,
        'n_clusters': None,  # Auto-determine
        'create_dendrogram': True
    },
    dag=dag,
)

# Task 7: Comprehensive Model Evaluation
evaluation_task = PythonOperator(
    task_id='evaluate_clustering_models',
    python_callable=evaluate_clustering_models,
    op_args=[
        kmeans_task.output,
        gmm_task.output,
        hierarchical_task.output,
        pca_task.output
    ],
    op_kwargs={
        'metrics': [
            'silhouette', 'calinski_harabasz', 'davies_bouldin',
            'dunn_index', 'cluster_stability', 'runtime_performance'
        ],
        'cross_validate': True
    },
    dag=dag,
)

# Task 8: Anomaly Detection on Best Model
anomaly_task = PythonOperator(
    task_id='detect_consumption_anomalies',
    python_callable=detect_anomalies,
    op_args=[evaluation_task.output, pca_task.output],
    op_kwargs={
        'methods': ['isolation_forest', 'local_outlier_factor', 'one_class_svm'],
        'contamination': 'auto',
        'ensemble_method': 'voting'
    },
    dag=dag,
)

# Task 9: Generate Interactive Dashboards
dashboard_task = PythonOperator(
    task_id='generate_interactive_dashboards',
    python_callable=generate_dashboards,
    op_args=[evaluation_task.output, anomaly_task.output],
    op_kwargs={
        'dashboard_types': [
            'energy_patterns_3d',
            'cluster_comparison',
            'temporal_heatmap',
            'anomaly_timeline',
            'building_profiles'
        ],
        'output_format': 'html',
        'interactive': True
    },
    dag=dag,
)

# Task 10: Create Optimization Report
report_task = PythonOperator(
    task_id='create_optimization_report',
    python_callable=create_optimization_report,
    op_args=[evaluation_task.output, anomaly_task.output, dashboard_task.output],
    op_kwargs={
        'include_recommendations': True,
        'cost_analysis': True,
        'sustainability_metrics': True,
        'format': 'pdf'
    },
    dag=dag,
)

# Task 11: Export Model Artifacts for Production
export_task = PythonOperator(
    task_id='export_model_artifacts',
    python_callable=export_model_artifacts,
    op_args=[evaluation_task.output],
    op_kwargs={
        'export_format': ['pickle', 'onnx', 'pmml'],
        'include_preprocessing': True,
        'create_api_endpoint': True
    },
    dag=dag,
)

# Bonus Task: Clean up temporary files
cleanup_task = BashOperator(
    task_id='cleanup_temp_files',
    bash_command='find /opt/airflow/working_data/temp -type f -mtime +7 -delete',
    trigger_rule='all_done',
    dag=dag,
)

# Define task dependencies with parallel processing
generate_data_task >> load_validate_task >> feature_engineering_task >> temporal_task >> pca_task

# Parallel model building
pca_task >> [kmeans_task, gmm_task, hierarchical_task]

# Convergence for evaluation
[kmeans_task, gmm_task, hierarchical_task] >> evaluation_task

# Sequential final steps
evaluation_task >> anomaly_task >> dashboard_task >> report_task

# Parallel export and cleanup
report_task >> [export_task, cleanup_task]

# Add comprehensive documentation
dag.doc_md = """
## 🏙️ Smart City Energy Consumption Pattern Analysis Pipeline

### 📊 Overview
This advanced pipeline analyzes energy consumption patterns from IoT sensors across smart buildings
to identify usage patterns, detect anomalies, and provide optimization recommendations.

### 🎯 Business Value
- **Cost Reduction**: Identify inefficient energy usage patterns
- **Predictive Maintenance**: Detect anomalies before equipment failure
- **Sustainability**: Optimize energy consumption for carbon reduction
- **Urban Planning**: Understand city-wide energy patterns

### 🔬 Technical Approach

#### 1. **Data Generation/Ingestion**
   - Simulates 500 buildings with hourly sensor readings
   - Includes weather data and special events
   - Incorporates realistic anomaly patterns

#### 2. **Feature Engineering**
   - **Temporal Features**: Hour, day, week, season cycles
   - **Lag Features**: Previous consumption patterns
   - **Rolling Statistics**: Moving averages and variations
   - **Weather Interactions**: Temperature-consumption relationships

#### 3. **Advanced Clustering Algorithms**
   - **K-Means**: Traditional clustering with advanced optimization
   - **Gaussian Mixture Model**: Probabilistic clustering for overlapping patterns
   - **Hierarchical Clustering**: Building taxonomy creation

#### 4. **Optimization Techniques**
   - **Elbow Method**: Classical approach for optimal K
   - **Silhouette Analysis**: Cluster separation quality
   - **Gap Statistic**: Statistical method for cluster count
   - **BIC/AIC**: Information criteria for model selection

#### 5. **Anomaly Detection Ensemble**
   - **Isolation Forest**: Tree-based anomaly detection
   - **Local Outlier Factor**: Density-based detection
   - **One-Class SVM**: Boundary-based detection

### 📈 Evaluation Metrics
- **Silhouette Score**: Cluster cohesion and separation
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance
- **Davies-Bouldin Index**: Average similarity between clusters
- **Dunn Index**: Ratio of minimum inter-cluster to maximum intra-cluster distance
- **Cluster Stability**: Bootstrap validation of cluster assignments

### 🎨 Visualizations
- 3D scatter plots of energy patterns
- Temporal heatmaps showing consumption trends
- Dendrograms for building hierarchy
- Anomaly detection timelines
- Interactive dashboards with drill-down capabilities

### 📦 Output Artifacts
- Trained models in multiple formats (pickle, ONNX, PMML)
- PDF reports with optimization recommendations
- Interactive HTML dashboards
- API endpoint specifications

### 🔄 Pipeline Features
- **Parallel Processing**: Multiple models train simultaneously
- **Error Handling**: Graceful failure with retries
- **Data Validation**: Quality checks at each step
- **Model Versioning**: Track model iterations
- **Production Ready**: Export formats for deployment

### 📊 Expected Results
- 3-5 distinct energy consumption patterns
- 3% anomaly detection rate
- 15-20% potential energy savings identified
- Building profiles for targeted interventions

### 🚀 Extension Possibilities
- Real-time streaming with Apache Kafka
- Deep learning with LSTM for forecasting
- Reinforcement learning for optimization
- Integration with building management systems
"""

if __name__ == "__main__":
    dag.cli()
