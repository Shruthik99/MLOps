"""
Smart City Energy Analysis Pipeline
Airflow_Lab1 with multiple clustering algorithms
"""

# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, load_model_elbow
from airflow import configuration as conf

# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')

# Define default arguments for your DAG
default_args = {
    'owner': 'smart_city_team',  # Changed to reflect new project
    'start_date': datetime(2024, 1, 1),
    'retries': 1,  # Increased retries for robustness
    'retry_delay': timedelta(minutes=5),
}

# Create a DAG instance - keeping original name for compatibility
dag = DAG(
    'Airflow_Lab1_SmartCity',  # Enhanced name but recognizable
    default_args=default_args,
    description='Smart City Energy Analysis with Multiple Clustering Algorithms',
    schedule_interval=None,  # Manual triggering
    catchup=False,
    tags=['energy', 'clustering', 'smart-city', 'mlops']  # Added tags
)

# Task 1: Load data
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
    doc_md="""
    ## Load Smart City Energy Data
    Loads energy consumption data from CSV file.
    If data doesn't exist, generates synthetic smart city data.
    """
)

# Task 2: Data preprocessing with feature engineering
data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
    doc_md="""
    ## Advanced Data Preprocessing
    - Feature selection and engineering
    - Standardization with StandardScaler
    - Handles missing values
    - Prepares data for multiple clustering algorithms
    """
)

# Task 3: Build and save multiple models
build_save_model_task = PythonOperator(
    task_id='build_save_model_task',
    python_callable=build_save_model,
    op_args=[data_preprocessing_task.output, "model.sav"],
    provide_context=True,
    dag=dag,
    doc_md="""
    ## Build Multiple Clustering Models
    Implements and compares:
    1. K-Means with Elbow Method
    2. DBSCAN for density-based clustering
    3. Hierarchical Clustering
    4. Isolation Forest for anomaly detection
    
    Saves all models and generates comparison report.
    """
)

# Task 4: Load model and analyze results
load_model_task = PythonOperator(
    task_id='load_model_task',
    python_callable=load_model_elbow,
    op_args=["model.sav", build_save_model_task.output],
    dag=dag,
    doc_md="""
    ## Final Analysis and Reporting
    - Determines optimal number of clusters
    - Loads saved models
    - Displays comprehensive results
    - Generates final report
    """
)

# Task 5: Data quality check (Enhancement)
data_quality_task = BashOperator(
    task_id='data_quality_check',
    bash_command="""
    echo "================================"
    echo "🔍 Data Quality Check"
    echo "================================"
    if [ -f /opt/airflow/dags/data/file.csv ]; then
        echo "✅ Training data exists:"
        wc -l /opt/airflow/dags/data/file.csv
    else
        echo "⚠️ Training data will be generated"
    fi
    echo "================================"
    """,
    dag=dag,
    trigger_rule='all_success'
)

# Task 6: Summary report (Enhancement)
summary_task = BashOperator(
    task_id='generate_summary',
    bash_command="""
    echo "============================================="
    echo "📊 SMART CITY ENERGY ANALYSIS COMPLETE"
    echo "============================================="
    echo ""
    echo "📁 Results Generated:"
    echo "-------------------"
    
    if [ -d /opt/airflow/dags/results ]; then
        echo "✅ Results folder created"
        ls -la /opt/airflow/dags/results/ 2>/dev/null || echo "   No result files yet"
    fi
    
    echo ""
    echo "🤖 Models Saved:"
    echo "---------------"
    if [ -d /opt/airflow/dags/model ]; then
        echo "✅ Model folder contains:"
        ls -la /opt/airflow/dags/model/*.pkl 2>/dev/null || echo "   No model files yet"
        ls -la /opt/airflow/dags/model/*.sav 2>/dev/null || echo ""
    fi
    
    echo ""
    echo "📊 Key Features of This Enhanced Lab:"
    echo "------------------------------------"
    echo "  ✓ 3 Clustering Algorithms (K-Means, DBSCAN, Hierarchical)"
    echo "  ✓ Anomaly Detection with Isolation Forest"
    echo "  ✓ PCA Dimensionality Reduction"
    echo "  ✓ Multiple Evaluation Metrics"
    echo "  ✓ Smart City Context with Energy Data"
    echo "  ✓ Comprehensive JSON Reports"
    echo ""
    echo "============================================="
    echo "🎉 Pipeline execution successful!"
    echo "============================================="
    """,
    dag=dag,
    trigger_rule='all_success'
)

# Set task dependencies
load_data_task >> data_quality_task >> data_preprocessing_task >> build_save_model_task >> load_model_task >> summary_task

# Add comprehensive documentation
dag.doc_md = """
# 🏙️ Smart City Energy Analysis Pipeline

## Overview
This Airflow_Lab1 analyzes energy consumption patterns in a smart city environment using multiple clustering algorithms and anomaly detection.



### Original Lab Features:
- Single K-Means clustering
- Basic Elbow method
- Simple preprocessing
- Single model output

### This Enhanced Version Adds:
1. **Multiple Clustering Algorithms:**
   - K-Means with automatic optimal k detection
   - DBSCAN for density-based clustering
   - Hierarchical clustering for relationship analysis

2. **Anomaly Detection:**
   - Isolation Forest to detect unusual energy consumption
   - Identifies potential energy theft or equipment malfunction

3. **Advanced Analytics:**
   - PCA for dimensionality reduction
   - Silhouette score for cluster validation
   - Comprehensive comparison metrics

4. **Smart City Context:**
   - Energy consumption analysis
   - Building efficiency scoring
   - Zone-based pattern recognition

5. **Production Features:**
   - Error handling and logging
   - Model versioning
   - JSON reports for integration
   - Data quality checks

## 📊 Data Features
- Building energy consumption
- Temperature impact
- Occupancy rates
- Solar generation
- Efficiency scores
- Time-based patterns

## 🎯 Business Value
- Identify energy waste: Save costs
- Detect anomalies: Prevent theft
- Optimize consumption: Improve sustainability
- Zone analysis: Better city planning

## 📈 Expected Results
- 4-5 optimal clusters for energy patterns
- ~5% anomaly detection rate
- Comparison showing best algorithm
- Actionable insights for city management

## 🔧 Technical Stack
- Apache Airflow for orchestration
- Scikit-learn for ML algorithms
- PCA for dimensionality reduction
- Isolation Forest for anomaly detection
- JSON reporting for integration

## 📝 Usage
1. Trigger DAG from Airflow UI
2. Monitor progress in Graph View
3. Check task logs for details
4. Review results in `/dags/results/`
5. Access models in `/dags/model/`

## ⏱️ Performance
- Total runtime: ~2-3 minutes
- Processes 10,000 records
- Generates 10+ features
- Creates multiple models
- Produces comprehensive reports
"""

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.cli()
