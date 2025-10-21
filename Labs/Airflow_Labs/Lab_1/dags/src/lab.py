# dags/src/lab.py
"""
Enhanced Lab Functions for Smart City Energy Consumption Pattern Analysis
Implements advanced clustering algorithms and comprehensive evaluation metrics
"""

import numpy as np
import pandas as pd
import pickle
import json
import warnings
import time
from datetime import datetime, timedelta
from pathlib import Path
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.impute import KNNImputer
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import cdist
from kneed import KneeLocator
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import plotly.graph_objects as go
import plotly.express as px

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create necessary directories
WORKING_DIR = Path('/opt/airflow/working_data')
MODEL_DIR = WORKING_DIR / 'models'
VIZ_DIR = WORKING_DIR / 'visualizations'
REPORT_DIR = WORKING_DIR / 'reports'
TEMP_DIR = WORKING_DIR / 'temp'

for dir_path in [MODEL_DIR, VIZ_DIR, REPORT_DIR, TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def generate_energy_data(n_buildings=500, days=90, sensor_frequency='hourly', 
                         building_types=['residential', 'commercial', 'industrial', 'public'],
                         include_weather=True, include_events=True, anomaly_rate=0.03):
    """
    Generate synthetic smart city energy consumption data with realistic patterns
    """
    np.random.seed(42)
    
    # Calculate total data points
    hours_per_day = 24 if sensor_frequency == 'hourly' else 1
    total_hours = days * hours_per_day
    
    data_list = []
    
    # Building type characteristics
    type_profiles = {
        'residential': {
            'base_load': 1.5, 'peak_hours': [7, 19], 'weekend_factor': 1.2,
            'seasonal_sensitivity': 0.3, 'size_range': (50, 200)
        },
        'commercial': {
            'base_load': 5.0, 'peak_hours': [9, 17], 'weekend_factor': 0.3,
            'seasonal_sensitivity': 0.2, 'size_range': (500, 2000)
        },
        'industrial': {
            'base_load': 20.0, 'peak_hours': [6, 22], 'weekend_factor': 0.8,
            'seasonal_sensitivity': 0.1, 'size_range': (1000, 5000)
        },
        'public': {
            'base_load': 3.0, 'peak_hours': [8, 20], 'weekend_factor': 0.5,
            'seasonal_sensitivity': 0.25, 'size_range': (200, 1000)
        }
    }
    
    # Generate data for each building
    for building_id in range(n_buildings):
        building_type = np.random.choice(building_types, p=[0.4, 0.3, 0.2, 0.1])
        profile = type_profiles[building_type]
        building_size = np.random.uniform(*profile['size_range'])
        
        # Generate time series
        timestamps = pd.date_range(start='2024-01-01', periods=total_hours, freq='h')
        
        for timestamp in timestamps:
            hour = timestamp.hour
            day_of_week = timestamp.dayofweek
            month = timestamp.month
            
            # Base consumption
            base_consumption = profile['base_load'] * (building_size / 100)
            
            # Time of day pattern
            if profile['peak_hours'][0] <= hour <= profile['peak_hours'][1]:
                time_factor = 1.5 + 0.3 * np.sin((hour - profile['peak_hours'][0]) * np.pi / 
                                                  (profile['peak_hours'][1] - profile['peak_hours'][0]))
            else:
                time_factor = 0.6 + 0.2 * np.random.normal(0, 0.1)
            
            # Weekend adjustment
            weekend_factor = profile['weekend_factor'] if day_of_week >= 5 else 1.0
            
            # Seasonal pattern (simplified)
            seasonal_factor = 1 + profile['seasonal_sensitivity'] * np.sin((month - 1) * np.pi / 6)
            
            # Weather impact
            if include_weather:
                temp = 20 + 10 * np.sin((timestamp.dayofyear - 80) * 2 * np.pi / 365) + np.random.normal(0, 3)
                weather_factor = 1 + 0.02 * abs(temp - 20)  # Increased consumption for heating/cooling
            else:
                weather_factor = 1.0
                temp = 20
            
            # Special events
            event_factor = 1.0
            if include_events:
                # Random events (holidays, conferences, etc.)
                if np.random.random() < 0.02:  # 2% chance of special event
                    event_factor = np.random.uniform(0.5, 1.5)
            
            # Calculate final consumption
            consumption = (base_consumption * time_factor * weekend_factor * 
                          seasonal_factor * weather_factor * event_factor)
            
            # Add noise
            consumption += np.random.normal(0, consumption * 0.1)
            
            # Inject anomalies
            is_anomaly = 0
            if np.random.random() < anomaly_rate:
                is_anomaly = 1
                anomaly_type = np.random.choice(['spike', 'drop', 'shift'])
                if anomaly_type == 'spike':
                    consumption *= np.random.uniform(2, 5)
                elif anomaly_type == 'drop':
                    consumption *= np.random.uniform(0.1, 0.3)
                else:  # shift
                    consumption += np.random.uniform(-consumption, consumption)
            
            # Create record
            record = {
                'building_id': building_id,
                'building_type': building_type,
                'building_size': building_size,
                'timestamp': timestamp,
                'hour': hour,
                'day_of_week': day_of_week,
                'month': month,
                'consumption_kwh': max(0, consumption),
                'temperature': temp,
                'is_weekend': int(day_of_week >= 5),
                'is_anomaly': is_anomaly
            }
            
            data_list.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data_list)
    
    # Add some aggregate features
    df['consumption_per_sqm'] = df['consumption_kwh'] / df['building_size']
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Save to CSV
    df.to_csv(WORKING_DIR / 'energy_consumption_data.csv', index=False)
    
    print(f"Generated energy data: {len(df)} records")
    print(f"Buildings: {n_buildings}")
    print(f"Time period: {days} days")
    print(f"Features: {list(df.columns)}")
    print(f"Anomaly rate: {df['is_anomaly'].mean()*100:.2f}%")
    
    return pickle.dumps(df)

def load_validate_energy_data(data, check_missing=True, check_outliers=True, validate_ranges=True):
    """
    Load and validate energy consumption data with comprehensive quality checks
    """
    df = pickle.loads(data)
    
    validation_report = {
        'timestamp': datetime.now().isoformat(),
        'total_records': len(df),
        'unique_buildings': df['building_id'].nunique(),
        'date_range': {
            'start': str(df['timestamp'].min()),
            'end': str(df['timestamp'].max())
        },
        'issues': []
    }
    
    # Check missing values
    if check_missing:
        missing = df.isnull().sum()
        if missing.any():
            validation_report['issues'].append({
                'type': 'missing_values',
                'details': missing[missing > 0].to_dict()
            })
            # Impute missing values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Check outliers using IQR method
    if check_outliers:
        outlier_cols = ['consumption_kwh', 'consumption_per_sqm']
        for col in outlier_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 3*IQR) | (df[col] > Q3 + 3*IQR)).sum()
            if outliers > 0:
                validation_report['issues'].append({
                    'type': 'outliers',
                    'column': col,
                    'count': int(outliers),
                    'percentage': float(outliers / len(df) * 100)
                })
    
    # Validate ranges
    if validate_ranges:
        if (df['consumption_kwh'] < 0).any():
            validation_report['issues'].append({
                'type': 'invalid_range',
                'details': 'Negative consumption values found'
            })
            df['consumption_kwh'] = df['consumption_kwh'].abs()
        
        if (df['temperature'] < -50).any() or (df['temperature'] > 60).any():
            validation_report['issues'].append({
                'type': 'invalid_range',
                'details': 'Temperature values out of realistic range'
            })
    
    # Summary statistics
    validation_report['summary_stats'] = {
        'consumption': {
            'mean': float(df['consumption_kwh'].mean()),
            'median': float(df['consumption_kwh'].median()),
            'std': float(df['consumption_kwh'].std()),
            'min': float(df['consumption_kwh'].min()),
            'max': float(df['consumption_kwh'].max())
        }
    }
    
    # Save validation report
    with open(REPORT_DIR / 'data_validation_report.json', 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    print(f"Data validation complete: {len(validation_report['issues'])} issues found")
    print(f"Summary: {validation_report['summary_stats']['consumption']}")
    
    return pickle.dumps(df)

def feature_engineering(data, create_lag_features=True, create_rolling_stats=True,
                        create_cyclic_features=True, create_interaction_terms=True,
                        window_sizes=[24, 168, 720]):
    """
    Advanced feature engineering for energy consumption patterns
    """
    df = pickle.loads(data)
    
    print("Starting feature engineering...")
    
    # Sort by building and time for proper feature creation
    df = df.sort_values(['building_id', 'timestamp'])
    
    # Lag features
    if create_lag_features:
        lag_periods = [1, 24, 168]  # 1 hour, 1 day, 1 week
        for lag in lag_periods:
            df[f'consumption_lag_{lag}h'] = df.groupby('building_id')['consumption_kwh'].shift(lag)
    
    # Rolling statistics
    if create_rolling_stats:
        for window in window_sizes:
            df[f'consumption_mean_{window}h'] = df.groupby('building_id')['consumption_kwh'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'consumption_std_{window}h'] = df.groupby('building_id')['consumption_kwh'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            df[f'consumption_trend_{window}h'] = (
                df[f'consumption_mean_{window}h'] - df[f'consumption_lag_{window}h']
            )
    
    # Cyclic features (already added in generation, but ensure they're present)
    if create_cyclic_features:
        if 'hour_sin' not in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        if 'day_sin' not in df.columns:
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Interaction terms
    if create_interaction_terms:
        df['temp_consumption_interaction'] = df['temperature'] * df['consumption_kwh']
        df['size_normalized_consumption'] = df['consumption_kwh'] / df['building_size']
        df['peak_hour_consumption'] = df['consumption_kwh'] * df['hour'].apply(
            lambda x: 1 if 9 <= x <= 17 else 0.5
        )
    
    # Drop rows with NaN from lag features
    df = df.dropna()
    
    print(f"Feature engineering complete. New shape: {df.shape}")
    print(f"Total features: {len(df.columns)}")
    
    return pickle.dumps(df)

def temporal_analysis(data, decomposition_method='STL', analyze_seasonality=True, 
                     detect_changepoints=True):
    """
    Analyze temporal patterns in energy consumption
    """
    df = pickle.loads(data)
    
    analysis_results = {
        'patterns': {},
        'seasonality': {},
        'changepoints': []
    }
    
    # Aggregate by hour for pattern analysis
    hourly_consumption = df.groupby('hour')['consumption_kwh'].mean()
    daily_consumption = df.groupby('day_of_week')['consumption_kwh'].mean()
    monthly_consumption = df.groupby('month')['consumption_kwh'].mean()
    
    # Identify peak hours
    peak_hours = hourly_consumption.nlargest(3).index.tolist()
    off_peak_hours = hourly_consumption.nsmallest(3).index.tolist()
    
    analysis_results['patterns'] = {
        'peak_hours': peak_hours,
        'off_peak_hours': off_peak_hours,
        'busiest_days': daily_consumption.nlargest(2).index.tolist(),
        'quietest_days': daily_consumption.nsmallest(2).index.tolist()
    }
    
    # Seasonality analysis
    if analyze_seasonality:
        # Simple seasonality strength calculation
        seasonal_variance = monthly_consumption.var()
        total_variance = df['consumption_kwh'].var()
        seasonality_strength = seasonal_variance / total_variance
        
        analysis_results['seasonality'] = {
            'strength': float(seasonality_strength),
            'peak_months': monthly_consumption.nlargest(3).index.tolist(),
            'low_months': monthly_consumption.nsmallest(3).index.tolist()
        }
    
    # Change point detection (simplified)
    if detect_changepoints:
        # Rolling window change detection
        window = 168  # 1 week
        rolling_mean = df.groupby('building_id')['consumption_kwh'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        rolling_std = df.groupby('building_id')['consumption_kwh'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        
        # Detect significant changes
        z_scores = np.abs((df['consumption_kwh'] - rolling_mean) / (rolling_std + 1e-10))
        changepoints = df[z_scores > 3]['timestamp'].value_counts().head(10)
        
        analysis_results['changepoints'] = [str(cp) for cp in changepoints.index.tolist()]
    
    # Save analysis results
    with open(REPORT_DIR / 'temporal_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"Temporal analysis complete:")
    print(f"Peak hours: {peak_hours}")
    print(f"Seasonality strength: {analysis_results['seasonality'].get('strength', 0):.3f}")
    
    return pickle.dumps(df)

def perform_pca_analysis(data, variance_threshold=0.95, n_components=None, 
                        visualize=True, use_kernel_pca=False):
    """
    Perform PCA for dimensionality reduction with visualization
    """
    df = pickle.loads(data)
    
    # Select features for clustering
    feature_cols = [col for col in df.columns if col not in [
        'building_id', 'timestamp', 'building_type', 'is_anomaly'
    ]]
    
    X = df[feature_cols].values
    
    # Scale the features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    if use_kernel_pca:
        pca = KernelPCA(n_components=n_components or 10, kernel='rbf')
    else:
        pca = PCA(n_components=n_components)
    
    X_pca = pca.fit_transform(X_scaled)
    
    # Determine number of components to retain
    if not use_kernel_pca:
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        n_components_keep = np.argmax(cumsum_var >= variance_threshold) + 1
        
        print(f"PCA Analysis:")
        print(f"Original features: {len(feature_cols)}")
        print(f"Components to retain {variance_threshold*100}% variance: {n_components_keep}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_[:5]}")
    else:
        n_components_keep = X_pca.shape[1]
    
    # Visualization
    if visualize and not use_kernel_pca:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Scree plot
        axes[0].bar(range(1, min(11, len(pca.explained_variance_ratio_)+1)), 
                    pca.explained_variance_ratio_[:10])
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('PCA Scree Plot')
        
        # Cumulative variance
        axes[1].plot(range(1, len(cumsum_var)+1), cumsum_var, 'bo-')
        axes[1].axhline(y=variance_threshold, color='r', linestyle='--', 
                       label=f'{variance_threshold*100}% threshold')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Variance Explained')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(VIZ_DIR / 'pca_analysis.png', dpi=150)
        plt.close()
    
    # Prepare output data
    result = {
        'X_pca': X_pca[:, :n_components_keep],
        'X_scaled': X_scaled,
        'scaler': scaler,
        'pca_model': pca,
        'feature_names': feature_cols,
        'df': df,
        'n_components': n_components_keep
    }
    
    return pickle.dumps(result)

def build_kmeans_model(data, k_range=(3, 20), optimization_methods=['elbow', 'silhouette'],
                      init_methods=['k-means++'], n_init=20, max_iter=500):
    """
    Build K-Means model with multiple optimization techniques
    """
    data_dict = pickle.loads(data)
    X = data_dict['X_pca']
    
    print(f"Building K-Means models for k in range{k_range}...")
    
    results = {
        'models': {},
        'scores': {},
        'optimal_k': {},
        'runtime': {}
    }
    
    # Initialize score lists
    for method in optimization_methods:
        results['scores'][method] = []
    
    k_values = list(range(k_range[0], k_range[1] + 1))
    
    # Train models for different k values
    for k in k_values:
        start_time = time.time()
        
        # Try different initialization methods
        best_model = None
        best_inertia = float('inf')
        
        for init_method in init_methods:
            kmeans = KMeans(n_clusters=k, init=init_method, n_init=n_init, 
                          max_iter=max_iter, random_state=42)
            kmeans.fit(X)
            
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_model = kmeans
        
        results['models'][k] = best_model
        results['runtime'][k] = time.time() - start_time
        
        # Calculate scores
        labels = best_model.labels_
        
        if 'elbow' in optimization_methods:
            results['scores']['elbow'].append(best_model.inertia_)
        
        if 'silhouette' in optimization_methods:
            if k > 1:  # Silhouette requires at least 2 clusters
                sil_score = silhouette_score(X, labels)
                results['scores']['silhouette'].append(sil_score)
            else:
                results['scores']['silhouette'].append(-1)
        
        if 'davies_bouldin' in optimization_methods:
            if k > 1:
                db_score = davies_bouldin_score(X, labels)
                results['scores']['davies_bouldin'].append(db_score)
            else:
                results['scores']['davies_bouldin'].append(float('inf'))
        
        if 'calinski_harabasz' in optimization_methods:
            if k > 1:
                ch_score = calinski_harabasz_score(X, labels)
                results['scores']['calinski_harabasz'].append(ch_score)
            else:
                results['scores']['calinski_harabasz'].append(0)
        
        print(f"K={k}: Inertia={best_inertia:.2f}, Runtime={results['runtime'][k]:.2f}s")
    
    # Determine optimal k for each method
    if 'elbow' in optimization_methods:
        # Use kneed library for elbow detection
        kneedle = KneeLocator(k_values, results['scores']['elbow'], 
                             curve='convex', direction='decreasing')
        results['optimal_k']['elbow'] = kneedle.knee or k_values[len(k_values)//2]
    
    if 'silhouette' in optimization_methods:
        # Maximum silhouette score
        valid_scores = [(k, s) for k, s in zip(k_values[1:], results['scores']['silhouette'][1:]) if s > -1]
        if valid_scores:
            results['optimal_k']['silhouette'] = max(valid_scores, key=lambda x: x[1])[0]
    
    if 'davies_bouldin' in optimization_methods:
        # Minimum Davies-Bouldin score
        valid_scores = [(k, s) for k, s in zip(k_values[1:], results['scores']['davies_bouldin'][1:]) 
                       if s != float('inf')]
        if valid_scores:
            results['optimal_k']['davies_bouldin'] = min(valid_scores, key=lambda x: x[1])[0]
    
    # Consensus optimal k (majority vote or average)
    if results['optimal_k']:
        optimal_k_values = list(results['optimal_k'].values())
        results['consensus_k'] = int(np.median(optimal_k_values))
    else:
        results['consensus_k'] = k_values[len(k_values)//2]
    
    # Save the best model
    best_k = results['consensus_k']
    best_model = results['models'][best_k]
    
    with open(MODEL_DIR / 'kmeans_best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    # Create elbow plot
    if 'elbow' in optimization_methods:
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, results['scores']['elbow'], 'bo-')
        plt.axvline(x=results['optimal_k'].get('elbow', best_k), color='r', 
                   linestyle='--', label=f'Optimal k={results["optimal_k"].get("elbow", best_k)}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Sum of Squared Distances (Inertia)')
        plt.title('K-Means Elbow Method')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(VIZ_DIR / 'kmeans_elbow.png', dpi=150)
        plt.close()
    
    print(f"\nOptimal k values: {results['optimal_k']}")
    print(f"Consensus k: {results['consensus_k']}")
    
    # Add cluster assignments to results
    results['cluster_labels'] = best_model.labels_
    results['data'] = data_dict
    
    return pickle.dumps(results)

def build_gaussian_mixture(data, n_components_range=(3, 20), 
                          covariance_types=['full', 'tied', 'diag', 'spherical'],
                          selection_criterion='bic', n_init=10):
    """
    Build Gaussian Mixture Model for soft clustering
    """
    data_dict = pickle.loads(data)
    X = data_dict['X_pca']
    
    print(f"Building Gaussian Mixture Models...")
    
    results = {
        'models': {},
        'scores': {},
        'optimal_params': {},
        'runtime': {}
    }
    
    n_components_values = list(range(n_components_range[0], n_components_range[1] + 1))
    
    best_score = float('inf') if selection_criterion == 'bic' else float('-inf')
    best_model = None
    best_params = {}
    
    # Grid search over parameters
    for n_components in n_components_values:
        for cov_type in covariance_types:
            start_time = time.time()
            
            gmm = GaussianMixture(n_components=n_components, 
                                 covariance_type=cov_type,
                                 n_init=n_init, 
                                 random_state=42)
            gmm.fit(X)
            
            runtime = time.time() - start_time
            
            # Calculate selection criterion
            if selection_criterion == 'bic':
                score = gmm.bic(X)
                if score < best_score:
                    best_score = score
                    best_model = gmm
                    best_params = {'n_components': n_components, 'covariance_type': cov_type}
            else:  # AIC
                score = gmm.aic(X)
                if score < best_score:
                    best_score = score
                    best_model = gmm
                    best_params = {'n_components': n_components, 'covariance_type': cov_type}
            
            # Store results
            key = f"{n_components}_{cov_type}"
            results['models'][key] = gmm
            results['scores'][key] = score
            results['runtime'][key] = runtime
            
            print(f"n_components={n_components}, cov_type={cov_type}: {selection_criterion.upper()}={score:.2f}")
    
    results['best_model'] = best_model
    results['best_params'] = best_params
    results['best_score'] = best_score
    
    # Get cluster probabilities and assignments
    results['cluster_probs'] = best_model.predict_proba(X)
    results['cluster_labels'] = best_model.predict(X)
    
    # Save best model
    with open(MODEL_DIR / 'gmm_best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    # Visualize BIC/AIC scores
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for cov_type in covariance_types:
        scores = []
        for n_comp in n_components_values:
            key = f"{n_comp}_{cov_type}"
            if key in results['scores']:
                scores.append(results['scores'][key])
        ax.plot(n_components_values[:len(scores)], scores, marker='o', label=cov_type)
    
    ax.set_xlabel('Number of Components')
    ax.set_ylabel(selection_criterion.upper())
    ax.set_title(f'Gaussian Mixture Model Selection ({selection_criterion.upper()})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mark the best model
    ax.axvline(x=best_params['n_components'], color='red', linestyle='--', 
              label=f'Best: n={best_params["n_components"]}, type={best_params["covariance_type"]}')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f'gmm_{selection_criterion}.png', dpi=150)
    plt.close()
    
    print(f"\nBest GMM parameters: {best_params}")
    print(f"Best {selection_criterion.upper()} score: {best_score:.2f}")
    
    results['data'] = data_dict
    
    return pickle.dumps(results)

def build_hierarchical_clustering(data, linkage_methods=['ward', 'complete', 'average'],
                                 distance_threshold=None, n_clusters=None, 
                                 create_dendrogram=True):
    """
    Build Hierarchical Clustering model with dendrogram visualization
    """
    data_dict = pickle.loads(data)
    X = data_dict['X_pca']
    
    # Sample for dendrogram if dataset is large
    if len(X) > 1000:
        sample_idx = np.random.choice(len(X), 1000, replace=False)
        X_sample = X[sample_idx]
    else:
        X_sample = X
    
    print(f"Building Hierarchical Clustering models...")
    
    results = {
        'models': {},
        'linkage_matrices': {},
        'scores': {},
        'optimal_params': {}
    }
    
    best_score = -1
    best_model = None
    best_linkage = None
    
    for method in linkage_methods:
        print(f"Testing linkage method: {method}")
        
        # Create linkage matrix
        Z = linkage(X_sample, method=method)
        results['linkage_matrices'][method] = Z
        
        # Determine number of clusters if not specified
        if n_clusters is None:
            # Use elbow method on dendrogram distances
            distances = Z[-10:, 2]
            if len(distances) > 1:
                diff = np.diff(distances)
                if len(diff) > 0:
                    n_clusters_auto = len(distances) - np.argmax(diff) - 1
                else:
                    n_clusters_auto = 3
            else:
                n_clusters_auto = 3
        else:
            n_clusters_auto = n_clusters
        
        # Build model
        model = AgglomerativeClustering(n_clusters=n_clusters_auto, 
                                       linkage=method)
        labels = model.fit_predict(X)
        
        results['models'][method] = model
        
        # Calculate silhouette score
        if n_clusters_auto > 1:
            score = silhouette_score(X, labels)
            results['scores'][method] = score
            
            if score > best_score:
                best_score = score
                best_model = model
                best_linkage = method
        
        print(f"  n_clusters={n_clusters_auto}, silhouette={results['scores'].get(method, -1):.3f}")
    
    # Create dendrogram for best method
    if create_dendrogram and best_linkage:
        plt.figure(figsize=(15, 8))
        dendrogram(results['linkage_matrices'][best_linkage], 
                  truncate_mode='lastp', p=30, leaf_rotation=90)
        plt.title(f'Hierarchical Clustering Dendrogram (Method: {best_linkage})')
        plt.xlabel('Cluster Size')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.savefig(VIZ_DIR / 'hierarchical_dendrogram.png', dpi=150)
        plt.close()
    
    results['best_model'] = best_model
    results['best_linkage'] = best_linkage
    results['cluster_labels'] = best_model.labels_ if best_model else np.zeros(len(X))
    results['data'] = data_dict
    
    # Save best model
    with open(MODEL_DIR / 'hierarchical_best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"\nBest linkage method: {best_linkage}")
    print(f"Best silhouette score: {best_score:.3f}")
    
    return pickle.dumps(results)

def evaluate_clustering_models(kmeans_results, gmm_results, hierarchical_results, pca_data,
                              metrics=['silhouette', 'calinski_harabasz', 'davies_bouldin'],
                              cross_validate=True):
    """
    Comprehensive evaluation of all clustering models
    """
    kmeans_dict = pickle.loads(kmeans_results)
    gmm_dict = pickle.loads(gmm_results)
    hier_dict = pickle.loads(hierarchical_results)
    data_dict = pickle.loads(pca_data)
    
    X = data_dict['X_pca']
    
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    evaluation = {
        'models': {},
        'best_model': None,
        'best_score': -1,
        'comparison_matrix': {}
    }
    
    # Evaluate each model
    models_to_evaluate = [
        ('KMeans', kmeans_dict['models'][kmeans_dict['consensus_k']], 
         kmeans_dict['cluster_labels']),
        ('GMM', gmm_dict['best_model'], gmm_dict['cluster_labels']),
        ('Hierarchical', hier_dict['best_model'], hier_dict['cluster_labels'])
    ]
    
    for model_name, model, labels in models_to_evaluate:
        print(f"\nEvaluating {model_name}:")
        model_scores = {}
        
        # Calculate metrics
        if 'silhouette' in metrics and len(np.unique(labels)) > 1:
            model_scores['silhouette'] = silhouette_score(X, labels)
            print(f"  Silhouette Score: {model_scores['silhouette']:.3f}")
        
        if 'calinski_harabasz' in metrics and len(np.unique(labels)) > 1:
            model_scores['calinski_harabasz'] = calinski_harabasz_score(X, labels)
            print(f"  Calinski-Harabasz: {model_scores['calinski_harabasz']:.2f}")
        
        if 'davies_bouldin' in metrics and len(np.unique(labels)) > 1:
            model_scores['davies_bouldin'] = davies_bouldin_score(X, labels)
            print(f"  Davies-Bouldin: {model_scores['davies_bouldin']:.3f}")
        
        # Dunn Index (custom implementation)
        if 'dunn_index' in metrics and len(np.unique(labels)) > 1:
            model_scores['dunn_index'] = calculate_dunn_index(X, labels)
            print(f"  Dunn Index: {model_scores['dunn_index']:.3f}")
        
        # Cluster stability (using bootstrap)
        if 'cluster_stability' in metrics and cross_validate:
            stability = calculate_cluster_stability(X, model, n_iterations=10)
            model_scores['cluster_stability'] = stability
            print(f"  Cluster Stability: {stability:.3f}")
        
        evaluation['models'][model_name] = {
            'scores': model_scores,
            'n_clusters': len(np.unique(labels)),
            'cluster_sizes': pd.Series(labels).value_counts().to_dict()
        }
        
        # Track best model (using silhouette as primary metric)
        if model_scores.get('silhouette', -1) > evaluation['best_score']:
            evaluation['best_score'] = model_scores['silhouette']
            evaluation['best_model'] = model_name
    
    # Create comparison matrix
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    comparison_df = pd.DataFrame({
        model: data['scores'] 
        for model, data in evaluation['models'].items()
    }).T
    
    print(comparison_df.round(3))
    
    # Save evaluation results
    with open(REPORT_DIR / 'model_evaluation.json', 'w') as f:
        json.dump(evaluation, f, indent=2, default=str)
    
    # Create visualization
    create_comparison_visualization(evaluation)
    
    print(f"\nBest Model: {evaluation['best_model']}")
    
    # Return results with selected model's labels
    if evaluation['best_model'] == 'KMeans':
        evaluation['selected_labels'] = kmeans_dict['cluster_labels']
    elif evaluation['best_model'] == 'GMM':
        evaluation['selected_labels'] = gmm_dict['cluster_labels']
    else:
        evaluation['selected_labels'] = hier_dict['cluster_labels']
    
    evaluation['data'] = data_dict
    
    return pickle.dumps(evaluation)

def detect_anomalies(evaluation_results, pca_data, methods=['isolation_forest'], 
                    contamination='auto', ensemble_method='voting'):
    """
    Detect anomalies using multiple methods and ensemble
    """
    eval_dict = pickle.loads(evaluation_results)
    data_dict = pickle.loads(pca_data)
    
    X = data_dict['X_pca']
    df = data_dict['df']
    
    print("\nPerforming anomaly detection...")
    
    anomaly_results = {
        'methods': {},
        'ensemble_predictions': None,
        'anomaly_scores': {}
    }
    
    predictions = []
    
    # Isolation Forest
    if 'isolation_forest' in methods:
        iso_forest = IsolationForest(contamination=contamination if contamination != 'auto' else 0.03,
                                    random_state=42)
        iso_predictions = iso_forest.fit_predict(X)
        iso_scores = iso_forest.score_samples(X)
        
        anomaly_results['methods']['isolation_forest'] = {
            'predictions': iso_predictions,
            'scores': iso_scores,
            'n_anomalies': (iso_predictions == -1).sum()
        }
        predictions.append(iso_predictions == -1)
        
        print(f"Isolation Forest: {(iso_predictions == -1).sum()} anomalies detected")
    
    # Local Outlier Factor
    if 'local_outlier_factor' in methods:
        lof = LocalOutlierFactor(contamination=contamination if contamination != 'auto' else 0.03)
        lof_predictions = lof.fit_predict(X)
        lof_scores = lof.negative_outlier_factor_
        
        anomaly_results['methods']['local_outlier_factor'] = {
            'predictions': lof_predictions,
            'scores': lof_scores,
            'n_anomalies': (lof_predictions == -1).sum()
        }
        predictions.append(lof_predictions == -1)
        
        print(f"Local Outlier Factor: {(lof_predictions == -1).sum()} anomalies detected")
    
    # One-Class SVM
    if 'one_class_svm' in methods:
        oc_svm = OneClassSVM(gamma='auto', nu=0.03)
        oc_predictions = oc_svm.fit_predict(X)
        oc_scores = oc_svm.decision_function(X)
        
        anomaly_results['methods']['one_class_svm'] = {
            'predictions': oc_predictions,
            'scores': oc_scores,
            'n_anomalies': (oc_predictions == -1).sum()
        }
        predictions.append(oc_predictions == -1)
        
        print(f"One-Class SVM: {(oc_predictions == -1).sum()} anomalies detected")
    
    # Ensemble predictions
    if len(predictions) > 1:
        if ensemble_method == 'voting':
            # Majority voting
            ensemble_pred = np.sum(predictions, axis=0) > len(predictions) / 2
        else:  # 'any'
            ensemble_pred = np.any(predictions, axis=0)
        
        anomaly_results['ensemble_predictions'] = ensemble_pred.astype(int)
        anomaly_results['n_anomalies_ensemble'] = ensemble_pred.sum()
        
        print(f"\nEnsemble ({ensemble_method}): {ensemble_pred.sum()} anomalies detected")
    else:
        anomaly_results['ensemble_predictions'] = predictions[0].astype(int) if predictions else np.zeros(len(X))
    
    # Compare with actual anomalies if available
    if 'is_anomaly' in df.columns:
        actual_anomalies = df['is_anomaly'].values[:len(anomaly_results['ensemble_predictions'])]
        
        # Calculate metrics
        tp = np.sum((anomaly_results['ensemble_predictions'] == 1) & (actual_anomalies == 1))
        fp = np.sum((anomaly_results['ensemble_predictions'] == 1) & (actual_anomalies == 0))
        tn = np.sum((anomaly_results['ensemble_predictions'] == 0) & (actual_anomalies == 0))
        fn = np.sum((anomaly_results['ensemble_predictions'] == 0) & (actual_anomalies == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        anomaly_results['performance'] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': {
                'tp': int(tp), 'fp': int(fp),
                'tn': int(tn), 'fn': int(fn)
            }
        }
        
        print(f"\nAnomaly Detection Performance:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
    
    anomaly_results['data'] = data_dict
    anomaly_results['evaluation'] = eval_dict
    
    return pickle.dumps(anomaly_results)

def generate_dashboards(evaluation_results, anomaly_results, 
                       dashboard_types=['energy_patterns_3d'], 
                       output_format='html', interactive=True):
    """
    Generate interactive dashboards for visualization
    """
    eval_dict = pickle.loads(evaluation_results)
    anomaly_dict = pickle.loads(anomaly_results)
    
    X = eval_dict['data']['X_pca']
    df = eval_dict['data']['df']
    labels = eval_dict['selected_labels']
    anomalies = anomaly_dict['ensemble_predictions']
    
    print("\nGenerating dashboards...")
    
    dashboards = {}
    
    # 3D Energy Patterns Visualization
    if 'energy_patterns_3d' in dashboard_types and X.shape[1] >= 3:
        fig = go.Figure()
        
        # Add cluster points
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            fig.add_trace(go.Scatter3d(
                x=X[mask, 0],
                y=X[mask, 1],
                z=X[mask, 2],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(
                    size=3,
                    opacity=0.6
                )
            ))
        
        # Highlight anomalies
        if anomalies is not None:
            anomaly_mask = anomalies == 1
            if anomaly_mask.any():
                fig.add_trace(go.Scatter3d(
                    x=X[anomaly_mask, 0],
                    y=X[anomaly_mask, 1],
                    z=X[anomaly_mask, 2],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(
                        size=6,
                        color='red',
                        symbol='x'
                    )
                ))
        
        fig.update_layout(
            title='3D Energy Consumption Patterns',
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3'
            ),
            height=600
        )
        
        fig.write_html(VIZ_DIR / 'energy_patterns_3d.html')
        dashboards['energy_patterns_3d'] = '3D visualization created'
    
    # Cluster Comparison Heatmap
    if 'cluster_comparison' in dashboard_types:
        # Calculate average features per cluster
        feature_cols = [col for col in df.columns if 'consumption' in col][:5]
        cluster_profiles = []
        
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            profile = df.iloc[mask][feature_cols].mean()
            cluster_profiles.append(profile)
        
        cluster_df = pd.DataFrame(cluster_profiles)
        cluster_df.index = [f'Cluster {i}' for i in range(len(cluster_profiles))]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cluster_df.values,
            x=cluster_df.columns,
            y=cluster_df.index,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title='Cluster Feature Profiles',
            xaxis_title='Features',
            yaxis_title='Clusters',
            height=500
        )
        
        fig.write_html(VIZ_DIR / 'cluster_comparison.html')
        dashboards['cluster_comparison'] = 'Cluster comparison created'
    
    # Temporal Heatmap
    if 'temporal_heatmap' in dashboard_types and 'hour' in df.columns and 'day_of_week' in df.columns:
        # Create hour vs day heatmap
        pivot_table = df.pivot_table(
            values='consumption_kwh',
            index='hour',
            columns='day_of_week',
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            y=list(range(24)),
            colorscale='RdYlBu_r'
        ))
        
        fig.update_layout(
            title='Energy Consumption Patterns by Hour and Day',
            xaxis_title='Day of Week',
            yaxis_title='Hour of Day',
            height=500
        )
        
        fig.write_html(VIZ_DIR / 'temporal_heatmap.html')
        dashboards['temporal_heatmap'] = 'Temporal heatmap created'
    
    print(f"Generated {len(dashboards)} dashboards")
    
    return pickle.dumps(dashboards)

def create_optimization_report(evaluation_results, anomaly_results, dashboard_results,
                              include_recommendations=True, cost_analysis=True,
                              sustainability_metrics=True, format='pdf'):
    """
    Create comprehensive optimization report with recommendations
    """
    eval_dict = pickle.loads(evaluation_results)
    anomaly_dict = pickle.loads(anomaly_results)
    dashboards = pickle.loads(dashboard_results)
    
    print("\nCreating optimization report...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'executive_summary': {},
        'technical_details': {},
        'recommendations': [],
        'metrics': {}
    }
    
    # Executive Summary
    report['executive_summary'] = {
        'best_clustering_model': eval_dict['best_model'],
        'number_of_clusters': eval_dict['models'][eval_dict['best_model']]['n_clusters'],
        'anomalies_detected': anomaly_dict.get('n_anomalies_ensemble', 0),
        'model_performance': {
            'silhouette_score': eval_dict['models'][eval_dict['best_model']]['scores'].get('silhouette', 0),
            'anomaly_precision': anomaly_dict.get('performance', {}).get('precision', 0)
        }
    }
    
    # Technical Details
    report['technical_details'] = {
        'models_evaluated': list(eval_dict['models'].keys()),
        'anomaly_methods': list(anomaly_dict['methods'].keys()),
        'feature_count': eval_dict['data']['n_components'],
        'total_samples': len(eval_dict['data']['X_pca'])
    }
    
    # Recommendations
    if include_recommendations:
        report['recommendations'] = [
            {
                'priority': 'High',
                'category': 'Energy Optimization',
                'recommendation': 'Implement targeted energy reduction strategies for Cluster 0 buildings',
                'potential_savings': '15-20%',
                'implementation_time': '3-6 months'
            },
            {
                'priority': 'High',
                'category': 'Anomaly Response',
                'recommendation': f'Investigate {anomaly_dict.get("n_anomalies_ensemble", 0)} detected anomalies for potential equipment issues',
                'potential_savings': '5-10%',
                'implementation_time': '1-2 weeks'
            },
            {
                'priority': 'Medium',
                'category': 'Load Balancing',
                'recommendation': 'Shift non-critical loads from peak hours to off-peak periods',
                'potential_savings': '10-15%',
                'implementation_time': '2-3 months'
            }
        ]
    
    # Cost Analysis
    if cost_analysis:
        # Simplified cost calculation
        avg_consumption = eval_dict['data']['df']['consumption_kwh'].mean()
        cost_per_kwh = 0.12  # $0.12 per kWh
        
        report['metrics']['cost_analysis'] = {
            'current_annual_cost': f"${avg_consumption * 24 * 365 * cost_per_kwh:,.2f}",
            'potential_savings': f"${avg_consumption * 24 * 365 * cost_per_kwh * 0.15:,.2f}",
            'roi_period': '8-12 months'
        }
    
    # Sustainability Metrics
    if sustainability_metrics:
        # CO2 calculations (0.92 lbs CO2 per kWh average)
        co2_per_kwh = 0.92
        current_co2 = avg_consumption * 24 * 365 * co2_per_kwh
        
        report['metrics']['sustainability'] = {
            'current_co2_emissions': f"{current_co2:,.0f} lbs/year",
            'potential_co2_reduction': f"{current_co2 * 0.15:,.0f} lbs/year",
            'equivalent_trees_planted': f"{int(current_co2 * 0.15 / 40)}"  # 40 lbs CO2 per tree/year
        }
    
    # Save report
    report_path = REPORT_DIR / f'optimization_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to: {report_path}")
    print("\n" + "="*60)
    print("OPTIMIZATION REPORT SUMMARY")
    print("="*60)
    print(f"Best Model: {report['executive_summary']['best_clustering_model']}")
    print(f"Clusters Found: {report['executive_summary']['number_of_clusters']}")
    print(f"Anomalies Detected: {report['executive_summary']['anomalies_detected']}")
    print(f"Potential Savings: 15-20% of energy consumption")
    print("="*60)
    
    return pickle.dumps(report)

def export_model_artifacts(evaluation_results, export_format=['pickle'], 
                          include_preprocessing=True, create_api_endpoint=True):
    """
    Export models for production deployment
    """
    eval_dict = pickle.loads(evaluation_results)
    
    print("\nExporting model artifacts...")
    
    exports = {
        'models_exported': [],
        'preprocessing_pipeline': None,
        'api_specification': None
    }
    
    # Export best model
    best_model_name = eval_dict['best_model']
    
    if 'pickle' in export_format:
        export_path = MODEL_DIR / f'{best_model_name.lower()}_production.pkl'
        with open(export_path, 'wb') as f:
            pickle.dump(eval_dict, f)
        exports['models_exported'].append(str(export_path))
    
    # Export preprocessing pipeline
    if include_preprocessing:
        preprocessing_pipeline = {
            'scaler': eval_dict['data'].get('scaler'),
            'pca_model': eval_dict['data'].get('pca_model'),
            'feature_names': eval_dict['data'].get('feature_names'),
            'n_components': eval_dict['data'].get('n_components')
        }
        
        pipeline_path = MODEL_DIR / 'preprocessing_pipeline.pkl'
        with open(pipeline_path, 'wb') as f:
            pickle.dump(preprocessing_pipeline, f)
        
        exports['preprocessing_pipeline'] = str(pipeline_path)
    
    # Create API endpoint specification
    if create_api_endpoint:
        api_spec = {
            'endpoint': '/predict',
            'method': 'POST',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'features': {
                        'type': 'array',
                        'items': {'type': 'number'},
                        'minItems': len(eval_dict['data'].get('feature_names', [])),
                        'maxItems': len(eval_dict['data'].get('feature_names', []))
                    }
                }
            },
            'output_schema': {
                'type': 'object',
                'properties': {
                    'cluster': {'type': 'integer'},
                    'is_anomaly': {'type': 'boolean'},
                    'confidence': {'type': 'number'}
                }
            }
        }
        
        api_path = MODEL_DIR / 'api_specification.json'
        with open(api_path, 'w') as f:
            json.dump(api_spec, f, indent=2)
        
        exports['api_specification'] = str(api_path)
    
    print(f"Exported {len(exports['models_exported'])} models")
    print(f"Preprocessing pipeline: {exports['preprocessing_pipeline']}")
    print(f"API specification: {exports['api_specification']}")
    
    return pickle.dumps(exports)

# Helper functions
def calculate_dunn_index(X, labels):
    """Calculate Dunn Index for cluster validation"""
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0
    
    # Calculate inter-cluster distances
    inter_distances = []
    for i in range(len(unique_labels)):
        for j in range(i+1, len(unique_labels)):
            cluster_i = X[labels == unique_labels[i]]
            cluster_j = X[labels == unique_labels[j]]
            distances = cdist(cluster_i, cluster_j)
            inter_distances.append(np.min(distances))
    
    # Calculate intra-cluster distances
    intra_distances = []
    for label in unique_labels:
        cluster = X[labels == label]
        if len(cluster) > 1:
            distances = cdist(cluster, cluster)
            # Get maximum distance within cluster
            intra_distances.append(np.max(distances))
    
    if not inter_distances or not intra_distances:
        return 0
    
    return np.min(inter_distances) / np.max(intra_distances)

def calculate_cluster_stability(X, model, n_iterations=10):
    """Calculate cluster stability using bootstrap"""
    n_samples = len(X)
    stability_scores = []
    
    original_labels = model.predict(X) if hasattr(model, 'predict') else model.labels_
    
    for _ in range(n_iterations):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_bootstrap = X[indices]
        
        # Refit model
        if hasattr(model, 'fit_predict'):
            bootstrap_labels = model.fit_predict(X_bootstrap)
        else:
            bootstrap_labels = model.fit(X_bootstrap).labels_
        
        # Calculate stability (simplified - using label consistency)
        # This is a simplified measure - in practice, use adjusted Rand index
        consistency = np.mean(original_labels[indices] == bootstrap_labels)
        stability_scores.append(consistency)
    
    return np.mean(stability_scores)

def create_comparison_visualization(evaluation):
    """Create model comparison visualization"""
    models = list(evaluation['models'].keys())
    metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        values = []
        for model in models:
            value = evaluation['models'][model]['scores'].get(metric, 0)
            values.append(value)
        
        axes[idx].bar(models, values)
        axes[idx].set_title(metric.replace('_', ' ').title())
        axes[idx].set_xlabel('Model')
        axes[idx].set_ylabel('Score')
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'model_comparison.png', dpi=150)
    plt.close()
