"""
Smart City Energy Analysis Module
lab.py with multiple clustering algorithms
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from kneed import KneeLocator
import json
import os
import warnings
warnings.filterwarnings('ignore')



def load_data():
    """Load energy consumption data - compatible with original DAG"""
    try:
        print("📂 Loading smart city energy data...")
        data_path = '/opt/airflow/dags/data/file.csv'
        
        # Check if our smart city data exists, otherwise generate it
        if not os.path.exists(data_path):
            print("Generating smart city energy data...")
            generate_smart_city_data()
        
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} records with {len(df.columns)} features")
        
        return pickle.dumps(df)
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        raise

def data_preprocessing(data_bytes):
    """Enhanced preprocessing - compatible with original DAG"""
    try:
        print("\n🔧 Starting advanced data preprocessing...")
        df = pickle.loads(data_bytes)
        
        # Select numerical features for clustering
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove ID columns if present
        id_cols = [col for col in numerical_cols if 'id' in col.lower() or 'index' in col.lower()]
        feature_cols = [col for col in numerical_cols if col not in id_cols]
        
        if len(feature_cols) < 2:
            # If not enough features, create synthetic features
            print("Creating synthetic features for clustering...")
            df = create_energy_features(df)
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        X = df[feature_cols].fillna(df[feature_cols].mean())
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save scaler
        os.makedirs('/opt/airflow/dags/model', exist_ok=True)
        with open('/opt/airflow/dags/model/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"✅ Preprocessing complete. Shape: {X_scaled.shape}")
        
        return pickle.dumps({'X_scaled': X_scaled, 'df': df, 'features': list(X.columns)})
    
    except Exception as e:
        print(f"❌ Error in preprocessing: {str(e)}")
        raise

def build_save_model(preprocessed_data_bytes, model_filename):
    """Build and save clustering models - compatible with original DAG"""
    try:
        print("\n🚀 Building multiple clustering models...")
        data_dict = pickle.loads(preprocessed_data_bytes)
        X_scaled = data_dict['X_scaled']
        
        # Apply PCA if too many features
        if X_scaled.shape[1] > 10:
            pca = PCA(n_components=min(10, X_scaled.shape[1]), random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            print(f"📊 PCA: {X_scaled.shape[1]} → {X_pca.shape[1]} dimensions")
            with open('/opt/airflow/dags/model/pca_transformer.pkl', 'wb') as f:
                pickle.dump(pca, f)
        else:
            X_pca = X_scaled
        
        results = {}
        
        # 1. K-Means with Elbow Method (Original functionality)
        print("\n🔵 K-Means Clustering with Elbow Method...")
        sse_values = []
        silhouette_scores = []
        k_range = range(2, min(11, len(X_pca)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_pca)
            sse_values.append(kmeans.inertia_)
            
            if k < len(X_pca):
                sil_score = silhouette_score(X_pca, kmeans.labels_)
                silhouette_scores.append(sil_score)
                print(f"   K={k}: SSE={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")
        
        # Find optimal k
        kn = KneeLocator(list(k_range), sse_values, curve='convex', direction='decreasing')
        optimal_k = kn.elbow if kn.elbow else 4
        
        # Train final model
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_labels = kmeans_final.fit_predict(X_pca)
        
        # Save model (using the filename from original DAG)
        model_path = f'/opt/airflow/dags/model/{model_filename}'
        with open(model_path, 'wb') as f:
            pickle.dump(kmeans_final, f)
        
        results['kmeans'] = {
            'algorithm': 'K-Means',
            'n_clusters': optimal_k,
            'silhouette_score': silhouette_score(X_pca, kmeans_labels),
            'sse_values': sse_values
        }
        
        # 2. DBSCAN (Enhancement)
        print("\n🟢 DBSCAN Clustering...")
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_pca)
        
        n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)
        
        with open('/opt/airflow/dags/model/dbscan_model.pkl', 'wb') as f:
            pickle.dump(dbscan, f)
        
        results['dbscan'] = {
            'algorithm': 'DBSCAN',
            'n_clusters': n_clusters_dbscan,
            'n_noise_points': n_noise
        }
        
        # 3. Hierarchical (Enhancement)
        print("\n🟡 Hierarchical Clustering...")
        hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
        hierarchical_labels = hierarchical.fit_predict(X_pca)
        
        results['hierarchical'] = {
            'algorithm': 'Hierarchical',
            'n_clusters': optimal_k,
            'silhouette_score': silhouette_score(X_pca, hierarchical_labels)
        }
        
        # 4. Anomaly Detection (Enhancement)
        print("\n🔍 Anomaly Detection with Isolation Forest...")
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        anomaly_predictions = iso_forest.fit_predict(X_pca)
        
        with open('/opt/airflow/dags/model/isolation_forest.pkl', 'wb') as f:
            pickle.dump(iso_forest, f)
        
        n_anomalies = list(anomaly_predictions).count(-1)
        results['anomaly_detection'] = {
            'total_anomalies': n_anomalies,
            'anomaly_percentage': (n_anomalies / len(X_pca)) * 100
        }
        
        # Save results
        os.makedirs('/opt/airflow/dags/results', exist_ok=True)
        with open('/opt/airflow/dags/results/clustering_comparison.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\n📊 Model Comparison Summary:")
        print("=" * 50)
        for algo, metrics in results.items():
            print(f"{algo}: {metrics.get('n_clusters', 'N/A')} clusters")
        
        # Return SSE values for compatibility with original DAG
        return pickle.dumps(sse_values)
    
    except Exception as e:
        print(f"❌ Error building models: {str(e)}")
        raise

def load_model_elbow(model_filename, sse_values_bytes):
    """Load model and determine optimal clusters - enhanced version"""
    try:
        print("\n📈 Loading model and analyzing results...")
        sse_values = pickle.loads(sse_values_bytes)
        
        # Load the saved model
        model_path = f'/opt/airflow/dags/model/{model_filename}'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Find optimal number of clusters using elbow
        k_range = range(2, len(sse_values) + 2)
        kn = KneeLocator(list(k_range), sse_values, curve='convex', direction='decreasing')
        optimal_k = kn.elbow if kn.elbow else 4
        
        # Load and display all results
        try:
            with open('/opt/airflow/dags/results/clustering_comparison.json', 'r') as f:
                results = json.load(f)
            
            print("\n" + "=" * 60)
            print("🏆 SMART CITY ENERGY ANALYSIS COMPLETE!")
            print("=" * 60)
            print(f"📊 Optimal Clusters (Elbow Method): {optimal_k}")
            print("\n📈 Algorithm Comparison:")
            for algo, metrics in results.items():
                if algo != 'anomaly_detection':
                    print(f"  {algo}: {metrics.get('n_clusters', 'N/A')} clusters")
                else:
                    print(f"\n🔍 Anomaly Detection:")
                    print(f"  Total Anomalies: {metrics.get('total_anomalies', 0)}")
                    print(f"  Percentage: {metrics.get('anomaly_percentage', 0):.2f}%")
        except:
            pass
        
        result_message = f"""
        ✅ Analysis Complete!
        
        Optimal number of clusters: {optimal_k}
        Model saved at: {model_path}
        
        Check /opt/airflow/dags/results/ for detailed reports:
        - clustering_comparison.json
        - Additional models in /opt/airflow/dags/model/
        """
        
        print(result_message)
        return result_message
    
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise

# ============= Helper Functions =============

def generate_smart_city_data():
    """Generate synthetic smart city energy data"""
    np.random.seed(42)
    n_records = 10000
    
    # Generate synthetic features
    data = {
        'building_id': range(n_records),
        'energy_consumption': np.random.normal(250, 100, n_records),
        'temperature': np.random.normal(20, 5, n_records),
        'occupancy_rate': np.random.uniform(0.3, 0.95, n_records),
        'hour_of_day': np.random.randint(0, 24, n_records),
        'is_weekend': np.random.choice([0, 1], n_records),
        'solar_generation': np.random.uniform(0, 50, n_records),
        'building_age': np.random.randint(1, 50, n_records),
        'floor_area': np.random.normal(5000, 2000, n_records)
    }
    
    # Calculate derived features
    data['net_consumption'] = data['energy_consumption'] - data['solar_generation']
    data['efficiency_score'] = 100 * (1 - data['net_consumption'] / (data['energy_consumption'] + 1))
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_records, int(n_records * 0.05), replace=False)
    for idx in anomaly_indices:
        data['energy_consumption'][idx] *= np.random.uniform(2, 4)
    
    df = pd.DataFrame(data)
    
    # Save as file.csv to work with original DAG
    df.to_csv('/opt/airflow/dags/data/file.csv', index=False)
    
    # Also save test data
    test_size = int(0.2 * len(df))
    test_df = df.iloc[-test_size:]
    test_df.to_csv('/opt/airflow/dags/data/test.csv', index=False)
    
    print(f"✅ Generated {n_records} records of smart city energy data")

def create_energy_features(df):
    """Create energy-related features if original data lacks them"""
    # Add synthetic energy features if not present
    if 'energy_consumption' not in df.columns:
        df['energy_consumption'] = np.random.normal(250, 100, len(df))
    if 'temperature' not in df.columns:
        df['temperature'] = np.random.normal(20, 5, len(df))
    if 'occupancy_rate' not in df.columns:
        df['occupancy_rate'] = np.random.uniform(0.3, 0.95, len(df))
    if 'efficiency_score' not in df.columns:
        df['efficiency_score'] = np.random.uniform(50, 100, len(df))
    
    return df
