"""
Smart City Energy Data Generator
Generates synthetic energy consumption data for different city zones
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_smart_city_energy_data(n_records=10000, save_path='./'):
    """
    Generated synthetic smart city energy consumption data
    
    Features:
    - zone_id: City zone identifier (residential, commercial, industrial, mixed)
    - building_type: Type of building
    - timestamp: Time of measurement
    - energy_consumption: Energy usage in kWh
    - temperature: Outside temperature
    - occupancy_rate: Building occupancy percentage
    - solar_generation: Solar energy generated (if applicable)
    - is_weekend: Weekend indicator
    - hour_of_day: Hour when measurement was taken
    """
    
    np.random.seed(42)  # For reproducibility
    
    # Define zones and building types
    zones = ['Zone_A_Residential', 'Zone_B_Commercial', 'Zone_C_Industrial', 
             'Zone_D_Mixed', 'Zone_E_Tech_Park', 'Zone_F_Downtown']
    
    building_types = ['Office', 'Residential_High_Rise', 'Residential_Low_Rise', 
                      'Factory', 'Shopping_Mall', 'Hospital', 'School', 'Hotel']
    
    # Generate base data
    data = {
        'record_id': range(1, n_records + 1),
        'zone_id': np.random.choice(zones, n_records),
        'building_type': np.random.choice(building_types, n_records),
    }
    
    # Generate timestamps (last 30 days, hourly readings)
    base_date = datetime.now() - timedelta(days=30)
    timestamps = []
    for i in range(n_records):
        timestamps.append(base_date + timedelta(hours=i % 720))  # 720 hours = 30 days
    data['timestamp'] = timestamps
    
    # Extract time features
    data['hour_of_day'] = [ts.hour for ts in timestamps]
    data['day_of_week'] = [ts.weekday() for ts in timestamps]
    data['is_weekend'] = [1 if ts.weekday() >= 5 else 0 for ts in timestamps]
    data['month'] = [ts.month for ts in timestamps]
    
    # Generate temperature (seasonal variation)
    base_temp = 20  # Celsius
    data['temperature'] = base_temp + 10 * np.sin(np.arange(n_records) * 2 * np.pi / 720) + \
                         np.random.normal(0, 3, n_records)
    
    # Generate occupancy rate (varies by building type and time)
    occupancy = []
    for i in range(n_records):
        if data['building_type'][i] in ['Office', 'School']:
            # Lower occupancy on weekends, higher during work hours
            if data['is_weekend'][i]:
                occ = np.random.uniform(0.1, 0.3)
            elif 9 <= data['hour_of_day'][i] <= 17:
                occ = np.random.uniform(0.7, 0.95)
            else:
                occ = np.random.uniform(0.1, 0.4)
        elif data['building_type'][i].startswith('Residential'):
            # Inverse pattern for residential
            if 9 <= data['hour_of_day'][i] <= 17 and not data['is_weekend'][i]:
                occ = np.random.uniform(0.3, 0.6)
            else:
                occ = np.random.uniform(0.7, 0.95)
        else:
            # Random for other types
            occ = np.random.uniform(0.3, 0.9)
        occupancy.append(occ)
    data['occupancy_rate'] = occupancy
    
    # Generate base energy consumption
    base_consumption = {
        'Office': 150, 'Residential_High_Rise': 100, 'Residential_Low_Rise': 50,
        'Factory': 500, 'Shopping_Mall': 300, 'Hospital': 400, 'School': 200, 'Hotel': 250
    }
    
    energy = []
    for i in range(n_records):
        base = base_consumption[data['building_type'][i]]
        
        # Factors affecting consumption
        temp_factor = 1 + abs(data['temperature'][i] - 20) * 0.02  # More AC/heating
        occupancy_factor = 0.5 + data['occupancy_rate'][i] * 0.5
        time_factor = 1.0
        
        # Peak hours
        if data['building_type'][i] in ['Office', 'School'] and 9 <= data['hour_of_day'][i] <= 17:
            time_factor = 1.3
        elif data['building_type'][i].startswith('Residential') and (data['hour_of_day'][i] < 9 or data['hour_of_day'][i] > 18):
            time_factor = 1.2
        
        consumption = base * temp_factor * occupancy_factor * time_factor
        # Add some noise
        consumption += np.random.normal(0, base * 0.1)
        energy.append(max(consumption, 10))  # Minimum consumption
    
    data['energy_consumption'] = energy
    
    # Generate solar generation (some buildings have solar panels)
    solar = []
    for i in range(n_records):
        if np.random.random() < 0.3:  # 30% buildings have solar
            # Solar generation peaks at noon
            solar_peak = 50 if data['building_type'][i] != 'Factory' else 200
            hour = data['hour_of_day'][i]
            if 6 <= hour <= 18:
                generation = solar_peak * np.sin((hour - 6) * np.pi / 12)
                generation *= (1 - 0.3 * data['is_weekend'][i])  # Less on weekends
                generation += np.random.normal(0, 5)
                solar.append(max(generation, 0))
            else:
                solar.append(0)
        else:
            solar.append(0)
    data['solar_generation'] = solar
    
    # Calculate net consumption
    data['net_consumption'] = [data['energy_consumption'][i] - data['solar_generation'][i] 
                               for i in range(n_records)]
    
    # Add some anomalies (5% of data)
    anomaly_indices = np.random.choice(n_records, int(n_records * 0.05), replace=False)
    for idx in anomaly_indices:
        # Create different types of anomalies
        anomaly_type = np.random.choice(['spike', 'drop', 'unusual_pattern'])
        if anomaly_type == 'spike':
            data['energy_consumption'][idx] *= np.random.uniform(2, 4)
        elif anomaly_type == 'drop':
            data['energy_consumption'][idx] *= np.random.uniform(0.1, 0.3)
        else:
            # Unusual pattern (high consumption at odd hours)
            if data['hour_of_day'][idx] in [2, 3, 4]:
                data['energy_consumption'][idx] *= 2.5
    
    # Add is_anomaly flag for validation
    data['is_anomaly'] = [1 if i in anomaly_indices else 0 for i in range(n_records)]
    
    # Calculate energy efficiency score
    efficiency = []
    for i in range(n_records):
        expected = base_consumption[data['building_type'][i]] * data['occupancy_rate'][i]
        actual = data['net_consumption'][i]
        eff_score = 100 * (1 - min(actual / (expected + 1), 2))  # Normalize to 0-100
        efficiency.append(max(min(eff_score, 100), 0))
    data['efficiency_score'] = efficiency
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create train and test datasets
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    # Save to CSV
    os.makedirs(save_path, exist_ok=True)
    train_df.to_csv(os.path.join(save_path, 'smart_city_energy.csv'), index=False)
    test_df.to_csv(os.path.join(save_path, 'test_energy.csv'), index=False)
    
    print(f"✅ Generated {n_records} records of smart city energy data")
    print(f"   - Training set: {len(train_df)} records")
    print(f"   - Test set: {len(test_df)} records")
    print(f"   - Features: {list(df.columns)}")
    print(f"   - Anomalies: {data['is_anomaly'].count(1)} records")
    print(f"   - Zones: {len(set(data['zone_id']))}")
    print(f"   - Building types: {len(set(data['building_type']))}")
    
    return train_df, test_df

if __name__ == "__main__":
    # Generate the dataset
    train, test = generate_smart_city_energy_data(10000)
    print("\n📊 Sample of generated data:")
    print(train.head())
    print("\n📈 Data statistics:")
    print(train.describe())