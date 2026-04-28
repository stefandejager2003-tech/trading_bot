import pandas as pd
import numpy as np
import faiss
import joblib
import os
from sklearn.preprocessing import StandardScaler

def build_gpu_indexes(processed_csv_path='data/processed/BTC_4H_GOLD.csv'):
    # 1. Load and Filter Data
    df = pd.read_csv(processed_csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # --- THE FILTER ---
    # We only train the brain on history up to end of 2023
    train_df = df[df['timestamp'] < '2024-01-01'].copy()
    print(f"Indexing {len(train_df)} rows of history (2020-2023)...")

    # Define your feature columns
    feature_cols = [
        'price_change_1', 'price_change_5', 'volume_change', 'ema_trend',
        'price_vs_ema', 'momentum_10', 'momentum_20', 'rsi_14', 
        'volatility_change', 'range_position', 'higher_highs', 'lower_lows',
        'volume_trend', 'pv_divergence' 
    ]
    
    # 2. Normalize the Data (FIT ONLY ON TRAIN DATA)
    scaler = StandardScaler()
    vectors = train_df[feature_cols].values.astype('float32')
    scaled_vectors = scaler.fit_transform(vectors)
    
    # Save the scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # 3. Separate by Trend using train_df
    long_mask = train_df['ema_trend'] == 1
    short_mask = train_df['ema_trend'] == -1
    
    # 4. Build FAISS-GPU Indexes
    res = faiss.StandardGpuResources()
    dimension = len(feature_cols)
    
    def create_index(data, filename):
        cpu_index = faiss.IndexFlatL2(dimension)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.add(data)
        
        final_index = faiss.index_gpu_to_cpu(gpu_index)
        faiss.write_index(final_index, f'models/{filename}')
        print(f"Saved {filename} with {data.shape[0]} vectors.")

    # Create indexes from the filtered, scaled vectors
    create_index(scaled_vectors[long_mask.values], 'btc_long.index')
    create_index(scaled_vectors[short_mask.values], 'btc_short.index')
    
    # 5. Save the Map (Outcomes)
    # Crucial: The simulator needs the train_df to map index results to past results
    train_df.to_csv('data/processed/train_outcomes.csv', index=False)
    
    # Also save the full df separately if needed for the simulator to "read" from
    df.to_csv('data/processed/full_data_map.csv', index=False)
    
    print("Indexer complete. Brain is locked to 2020-2023 history.")

if __name__ == "__main__":
    build_gpu_indexes()