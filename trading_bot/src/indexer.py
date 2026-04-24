import pandas as pd
import numpy as np
import faiss
import joblib
import os
from sklearn.preprocessing import StandardScaler

def build_gpu_indexes(processed_csv_path='data/processed/BTC_4H_GOLD.csv'):
    # 1. Load the Gold Data
    df = pd.read_csv(processed_csv_path)
    
    # Define your 17 feature columns (must match features.py exactly)
    feature_cols = [
        'price_change_1', 'price_change_5', 'volume_change', 'ema_trend',
        'price_vs_ema', 'momentum_10', 'momentum_20', 'rsi_14', 
        'volatility_change', 'range_position', 'higher_highs', 'lower_lows',
        'volume_trend', 'pv_divergence' 
        # Note: Add funding_z and oi_change here once merged
    ]
    
    # 2. Normalize the Data
    scaler = StandardScaler()
    vectors = df[feature_cols].values.astype('float32')
    scaled_vectors = scaler.fit_transform(vectors)
    
    # Save the scaler - the Engine NEEDS this for live trades
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # 3. Separate by Trend (The Directional Guardrail)
    # Long Index: Price > EMA | Short Index: Price < EMA
    long_mask = df['ema_trend'] == 1
    short_mask = df['ema_trend'] == -1
    
    # 4. Build FAISS-GPU Indexes
    res = faiss.StandardGpuResources()
    dimension = len(feature_cols)
    
    def create_index(data, filename):
        # IndexFlatL2 is perfect for ~7,000 vectors on a 3060 Ti
        cpu_index = faiss.IndexFlatL2(dimension)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.add(data)
        
        # Save back to CPU format for disk storage
        final_index = faiss.index_gpu_to_cpu(gpu_index)
        faiss.write_index(final_index, f'models/{filename}')
        print(f"Saved {filename} with {data.shape[0]} vectors.")

    create_index(scaled_vectors[long_mask], 'btc_long.index')
    create_index(scaled_vectors[short_mask], 'btc_short.index')
    
    # 5. Save the Map (Outcomes)
    # We save a copy of the DF so the Engine knows what happened after each vector
    df.to_csv('data/processed/outcomes.csv', index=False)
    print("Indexer complete. Scaler and Indexes ready.")

if __name__ == "__main__":
    build_gpu_indexes()