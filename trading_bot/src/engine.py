import faiss
import numpy as np
import pandas as pd
import joblib # To load your scaler

class StrategyEngine:
    def __init__(self, index_path, scaler_path, outcomes_df_path):
        # 1. Load the Brain
        # Use GPU resources for your 3060 Ti
        self.res = faiss.StandardGpuResources()
        self.index = faiss.read_index(index_path)
        self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
        
        # 2. Load Normalization & Outcomes
        self.scaler = joblib.load(scaler_path)
        self.outcomes = pd.read_csv(outcomes_df_path)

    def get_market_matches(self, current_features, k=50):
        """
        Performs the Dual-Metric search.
        """
        # Normalize the live data exactly like the historical data
        scaled_vector = self.scaler.transform([current_features]).astype('float32')
        
        # Search 1: Initial Euclidean/Cosine distance on GPU
        # (L2 distance is standard for IndexFlatL2)
        distances, indices = self.gpu_index.search(scaled_vector, k)
        
        # Search 2: Filter/Score the outcomes
        match_results = self.outcomes.iloc[indices[0]].copy()
        match_results['similarity_score'] = distances[0]
        
        return match_results

    def calculate_expectancy(self, matches):
        """
        The 'Is it worth it?' check.
        """
        win_rate = (matches['fwd_max_up'] > abs(matches['fwd_max_down'])).mean()
        avg_profit = matches['fwd_max_up'].mean()
        avg_drawdown = matches['fwd_max_down'].mean()
        
        # Expectancy = (Probability of Win * Avg Win) - (Probability of Loss * Avg Loss)
        expectancy = (win_rate * avg_profit) + ((1 - win_rate) * avg_drawdown)
        
        return {
            'expectancy': expectancy,
            'win_rate': win_rate,
            'avg_rr': abs(avg_profit / avg_drawdown) if avg_drawdown != 0 else 0
        }

# --- Example Usage ---
# engine = StrategyEngine('models/btc_long.index', 'models/scaler.pkl', 'data/processed/outcomes.csv')
# matches = engine.get_market_matches(latest_17_features)
# signal = engine.calculate_expectancy(matches)