import time
import os
from datetime import datetime
from dotenv import load_dotenv
from src.downloader import fetch_bybit_history
from src.features import calculate_features
from src.engine import StrategyEngine

# Load API Keys
load_dotenv()

def run_trading_cycle():
    print(f"\n--- Cycle Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    
    # 1. Sync Data
    # We fetch the last few days to ensure indicators like EMA 200 are accurate
    raw_df = fetch_bybit_history(years=0.05) # Fetch ~2 weeks of fresh data
    
    # 2. Vectorize
    # This turns the last closed 4h candle into your 17-feature fingerprint
    gold_df = calculate_features(raw_df)
    latest_state = gold_df.iloc[-1]
    
    # 3. Apply the Trend Filter (Your Directional Guardrail)
    regime = "LONG" if latest_state['ema_trend'] == 1 else "SHORT"
    print(f"Current Market Regime: {regime}")
    
    # 4. Search the Brain
    index_path = f"models/btc_{regime.lower()}.index"
    engine = StrategyEngine(index_path, 'models/scaler.pkl', 'data/processed/outcomes.csv')
    
    current_vector = latest_state.drop(['fwd_max_up', 'fwd_max_down']).values
    matches = engine.get_market_matches(current_vector)
    
    # 5. The Verdict
    stats = engine.calculate_expectancy(matches)
    
    print(f"Match Stats: WinRate {stats['win_rate']:.2%}, Avg R:R {stats['avg_rr']:.2f}")
    
    if stats['win_rate'] > 0.65 and stats['expectancy'] > 0.01:
        print(">>> SIGNAL DETECTED: Sending order to Bybit...")
        # execute_trade(regime, latest_state['close'])
    else:
        print(">>> No high-probability match found. Standing by.")

def main():
    print("RTX 3060 Ti Sniper Bot Online.")
    while True:
        # Calculate time until the next 4H candle (00:00, 04:00, 08:00, etc.)
        now = datetime.now()
        seconds_until_next_4h = (4 - (now.hour % 4)) * 3600 - (now.minute * 60) - now.second
        
        # If we are within 10 seconds of a new candle, run the cycle
        if seconds_until_next_4h <= 10:
            run_trading_cycle()
            time.sleep(60) # Wait a minute so we don't trigger twice
        else:
            # Sleep until 5 seconds before the next candle
            sleep_time = max(seconds_until_next_4h - 5, 1)
            print(f"Sleeping for {sleep_time/3600:.2f} hours until next candle close...", end='\r')
            time.sleep(sleep_time)

if __name__ == "__main__":
    main()