import ccxt
import pandas as pd
import time
import os
from datetime import datetime, timedelta

def fetch_bybit_history(symbol='BTC/USDT:USDT', timeframe='1m', years=3):
    exchange = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'linear'}})
    
    now = datetime.now()
    # If the API hits a wall, it will start from the earliest available date automatically
    start_time = now - timedelta(days=years * 365)
    since = int(start_time.timestamp() * 1000)
    
    filename = f"data/raw/{symbol.replace('/', '_').replace(':', '_')}_{timeframe}.csv"
    os.makedirs('data/raw', exist_ok=True)

    print(f"--- Starting Download: {symbol} ---")
    
    # Header for the CSV (first run only)
    first_run = True

    while since < int(now.timestamp() * 1000):
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            
            if not candles:
                print("\nNo more data returned from API.")
                break
                
            df_chunk = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'], unit='ms')
            
            # Append to CSV immediately
            df_chunk.to_csv(filename, mode='a', index=False, header=first_run)
            first_run = False

            since = candles[-1][0] + 60000 
            
            print(f"Progress: {df_chunk['timestamp'].iloc[-1]}", end='\r')
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"\nError: {e}. Retrying...")
            time.sleep(10)
            continue

    print(f"\n--- Download Complete: {filename} ---")

if __name__ == "__main__":
    fetch_bybit_history()