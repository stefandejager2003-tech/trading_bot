import ccxt
import pandas as pd
import time
import os
from datetime import datetime, timedelta

def fetch_bybit_history(symbol='BTC/USDT:USDT', timeframe='1m', years=3):
    # 1. Setup the Connection
    exchange = ccxt.bybit({
        'enableRateLimit': True,
        'options': {'defaultType': 'linear'} # Crucial for USDT Perpetuals
    })

    # 2. Determine Timeframe Boundaries
    now = datetime.now()
    start_time = now - timedelta(days=years * 365)
    since = int(start_time.timestamp() * 1000)
    
    all_ohlcv = []
    filename = f"data/raw/{symbol.replace('/', '_').replace(':', '_')}_{timeframe}.csv"
    
    # Ensure the directory exists
    os.makedirs('data/raw', exist_ok=True)

    print(f"--- Starting Download: {symbol} ---")
    print(f"Target Start: {start_time}")

    while since < int(now.timestamp() * 1000):
        try:
            # Bybit allows up to 1000 candles per request for 1m timeframe
            candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            
            if not candles or len(candles) == 0:
                print("\nReached the end of available data.")
                break
                
            all_ohlcv.extend(candles)
            
            # Move 'since' forward to the timestamp of the last candle + 1 unit
            since = candles[-1][0] + 60000 
            
            # Progress reporting
            current_date = datetime.fromtimestamp(candles[-1][0] / 1000)
            print(f"Progress: {current_date} | Total Rows: {len(all_ohlcv)}", end='\r')
            
            # Small sleep to respect the rate limiter
            time.sleep(exchange.rateLimit / 1000)
            
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            print(f"\nConnection issue: {e}. Retrying in 10 seconds...")
            time.sleep(10)
            continue

    # 3. Save to Disk
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.to_csv(filename, index=False)
    
    print(f"\n--- Download Complete ---")
    print(f"File Saved: {filename}")
    return df

if __name__ == "__main__":
    fetch_bybit_history()