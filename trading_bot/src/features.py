import pandas as pd
import pandas_ta as ta
import numpy as np

def calculate_features(df_1m):
    """
    Transforms raw 1m data into 4h feature vectors with 17 indicators.
    """
    # 1. Resample to 4H
    # We use 'label=right' so the timestamp represents the END of the 4h period
    df = df_1m.resample('4H', on='timestamp').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # --- Trend & Momentum (Features 1-7) ---
    df['ema_200'] = ta.ema(df['close'], length=200)
    df['ema_trend'] = np.where(df['close'] > df['ema_200'], 1, -1)
    df['price_vs_ema'] = (df['close'] - df['ema_200']) / df['ema_200']
    
    df['price_change_1'] = df['close'].pct_change(1)
    df['price_change_5'] = df['close'].pct_change(5)
    df['momentum_10'] = ta.mom(df['close'], length=10)
    df['momentum_20'] = ta.mom(df['close'], length=20)
    
    # --- Oscillators & Volatility (Features 8-11) ---
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['volatility_change'] = df['atr'].pct_change(5)
    # Range Position: Where is price relative to the recent High/Low? (0 to 1)
    df['range_position'] = (df['close'] - df['low'].rolling(10).min()) / (df['high'].rolling(10).max() - df['low'].rolling(10).min())

    # --- Volume & Structure (Features 12-15) ---
    df['volume_change'] = df['volume'].pct_change(1)
    df['volume_trend'] = ta.sma(df['volume'], length=10).pct_change(1)
    
    # Higher Highs / Lower Lows (Lookback 3)
    df['higher_highs'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_lows'] = (df['low'] < df['low'].shift(1)).astype(int)
    
    # Price-Volume Divergence
    df['pv_divergence'] = np.where((df['price_change_1'] > 0) & (df['volume_change'] < 0), 1, 0)

    # --- Outcome Labeling (The Target) ---
    # Look ahead 12 bars (48 hours) to find the best/worst case scenario
    df['fwd_max_up'] = df['high'].shift(-12).rolling(window=12).max() / df['close'] - 1
    df['fwd_max_down'] = df['low'].shift(-12).rolling(window=12).min() / df['close'] - 1
    
    # Drop rows where we don't have enough data for indicators or outcomes
    return df.dropna()

if __name__ == "__main__":
    # Test run
    raw_data = pd.read_csv('data/raw/BTC_USDT_USDT_1m_3yr.csv', parse_dates=['timestamp'])
    gold_data = calculate_features(raw_data)
    gold_data.to_csv('data/processed/BTC_4H_GOLD.csv')
    print(f"Features created. Shape: {gold_data.shape}")