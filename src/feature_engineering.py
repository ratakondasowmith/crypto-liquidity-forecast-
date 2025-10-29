import pandas as pd
import numpy as np

def create_features(df, forecast_horizon=1):
    print("🔹 Creating features...")

    price_col = next((c for c in df.columns if 'price' in c.lower()), None)
    vol_col = next((c for c in df.columns if 'volume' in c.lower()), None)
    mc_col  = next((c for c in df.columns if 'market' in c.lower() and 'cap' in c.lower()), None)

    if price_col is None:
        price_col = df.select_dtypes(include=[np.number]).columns[0]

    if vol_col is None:
        df['volume'] = df[price_col] * 0.1
        vol_col = 'volume'
    if mc_col is None:
        df['market_cap'] = df[price_col] * df[vol_col]
        mc_col = 'market_cap'

    df['price_ma_7'] = df[price_col].rolling(7, 1).mean()
    df['price_ma_30'] = df[price_col].rolling(30, 1).mean()
    df['volatility_7'] = df[price_col].rolling(7, 1).std().fillna(0)
    df['volatility_30'] = df[price_col].rolling(30, 1).std().fillna(0)
    df['liquidity_ratio'] = df[vol_col] / (df[mc_col].replace(0, np.nan) + 1e-9)
    df['liquidity_ratio'] = df['liquidity_ratio'].fillna(0)
    df['liq_ratio_lag_1'] = df['liquidity_ratio'].shift(1).fillna(method='bfill')
    df['liq_ratio_diff_1'] = df['liquidity_ratio'] - df['liq_ratio_lag_1']

    df['liquidity'] = df['liquidity_ratio'].rolling(3, 1).mean()
    df['target_liquidity'] = df['liquidity'].shift(-forecast_horizon)
    df = df.dropna(subset=['target_liquidity']).fillna(0)

    print("✅ Feature engineering completed.")
    return df
