import pandas as pd
import numpy as np

def add_tick_microstructure_features(df):
    """
    Calculates high-fidelity microstructure features from raw L1 tick data.
    Input df is expected to be the raw tick dataframe before volume bar aggregation.
    """
    if df is None: return None
    
    print("Calculating Tick-Level Microstructure (OFI/Spread)...")
    
    # 1. Mid-Price & Spread
    df['mid'] = (df['pricebid'] + df['priceask']) / 2
    df['spread'] = df['priceask'] - df['pricebid']
    
    # 2. Order Flow Imbalance (OFI) - Cont & Lipton (2012)
    # OFI = Delta_Bid_Size (if P_bid >= P_bid_prev) - Delta_Ask_Size (if P_ask <= P_ask_prev)
    
    bid_p = df['pricebid']
    bid_s = df['sizebid']
    ask_p = df['priceask']
    ask_s = df['sizeask']
    
    bid_p_prev = bid_p.shift(1)
    bid_s_prev = bid_s.shift(1)
    ask_p_prev = ask_p.shift(1)
    ask_s_prev = ask_s.shift(1)
    
    # Bid contribution
    bid_contrib = pd.Series(0, index=df.index)
    bid_contrib.loc[bid_p > bid_p_prev] = bid_s.loc[bid_p > bid_p_prev]
    bid_contrib.loc[bid_p == bid_p_prev] = bid_s.loc[bid_p == bid_p_prev] - bid_s_prev.loc[bid_p == bid_p_prev]
    bid_contrib.loc[bid_p < bid_p_prev] = -bid_s_prev.loc[bid_p < bid_p_prev]
    
    # Ask contribution
    ask_contrib = pd.Series(0, index=df.index)
    ask_contrib.loc[ask_p < ask_p_prev] = ask_s.loc[ask_p < ask_p_prev]
    ask_contrib.loc[ask_p == ask_p_prev] = ask_s.loc[ask_p == ask_p_prev] - ask_s_prev.loc[ask_p == ask_p_prev]
    ask_contrib.loc[ask_p > ask_p_prev] = -ask_s_prev.loc[ask_p > ask_p_prev]
    
    df['ofi'] = bid_contrib - ask_contrib
    
    return df

def aggregate_tick_features_to_bars(tick_df, bar_indices):
    """
    Aggregates tick-level features (OFI, Spread) into volume bar buckets.
    bar_indices is a series matching tick_df length indicating which bar each tick belongs to.
    """
    print("Aggregating Tick Features to Volume Bars...")
    
    agg_funcs = {
        'ofi': 'sum',
        'spread': 'mean',
        'mid': ['std', 'first', 'last'] # For volatility and returns
    }
    
    grouped = tick_df.groupby(bar_indices).agg(agg_funcs)
    
    # Flatten columns
    grouped.columns = [f'tick_{c[0]}_{c[1]}' for c in grouped.columns]
    
    # Add Tick Volatility (Std of mid price within bar)
    grouped.rename(columns={'tick_mid_std': 'tick_volatility'}, inplace=True)
    
    return grouped
