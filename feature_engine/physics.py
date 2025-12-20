import numpy as np
import pandas as pd
from scipy.stats import linregress

# --- Math / Helper Functions (from physics_features.py) ---

def get_weights(d, size):
    """
    Calculates weights for fractional differentiation.
    w_k = -w_{k-1} * (d - k + 1) / k
    """
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * (d - k + 1) / k
        w.append(w_k)
    return np.array(w[::-1]) # Reverse for convolution

def frac_diff_ffd(series, d=0.4, thres=1e-4):
    """
    Fractional Differentiation with Fixed Window (FFD).
    Preserves memory while making series stationary.
    """
    # 1. Determine window size where weights drop below threshold
    w_test = [1.0]
    k = 1
    while abs(w_test[-1]) > thres:
        w_k = -w_test[-1] * (d - k + 1) / k
        w_test.append(w_k)
        k += 1
    
    width = len(w_test)
    weights = np.array(w_test[::-1]) # Weights for dot product
    
    # 2. Apply weights via rolling window
    # Valid only after 'width' data points
    df = pd.Series(series).dropna()
    res = df.rolling(window=width).apply(lambda x: np.dot(x, weights), raw=True)
    
    return res

def get_hurst_exponent(series, window=100, lags=[2, 4, 8, 16, 32, 64]):
    """
    Estimates Hurst Exponent using variance of differences at multiple lags.
    H = 0.5 (Random), >0.5 (Trend), <0.5 (Mean Revert).
    
    This is a rolling simplified version suitable for feature generation.
    """
    # We calculate the std dev of returns at different lags
    # log(std) ~ H * log(lag)
    
    def _calc_hurst(x):
        if len(x) < max(lags) + 1: return np.nan
        
        y_vals = []
        x_vals = np.log(lags)
        
        for lag in lags:
            # Std dev of price differences (returns) at this lag
            res = x - np.roll(x, lag)
            res = res[lag:] # remove nan
            if len(res) == 0: return np.nan
            y_vals.append(np.std(res))
        
        y_vals = np.log(y_vals)
        
        # Slope of log-log plot is H
        try:
            slope, _, _, _, _ = linregress(x_vals, y_vals)
            return slope
        except:
            return np.nan

    # Rolling application
    return series.rolling(window=window).apply(_calc_hurst, raw=True)

def get_efficiency_ratio(series, window=100):
    """
    Kaufman Efficiency Ratio (Fractal Efficiency).
    ER = Displacement / Path Length
    """
    # Displacement: Change from t-n to t
    change = series.diff(window).abs()
    
    # Path Length: Sum of absolute tick-to-tick changes
    # volatility = sum(|price[i] - price[i-1]|)
    path = series.diff().abs().rolling(window).sum()
    
    return change / path

def get_shannon_entropy(series, window=100, bins=20):
    """
    Calculates Rolling Shannon Entropy (Information Content).
    Higher Entropy = More Disorder/Noise. Lower Entropy = More Structure/Trend.
    
    H(X) = -sum(p(x) * log2(p(x)))
    """
    def _calc_entropy(x):
        # Discretize data into bins to estimate probability distribution
        counts, _ = np.histogram(x, bins=bins, density=False)
        # Probabilities
        p = counts / counts.sum()
        # Remove zeros to avoid log error
        p = p[p > 0]
        # Entropy
        return -np.sum(p * np.log2(p))

    # Apply on rolling log-returns (or stationarized series)
    return series.rolling(window=window).apply(_calc_entropy, raw=True)

def calc_yang_zhang_volatility(df, window=30):
    """
    Yang-Zhang Volatility Estimator.
    Considers Open, Close, High, Low and Drift.
    Minimum variance estimator among open-close estimators.
    """
    # Requires: open, close, high, low, time_start/prev_close
    # We assume 'open' captures the jump from previous close if bars are continuous.
    # In volume bars, O_t is usually C_{t-1}, but not always if gaps exist.
    
    # 1. Overnight Volatility (Close to Open)
    # o = log(O_t / C_{t-1})
    # Protect against div by zero or log(0)
    log_ho = np.log(np.maximum(df['high'], 1e-9) / np.maximum(df['open'], 1e-9))
    log_lo = np.log(np.maximum(df['low'], 1e-9) / np.maximum(df['open'], 1e-9))
    log_co = np.log(np.maximum(df['close'], 1e-9) / np.maximum(df['open'], 1e-9))
    
    # We need Close_{t-1}.
    log_oc = np.log(np.maximum(df['open'], 1e-9) / np.maximum(df['close'].shift(1), 1e-9))
    
    # Rogers-Satchell Volatility
    rs_var = (log_ho * (log_ho - log_co)) + (log_lo * (log_lo - log_co))
    
    # Open-Close Volatility
    oc_var = log_oc ** 2
    
    # Close-Open Volatility (Drift)
    co_var = log_co ** 2

    # k constant for weighting. Yang-Zhang recommends k=0.34 / (1.34 + (n+1)/(n-1))
    # Approximation: k = 0.34
    k = 0.34
    
    # Rolling Variances
    # We need rolling mean of variances
    roll_rs = rs_var.rolling(window).mean()
    roll_oc = oc_var.rolling(window).mean()
    roll_co = co_var.rolling(window).mean()
    
    # YZ Variance = V_o + k*V_c + (1-k)*V_rs
    # V_o ~ Variance of overnight (Open - PrevClose)
    # V_c ~ Variance of open-to-close (Close - Open)
    
    # However, standard formula uses sample variances of the log returns.
    # Let's use the definition:
    # sigma^2 = sigma_o^2 + k * sigma_c^2 + (1-k) * sigma_rs^2
    
    # sigma_o^2: Variance of (Open - PrevClose)
    # sigma_c^2: Variance of (Close - Open)
    
    sigma_o_sq = log_oc.rolling(window).var()
    sigma_c_sq = log_co.rolling(window).var()
    
    yz_var = sigma_o_sq + k * sigma_c_sq + (1-k) * roll_rs
    
    return np.sqrt(yz_var)

def calc_kyle_lambda(df, window=30):
    """
    Kyle's Lambda Proxy (Amihud Illiquidity adapted).
    Measures price impact per unit of volume.
    Lambda = |Return| / (Volume * Price)
    
    High Lambda = Low Liquidity (Price moves easily with volume)
    Low Lambda = High Liquidity (Price absorbs volume)
    """
    # Absolute Log Return
    abs_ret = np.abs(np.log(df['close'] / df['close'].shift(1)))
    
    # Dollar Volume = Volume * Price (Approx)
    dollar_vol = df['volume'] * df['close']
    
    # Avoid div by zero
    dollar_vol = dollar_vol.replace(0, np.nan)
    
    # Instantaneous Lambda
    inst_lambda = abs_ret / dollar_vol
    
    # Rolling Average
    return inst_lambda.rolling(window).mean()

def calc_market_force(df, window=10):
    """
    Physics-inspired 'Force' metric.
    Force = Mass * Acceleration
    Mass = Volume
    Velocity = Price Change / Time (or just Price Change per bar)
    Acceleration = Change in Velocity
    
    Force = Volume * (Velocity_t - Velocity_{t-1})
    """
    # Velocity: Log Return (per bar)
    velocity = np.log(df['close'] / df['close'].shift(1))
    
    # Acceleration: Change in Velocity
    acceleration = velocity.diff()
    
    # Mass: Volume
    mass = df['volume']
    
    # Force
    force = mass * acceleration
    
    # We might want a rolling smoothed version or magnitude
    return force.rolling(window).mean()

def calc_fractal_dimension(series, window=30):
    """
    Fractal Dimension Index (FDI).
    Uses the 'variogram' or 'box-counting' like approach approximation.
    Here we use the simple path-length definition (Sevcik).
    
    FDI = 1 + (log(L) + log(2)) / log(2*n)
    where L is total path length over window normalized to unit box.
    """
    # Need to normalize inputs to unit square [0,1]x[0,1] within the window
    
    def _fdi(x):
        # x is price series
        n = len(x)
        if n < 2: return np.nan
        
        # Normalize Price to [0,1]
        p_min = np.min(x)
        p_max = np.max(x)
        if p_max == p_min: return 1.0 # Flat line = dimension 1
        
        # Normalized Prices
        p_norm = (x - p_min) / (p_max - p_min)
        
        # Normalized Time (0 to 1)
        t_norm = np.linspace(0, 1, n)
        
        # Calculate Path Length
        # Euclidean distance between successive points (dt, dp)
        dt = t_norm[1] - t_norm[0] # Constant
        dp = np.diff(p_norm)
        
        length = np.sum(np.sqrt(dp**2 + dt**2))
        
        # Sevcik's formula approximation or similar
        # D = 1 + ln(L) / ln(2) # roughly
        # A clearer one: D = log(N_boxes) / log(1/scale)
        # Using a simpler metric often called 'Fractal Dimension' in trading:
        # FDI = 1 + (log(L) + log(2)) / log(2 * (n-1))  <-- Sevcik
        
        if length <= 0: return 1.0
        return 1 + (np.log(length) + np.log(2)) / np.log(2 * (n - 1))

    return series.rolling(window=window).apply(_fdi, raw=True)


# --- Feature Engineering Functions ---

def add_physics_features(df):
    if df is None: return None
    df = df.copy()
    print("Calculating Physics Features...")
    
    # Ensure log_ret exists for Entropy calc
    if 'log_ret' not in df.columns:
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
    df['frac_diff_04'] = frac_diff_ffd(df['close'], d=0.4)
    df['frac_diff_02'] = frac_diff_ffd(df['close'], d=0.2)
    df['hurst_100'] = get_hurst_exponent(df['close'], window=100)
    df['hurst_200'] = get_hurst_exponent(df['close'], window=200)
    df['hurst_400'] = get_hurst_exponent(df['close'], window=400)
    
    # Shannon Entropy (Disorder)
    # Using log_ret (stationarized) is better for distribution analysis than raw price
    df['entropy_100'] = get_shannon_entropy(df['log_ret'], window=100)
    df['entropy_200'] = get_shannon_entropy(df['log_ret'], window=200)
    df['entropy_400'] = get_shannon_entropy(df['log_ret'], window=400)
    
    return df

def add_advanced_physics_features(df, windows=[50, 100, 200]):
    if df is None: return None
    df = df.copy()
    print("Calculating Advanced Physics Features (YZ Vol, Kyle's Lambda, Force, FDI)...")
    
    # Yang-Zhang Volatility (Best Open-Close Estimator)
    for w in windows:
        df[f'yang_zhang_vol_{w}'] = calc_yang_zhang_volatility(df, window=w)
        
        # Kyle's Lambda (Liquidity Cost)
        df[f'kyle_lambda_{w}'] = calc_kyle_lambda(df, window=w)
        
        # Market Force (Physics)
        # Force is instantaneous, but we smooth it
        df[f'market_force_{w}'] = calc_market_force(df, window=w)
        
        # Fractal Dimension Index (Roughness/Complexity)
        df[f'fdi_{w}'] = calc_fractal_dimension(df['close'], window=w)
        
    return df
