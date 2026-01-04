import numpy as np
import pandas as pd
from scipy.stats import linregress
from numba import jit
from joblib import Parallel, delayed

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
    Optimized using np.convolve.
    """
    # 1. Determine window size where weights drop below threshold
    w_test = [1.0]
    k = 1
    while abs(w_test[-1]) > thres:
        w_k = -w_test[-1] * (d - k + 1) / k
        w_test.append(w_k)
        k += 1
    
    # w_test is [w_0, w_1, ..., w_k]
    # We want to apply this filter such that y[t] = w_0*x[t] + w_1*x[t-1] + ...
    # np.convolve(x, v, mode='valid') computes sum(x[n-m]*v[m])
    # If we pass v = w_test, it computes x[t]*w_k + x[t-1]*w_{k-1} ... which is REVERSED order of what we usually want 
    # if w_test was the chronological weights. 
    # But w_test is generated as lag weights: w_0 is weight for lag 0 (current), w_1 for lag 1.
    # So we want y[t] = w_test[0]*x[t] + w_test[1]*x[t-1] + ...
    # This requires convolving x with w_test in standard convolution.
    
    weights = np.array(w_test)
    
    series_clean = pd.Series(series).dropna()
    vals = series_clean.values
    
    if len(vals) < len(weights):
        return pd.Series(np.nan, index=series.index)
        
    # Convolve
    # Default np.convolve flips the second array. 
    # We want sum(x[t-i] * w[i]).
    # np.convolve(x, w) = sum(x[t-k] * w[k]). 
    # This matches exactly if w contains [w_0, w_1, ...]
    res = np.convolve(vals, weights, mode='valid')
    
    # Pad with NaNs at the beginning to match original index
    # 'valid' returns N - K + 1 points. We need N points.
    padding = np.full(len(weights) - 1, np.nan)
    res_full = np.concatenate([padding, res])
    
    return pd.Series(res_full, index=series_clean.index).reindex(series.index)

def get_hurst_exponent(series, window=100, lags=[2, 4, 8, 16, 32, 64]):
    """
    Estimates Hurst Exponent using variance of differences at multiple lags.
    Vectorized implementation for high performance.
    """
    # Pre-calculate regression weights for the lags
    # Slope m = sum(w_i * y_i) where y_i = log(std_lag_i)
    # x = log(lags)
    x = np.log(lags)
    x_mean = np.mean(x)
    denom = np.sum((x - x_mean)**2)
    
    # If denom is 0 (only 1 lag), we can't regress.
    if denom == 0:
        return pd.Series(np.nan, index=series.index)
        
    weights = (x - x_mean) / denom
    
    # Calculate rolling standard deviations for each lag globally
    # Hurst ~ log(std(diff)) vs log(lag)
    weighted_sum = 0
    
    for i, lag in enumerate(lags):
        # Rolling std of (price - price_lagged)
        # Note: series.diff(lag) calculates price[t] - price[t-lag]
        roll_std = series.diff(lag).rolling(window).std()
        
        # Log of std (handle 0/negatives gracefully, though std >= 0)
        # Add small epsilon to avoid log(0)
        log_std = np.log(roll_std.replace(0, np.nan))
        
        weighted_sum += weights[i] * log_std
        
    return weighted_sum

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
    Calculates Rolling Differential Entropy (Information Content).
    
    OPTIMIZATION NOTE:
    The original Shannon Entropy via histogram binning is extremely slow (O(N*W)).
    We replace it with Differential Entropy for a Gaussian process, which is 
    instantaneous to calculate and highly correlated with disorder/volatility.
    
    H(X) ~= 0.5 * log(2 * pi * e * sigma^2)
    """
    # We apply this on the series passed (usually log_ret)
    # Variance over the window
    var = series.rolling(window).var()
    
    # Differential Entropy
    # Protect against var <= 0
    var = var.replace(0, np.nan)
    entropy = 0.5 * np.log(2 * np.pi * np.e * var)
    
    return entropy

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

def calc_market_energy(df, window=10):
    """
    Physics-inspired 'Kinetic Energy'.
    KE = 0.5 * Mass * Velocity^2
    """
    velocity = np.log(df['close'] / df['close'].shift(1))
    mass = df['volume']
    
    ke = 0.5 * mass * (velocity ** 2)
    return ke.rolling(window).mean()

def calc_potential_energy(series, volume, window=10):
    """
    Physics-inspired 'Potential Energy' (Hooke's Law).
    U = 0.5 * k * x^2
    We assume 'k' (Spring Constant) is proportional to Volume (Mass).
    'x' is the displacement from the moving average (Equilibrium).
    """
    # Equilibrium position (Moving Average)
    ma = series.rolling(window).mean()
    
    # Displacement (Normalized)
    # x = (Price - MA) / MA  (Percentage deviation)
    x = (series - ma) / ma
    
    # Potential Energy
    # U = 0.5 * Mass * x^2
    pe = 0.5 * volume * (x ** 2)
    
    return pe.rolling(window).mean()

def calc_market_power(df, window=10):
    """
    Physics-inspired 'Power'.
    Power = Force * Velocity = (Mass * Accel) * Velocity
    """
    velocity = np.log(df['close'] / df['close'].shift(1))
    acceleration = velocity.diff()
    mass = df['volume']
    
    force = mass * acceleration
    power = force * velocity
    
    return power.rolling(window).mean()

from numba import jit

@jit(nopython=True)
def _jit_fdi_rolling(x, window):
    """
    Numba-optimized Rolling Fractal Dimension Index (Sevcik).
    """
    n = len(x)
    res = np.empty(n)
    res[:] = np.nan
    
    if n < window:
        return res
        
    for i in range(window - 1, n):
        # Slice window
        segment = x[i - window + 1 : i + 1]
        
        # Calculate min/max for normalization
        p_min = segment[0]
        p_max = segment[0]
        for val in segment:
            if val < p_min: p_min = val
            if val > p_max: p_max = val
            
        if p_max == p_min:
            res[i] = 1.0
            continue
            
        # Path Length Calculation (Sevcik)
        # Normalizing time to [0, 1] means dt = 1 / (window - 1)
        dt = 1.0 / (window - 1)
        length = 0.0
        
        # Pre-calc normalized prices to avoid redundant division
        # p_norm = (segment - p_min) / (p_max - p_min)
        inv_range = 1.0 / (p_max - p_min)
        
        prev_p = (segment[0] - p_min) * inv_range
        for k in range(1, window):
            curr_p = (segment[k] - p_min) * inv_range
            dp = curr_p - prev_p
            # dist = sqrt(dp^2 + dt^2)
            length += np.sqrt(dp*dp + dt*dt)
            prev_p = curr_p
            
        # FDI Formula: 1 + (log(L) + log(2)) / log(2 * (window - 1))
        # Numba supports np.log
        res[i] = 1.0 + (np.log(length) + np.log(2.0)) / np.log(2.0 * (window - 1))
        
    return res

def calc_fractal_dimension(series, window=30):
    """
    Fractal Dimension Index (FDI).
    Uses Numba-optimized implementation for speed.
    """
    vals = series.values.astype(np.float64)
    res = _jit_fdi_rolling(vals, window)
    return pd.Series(res, index=series.index)


# --- Feature Engineering Functions ---

def add_physics_features(df):
    if df is None: return None
    df = df.copy()
    print("Calculating Physics Features...")
    
    # Ensure log_ret exists for Entropy calc
    if 'log_ret' not in df.columns:
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
    # Purged redundant frac_diff_04
    df['frac_diff_02'] = frac_diff_ffd(df['close'], d=0.2)
    
    df['hurst_100'] = get_hurst_exponent(df['close'], window=100)
    df['hurst_200'] = get_hurst_exponent(df['close'], window=200)
    df['hurst_400'] = get_hurst_exponent(df['close'], window=400)
    
    # Rate of Change (ROC) for Physics Features
    # Hurst & Entropy are levels, so we check their absolute change (diff)
    df['hurst_roc_100'] = df['hurst_100'].diff()
    df['hurst_roc_200'] = df['hurst_200'].diff()
    
    # Shannon Entropy (Differential)
    # Measures disorder/uncertainty.
    df['entropy_100'] = get_shannon_entropy(df['log_ret'], window=100)
    df['entropy_200'] = get_shannon_entropy(df['log_ret'], window=200)
    df['entropy_400'] = get_shannon_entropy(df['log_ret'], window=400)
    
    df['entropy_roc_100'] = df['entropy_100'].diff()
    df['entropy_roc_200'] = df['entropy_200'].diff()
    
    return df

def _calc_physics_window_features(w, df_minimal):
    """Worker function for parallel window physics feature calculation."""
    res = {}
    
    # Kyle's Lambda
    lam = calc_kyle_lambda(df_minimal, window=w)
    res[f'kyle_lambda_{w}'] = lam
    
    # Market Energy (Kinetic)
    ke = calc_market_energy(df_minimal, window=w)
    res[f'market_energy_{w}'] = ke
    
    # Potential Energy (Spring/Tension)
    pe = calc_potential_energy(df_minimal['close'], df_minimal['volume'], window=w)
    res[f'potential_energy_{w}'] = pe
    
    # Lagrangian (Kinetic - Potential)
    # L > 0: High Motion, Low Tension (Breakout/Trend)
    # L < 0: Low Motion, High Tension (Overextended/Reversal Prone)
    lag = ke - pe
    res[f'lagrangian_{w}'] = lag
    
    # Action (Accumulated Lagrangian)
    # "Least Action Principle": Path of physical system minimizes action.
    # High Action accumulation = "Stressed" path (Potential Reversal/Shift).
    res[f'market_action_{w}'] = lag.rolling(w).sum()
    
    # Hamiltonian (Total Energy)
    # H = K + U (Total System Intensity)
    res[f'hamiltonian_{w}'] = ke + pe
    
    # Market Power
    res[f'market_power_{w}'] = calc_market_power(df_minimal, window=w)
    
    # Fractal Dimension Index
    fdi = calc_fractal_dimension(df_minimal['close'], window=w)
    res[f'fdi_{w}'] = fdi
    
    # ROC Features
    res[f'fdi_roc_{w}'] = fdi.diff()
            
    res[f'kyle_lambda_roc_{w}'] = lam.pct_change(fill_method=None)
    
    return res

def add_advanced_physics_features(df, windows=[50, 100, 200]):
    if df is None: return None
    df = df.copy()
    print("Calculating Advanced Physics Features (YZ Vol, Kyle's Lambda, FDI)...")
    
    # Minimal DF for workers to save memory/pickling time
    cols_needed = ['open', 'high', 'low', 'close', 'volume']
    df_min = df[cols_needed].copy()
    
    # Parallelize
    results = Parallel(n_jobs=-1)(
        delayed(_calc_physics_window_features)(w, df_min)
        for w in windows
    )
    
    new_cols = {}
    for r in results:
        new_cols.update(r)
        
    df_new = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, df_new], axis=1)

def calc_rolling_vwap(df, window):
    """
    Calculates Rolling Volume-Weighted Average Price.
    """
    product = df['close'] * df['volume']
    vwap = product.rolling(window).sum() / df['volume'].rolling(window).sum()
    return vwap

def add_interaction_features(df, windows=[100, 200, 400]):
    """
    Adds complex interaction features combining Physics, Microstructure, and Standard metrics.
    Must be called AFTER all base feature generation steps.
    """
    if df is None: return None
    print("Calculating Interaction Features (VWAP, Energy-Flow, FDI-Vol)...")
    df = df.copy()
    
    # We collect new columns in a dict to avoid fragmentation
    new_cols = {}
    
    for w in windows:
        # 1. VWAP Deviation (Mean Reversion / Trend)
        # (Close - VWAP) / VWAP
        vwap = calc_rolling_vwap(df, w)
        new_cols[f'vwap_diff_{w}'] = (df['close'] - vwap) / vwap
        
        # 2. Hurst-Adjusted Trend Strength
        # Hurst < 0.5 (Mean Rev) -> Flip Trend Strength?
        # Or just (Hurst - 0.5) * Trend.
        # If Hurst > 0.5 (Trend), Positive * Positive = Positive Signal.
        # If Hurst < 0.5 (Mean Rev), Negative * Positive = Negative Signal (Fade Trend).
        h_col = f'hurst_{w}'
        t_col = f'trend_strength_{w}'
        if h_col in df.columns and t_col in df.columns:
            new_cols[f'hurst_trend_{w}'] = (df[h_col] - 0.5) * df[t_col]
            
        # 3. FDI-Adjusted Volatility
        # FDI ~ 1.5 (Random). FDI < 1.5 (Trend). FDI > 1.5 (Mean Rev).
        # (FDI - 1.5) * Volatility.
        # If FDI > 1.5 (Choppy), Positive * Vol = High Volatility Chop (Risk).
        # If FDI < 1.5 (Trend), Negative * Vol = Trending Volatility (Opportunity?).
        f_col = f'fdi_{w}'
        v_col = f'volatility_{w}'
        if f_col in df.columns and v_col in df.columns:
             new_cols[f'fdi_vol_{w}'] = (df[f_col] - 1.5) * df[v_col]
             
        # 4. Energy-Flow (Intensity * Direction)
        # Market Energy (Kinetic) * Flow Trend ( buying pressure)
        # High Energy + High Buy Flow = Strong Upward Impulse
        e_col = f'market_energy_{w}'
        fl_col = f'flow_trend_{w}'
        if e_col in df.columns and fl_col in df.columns:
            new_cols[f'energy_flow_{w}'] = df[e_col] * df[fl_col]
            
        # 5. Clean Trend (Entropy-Adjusted Velocity)
        # Velocity / Entropy
        # High Velocity / Low Entropy = Strong Clean Trend
        vel_col = f'velocity_{w}'
        ent_col = f'entropy_{w}'
        if vel_col in df.columns and ent_col in df.columns:
            new_cols[f'clean_trend_{w}'] = df[vel_col] / df[ent_col].replace(0, 1)

        # 6. Market Temperature
        # Temp = Energy / Entropy
        # High Energy + Low Entropy = "Hot" Trend (Coherent)
        # High Energy + High Entropy = "Boiling" Chop (Incoherent)
        e_col = f'market_energy_{w}'
        ent_col = f'entropy_{w}'
        if e_col in df.columns and ent_col in df.columns:
             new_cols[f'market_temperature_{w}'] = df[e_col] / df[ent_col].replace(0, np.nan)

    # Batch assignment
    df_new = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, df_new], axis=1)
