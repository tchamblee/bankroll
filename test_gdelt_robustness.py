import pandas as pd
import numpy as np
import config
from genome import Strategy
from backtest.engine import BacktestEngine
from feature_engine.core import FeatureEngine
from feature_engine import loader
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------
# 1. SETUP & STRATEGY
# ------------------------------------------------------------------------------------
STRAT_JSON = {
    "name": "Mutant_9049_Simple_S2",
    "long_genes": [
        {"type": "zscore", "feature": "delta_epu_total_25", "operator": ">", "threshold": 1.8289, "window": 200},
        {"type": "divergence", "feature_a": "volatility_100", "feature_b": "pres_trend_50", "window": 50},
        {"type": "event", "feature": "delta_fdi_25_100", "operator": "<", "threshold": -1.0, "window": 52},
        {"type": "cross", "feature_left": "delta_news_tone_eur_100", "direction": "below", "feature_right": "delta_news_tone_eur_50"}
    ],
    "short_genes": [
        {"type": "flux", "feature": "delta_skew_200_100", "operator": ">", "threshold": 0.0395, "lag": 50},
        {"type": "zscore", "feature": "delta_velocity_200_25", "operator": ">", "threshold": 1.5, "window": 20},
        {"type": "extrema", "feature": "rel_strength_z_zn", "mode": "max", "window": 100}
    ],
    "min_concordance": 3,
    "horizon": 120
}

strategy = Strategy.from_dict(STRAT_JSON)
strategy.horizon = 120
strategy.stop_loss_pct = 2.0
strategy.take_profit_pct = 4.0

print(f"🧪 Testing Strategy: {strategy.name}")

# ------------------------------------------------------------------------------------
# 2. DATA LOADING & BASELINE (STEP)
# ------------------------------------------------------------------------------------
print("\n--- Generating BASELINE Features (Step Function) ---")
engine = FeatureEngine(config.DIRS['DATA_CLEAN_TICKS'])

# Load Primary
print("Loading EURUSD...")
primary_df = engine.load_ticker_data("CLEAN_EURUSD.parquet")
engine.create_volume_bars(primary_df, volume_threshold=config.VOLUME_THRESHOLD)

# Load Context (TNX, ZN, etc for Short logic)
print("Loading Context (TNX, ZN)...")
tnx = engine.load_ticker_data("CLEAN_TNX.parquet")
zn = engine.load_ticker_data("CLEAN_ZN.parquet")

if zn is not None:
    # Hack to mimic full pipeline for minimal test
    engine.add_intermarket_features({'_zn': zn})

# Load GDELT (Daily Step)
gdelt_daily = engine.load_gdelt_data()
print(f"DEBUG: gdelt_daily type: {type(gdelt_daily)}")
if isinstance(gdelt_daily, pd.DataFrame):
    print(f"DEBUG: gdelt_daily shape: {gdelt_daily.shape}")
    
engine.add_gdelt_features(gdelt_daily) # Uses the legacy logic (Shift 1 Day)

# Compute Standard Features
engine.add_features_to_bars(windows=[25, 50, 100, 200, 400, 800, 1600])
engine.add_physics_features()
engine.add_delta_features(lookback=25)
engine.add_delta_features(lookback=50)
engine.add_delta_features(lookback=100)

baseline_data = engine.bars.copy()
print(f"Baseline Data: {len(baseline_data)} bars")

# ------------------------------------------------------------------------------------
# 3. EXPERIMENT (SMOOTH/INTERPOLATED)
# ------------------------------------------------------------------------------------
print("\n--- Generating SMOOTHED Features (Interpolated) ---")
# Re-init engine to clear bars
engine_smooth = FeatureEngine(config.DIRS['DATA_CLEAN_TICKS'])
engine_smooth.bars = engine.create_volume_bars(primary_df, volume_threshold=config.VOLUME_THRESHOLD).copy()

if zn is not None:
    engine_smooth.add_intermarket_features({'_zn': zn})

# --- MANUAL SMOOTHING OF GDELT ---
# 1. Reindex Daily to Minutely
print("Interpolating GDELT Data...")
# Range: Start of Bars to End of Bars
start_dt = engine_smooth.bars['time_start'].min()
end_dt = engine_smooth.bars['time_start'].max()
idx = pd.date_range(start_dt, end_dt, freq='1min', tz='UTC')

# Reindex GDELT to this range
gdelt_resampled = gdelt_daily.reindex(idx, method='ffill') # First fill to establish baseline

# 2. Interpolate
# To truly verify robustness, we want gradual changes, not steps.
# However, reindex(ffill) just makes more steps.
# We need to map the daily values to 00:00 of each day, then interpolate between them.
gdelt_daily_idx = gdelt_daily.copy()
# Shift to avoid lookahead (T's data known at T+1)
gdelt_daily_idx = gdelt_daily_idx.shift(1).dropna() 

# Upsample to 1min and Interpolate
gdelt_smooth = gdelt_daily_idx.resample('1min').interpolate(method='linear')
# Fill remaining gaps
gdelt_smooth = gdelt_smooth.reindex(idx).ffill().bfill()

# 3. Merge_AsOf
gdelt_reset = gdelt_smooth.reset_index().rename(columns={'index': 'time_start'})
# Ensure types match
gdelt_reset['time_start'] = pd.to_datetime(gdelt_reset['time_start'], utc=True)
engine_smooth.bars['time_start'] = pd.to_datetime(engine_smooth.bars['time_start'], utc=True)

engine_smooth.bars = pd.merge_asof(
    engine_smooth.bars.sort_values('time_start'),
    gdelt_reset.sort_values('time_start'),
    on='time_start',
    direction='backward'
)

# Fill GDELT cols
cols_to_fill = ['epu_total', 'news_tone_eur', 'news_tone_usd', 'news_vol_eur', 'news_vol_usd']
for c in cols_to_fill:
    if c in engine_smooth.bars.columns:
        engine_smooth.bars[c] = engine_smooth.bars[c].fillna(0)

# Compute Features (Delta will now see small smooth changes instead of spikes)
engine_smooth.add_features_to_bars(windows=[25, 50, 100, 200, 400, 800, 1600])
engine_smooth.add_physics_features()
engine_smooth.add_delta_features(lookback=25)
engine_smooth.add_delta_features(lookback=50)
engine_smooth.add_delta_features(lookback=100)

smooth_data = engine_smooth.bars.copy()

# ------------------------------------------------------------------------------------
# 4. SIMULATION
# ------------------------------------------------------------------------------------

def run_test(name, data):
    print(f"\n🚀 Running Simulation: {name}")
    bt = BacktestEngine(data)
    # Force single strategy eval
    res, returns = bt.evaluate_population([strategy], set_type='test', return_series=True, time_limit=120)
    
    if not res.empty:
        stats = res.iloc[0]
        print(f"  Sortino: {stats['sortino']:.2f}")
        print(f"  Sharpe:  {stats['sharpe']:.2f}")
        print(f"  Trades:  {stats['trades']}")
        print(f"  Return:  {stats['total_return']:.4f}")
    else:
        print("  No trades generated.")
    
    bt.shutdown()
    return returns

# Run Baseline
ret_base = run_test("BASELINE (Step Data)", baseline_data)

# Run Smooth
ret_smooth = run_test("ROBUSTNESS (Smooth Data)", smooth_data)

# ------------------------------------------------------------------------------------
# 5. ANALYSIS
# ------------------------------------------------------------------------------------
# Check specific feature behavior
print("\n🔍 Feature Analysis (delta_epu_total_25):")
feat = "delta_epu_total_25"

if feat in baseline_data.columns:
    base_max = baseline_data[feat].max()
    base_99 = np.percentile(baseline_data[feat], 99)
    print(f"  Baseline Max: {base_max:.4f} | 99th%: {base_99:.4f}")

if feat in smooth_data.columns:
    smooth_max = smooth_data[feat].max()
    smooth_99 = np.percentile(smooth_data[feat], 99)
    print(f"  Smooth Max:   {smooth_max:.4f} | 99th%: {smooth_99:.4f}")
    
print("\n📝 Conclusion:")
if ret_base is not None and len(ret_base) > 0 and (ret_smooth is None or len(ret_smooth) == 0):
    print("❌ Strategy COLLAPSED on smooth data. It relies on the 'Daily Step' artifact.")
elif ret_smooth is not None and len(ret_smooth) > 0:
    print("✅ Strategy SURVIVED smoothing. It is robust.")
else:
    print("⚠️ Inconclusive results.")
