import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os
import time
import re
import glob
import shutil
import tempfile
import numpy as np
import nest_asyncio
from datetime import datetime, timezone, timedelta

# Project Imports
from feature_engine.core import FeatureEngine
from genome.strategy import Strategy
from backtest.feature_computation import precompute_base_features, ensure_feature_context
import config as cfg

nest_asyncio.apply()

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Bankroll Cockpit",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "output", "live_state.json")
BARS_FILE = os.path.join(BASE_DIR, "processed_data", "live_bars.parquet")
LOG_FILE = os.path.join(BASE_DIR, "logs", "paper_trade.log")
FRED_FILE = os.path.join(BASE_DIR, "data", "fred_macro_daily.parquet")
COT_FILE = os.path.join(BASE_DIR, "data", "cot_weekly.parquet")
GDELT_DIR = os.path.join(BASE_DIR, "data", "gdelt", "v2_gkg")
RAW_TICKS_DIR = os.path.join(BASE_DIR, "data", "raw_ticks")
MUTEX_FILE = os.path.join(BASE_DIR, "output", "strategies", "mutex_portfolio.json")

# --- HELPER FUNCTIONS ---

def load_strategy_names():
    if not os.path.exists(MUTEX_FILE):
        return {}
    try:
        with open(MUTEX_FILE, "r") as f:
            data = json.load(f)
            # Map Index -> Name
            return {i: s.get('name', f"Strat_{i}") for i, s in enumerate(data)}
    except Exception as e:
        return {}

def load_strategies_full():
    """Loads actual Strategy objects from mutex portfolio."""
    if not os.path.exists(MUTEX_FILE):
        return []
    strategies = []
    try:
        with open(MUTEX_FILE, "r") as f:
            data = json.load(f)
            for d in data:
                try:
                    s = Strategy.from_dict(d)
                    # Hydrate extras
                    s.horizon = d.get('horizon', 120)
                    s.stop_loss_pct = d.get('stop_loss_pct', 2.0)
                    s.take_profit_pct = d.get('take_profit_pct', 4.0)
                    strategies.append(s)
                except: pass
    except Exception as e:
        st.error(f"Error loading strategies: {e}")
    return strategies

def load_state():
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading state: {e}")
        return None

def load_bars(limit=300, full_history=False):
    if not os.path.exists(BARS_FILE):
        return None
    try:
        df = pd.read_parquet(BARS_FILE)
        # Ensure UTC
        if 'time_start' in df.columns:
            df['time_start'] = pd.to_datetime(df['time_start'], utc=True)
        if full_history:
            return df
        return df.tail(limit)
    except Exception as e:
        st.error(f"Error loading bars: {e}")
        return None

def compute_live_features(strategies, bars_df):
    """
    Computes features on the fly for the Inspector View.
    Returns a dict: {feature_key: current_value}
    """
    if bars_df is None or len(bars_df) < 200:
        return {}

    # 1. Feature Engine Setup
    engine = FeatureEngine(os.path.join(BASE_DIR, "data"))
    engine.bars = bars_df.copy()
    
    # Extract Helpers
    def extract_series(name):
        if name in engine.bars.columns:
            # Create a minimal DF as expected by add_... methods
            return engine.bars[[name, 'time_start']].rename(columns={name: 'close', 'time_start': 'ts_event'})
        return None

    # 2. Run Pipeline (Matching paper_trade.py)
    windows = [25, 50, 100, 200, 400, 800, 1600, 3200]
    
    # Standard
    engine.add_features_to_bars(windows=windows)
    engine.add_physics_features()
    engine.add_advanced_physics_features(windows=windows)
    engine.add_microstructure_features()
    engine.add_delta_features(lookback=25)
    
    # Macro
    engine.add_macro_voltage_features(
        us2y_df=extract_series('US2Y'), 
        schatz_df=extract_series('SCHATZ'), 
        tnx_df=extract_series('TNX'), 
        bund_df=extract_series('BUND'), 
        windows=[50, 100]
    )
    
    # Crypto
    engine.add_crypto_features(extract_series('IBIT'))
    
    # Intermarket
    intermarket_dfs = {
        '_es': extract_series('ES'),
        '_zn': extract_series('ZN'),
        '_6e': extract_series('6E')
    }
    intermarket_dfs = {k: v for k, v in intermarket_dfs.items() if v is not None}
    if intermarket_dfs:
        engine.add_intermarket_features(intermarket_dfs)

    if 'time_start' in engine.bars.columns:
        engine.bars['time_hour'] = engine.bars['time_start'].dt.hour
        engine.bars['time_weekday'] = engine.bars['time_start'].dt.dayofweek

    # 3. JIT Computation (Context)
    feature_snapshot = {}
    
    # Use Temp Dir for JIT
    with tempfile.TemporaryDirectory() as temp_dir:
        existing_keys = set()
        
        # A. Base Features
        precompute_base_features(engine.bars, temp_dir, existing_keys)
        
        # B. Derived Features (Genes)
        ensure_feature_context(strategies, temp_dir, existing_keys)
        
        # C. Load Latest Values
        for key in existing_keys:
            try:
                arr = np.load(os.path.join(temp_dir, f"{key}.npy"))
                if len(arr) > 0:
                    feature_snapshot[key] = arr[-1] # Take the latest value
            except: pass
            
    return feature_snapshot

def get_gene_key(gene):
    """Resolves the feature key used by the gene (e.g., 'zscore_...')."""
    if gene.type == 'delta': 
        return f"delta_{gene.feature}_{gene.lookback}"
    elif gene.type == 'zscore': 
        return f"zscore_{gene.feature}_{gene.window}"
    elif gene.type == 'slope':
        # Divergence uses slope internally
        pass 
    elif gene.type == 'persistence':
        return gene.feature # Uses raw feature
    elif gene.type == 'correlation':
        f1, f2 = sorted([gene.feature_left, gene.feature_right])
        return f"corr_{f1}_{f2}_{gene.window}"
    elif gene.type == 'flux':
        return f"flux_{gene.feature}_{gene.lag}"
    elif gene.type == 'efficiency':
        return f"eff_{gene.feature}_{gene.window}"
    
    # Fallback for simple features or Cross/Relational which use raw features
    if hasattr(gene, 'feature'): return gene.feature
    if hasattr(gene, 'feature_left'): return gene.feature_left # Just return one for display context
    return "unknown"

def load_logs(lines=100):
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r") as f:
            all_lines = f.readlines()
            return [line.strip() for line in all_lines[-lines:]]
    except Exception as e:
        st.error(f"Error loading logs: {e}")
        return []

def parse_trades_from_logs(strat_map=None):
    """Parses paper_trade.log to extract recent Entry/Exit events for plotting."""
    if not os.path.exists(LOG_FILE):
        return []
    
    events = []
    # Regex patterns
    entry_pattern = re.compile(r"VIRTUAL ENTRY: (BUY|SELL) (\d+) lots \(Strat (\d+)\) @ ([\d\.]+)")
    exit_pattern = re.compile(r"VIRTUAL EXIT: (.*?) @ ([\d\.]+)")
    
    try:
        with open(LOG_FILE, "r") as f:
            for line in f:
                ts_str = line.split(",")[0] 
                try:
                    ts = pd.to_datetime(ts_str) 
                except:
                    continue 
                
                # Check Entry
                m_entry = entry_pattern.search(line)
                if m_entry:
                    s_idx = int(m_entry.group(3))
                    s_name = strat_map.get(s_idx, str(s_idx)) if strat_map else str(s_idx)
                    
                    events.append({
                        "time": ts,
                        "type": "ENTRY",
                        "side": m_entry.group(1),
                        "lots": m_entry.group(2),
                        "strat": s_name,
                        "price": float(m_entry.group(4))
                    })
                    continue
                    
                # Check Exit
                m_exit = exit_pattern.search(line)
                if m_exit:
                    events.append({
                        "time": ts,
                        "type": "EXIT",
                        "reason": m_exit.group(1),
                        "price": float(m_exit.group(2))
                    })
    except Exception as e:
        pass
        
    return pd.DataFrame(events)

def get_file_age(filepath):
    if not os.path.exists(filepath):
        return None, "Missing"
    mtime = os.path.getmtime(filepath)
    age_seconds = time.time() - mtime
    return age_seconds, datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')

def get_dir_latest_age(dirpath, pattern):
    if not os.path.exists(dirpath):
        return None, "Missing Dir"
    files = glob.glob(os.path.join(dirpath, pattern))
    if not files:
        return None, "Empty Dir"
    latest_file = max(files, key=os.path.getmtime)
    return get_file_age(latest_file)

def format_age(seconds):
    if seconds is None: return "N/A"
    if seconds < 60: return f"{int(seconds)}s"
    if seconds < 3600: return f"{int(seconds/60)}m"
    if seconds < 86400: return f"{int(seconds/3600)}h"
    return f"{int(seconds/86400)}d"

# --- MAIN UI ---

st.title("Paper Trade Dashboard")

# Sidebar
st.sidebar.header("Controls")
view = st.sidebar.radio("View", ["Cockpit", "Strategy Inspector", "System Health"])
auto_refresh = st.sidebar.checkbox("Auto-Refresh (5s)", value=False)
if st.sidebar.button("Manual Refresh"):
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown("### System Status")

# --- RESET DATA ---
if st.sidebar.button("⚠️ RESET DATA", type="primary"):
    # 1. Create Trigger for Backend Process
    trigger_file = os.path.join(BASE_DIR, "output", "RESET_TRIGGER")
    with open(trigger_file, "w") as f:
        f.write("RESET")
        
    # 2. Clear Local Files (Immediate Feedback)
    if os.path.exists(STATE_FILE):
        try: os.remove(STATE_FILE)
        except: pass
        
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f: f.write("")
        
    st.toast("Reset Signal Sent!", icon="🗑️")
    time.sleep(1)
    st.rerun()

# --- HEALTH CHECK (SIDEBAR) ---
state = load_state()
is_alive = False
last_heartbeat_str = "Unknown"

if state and 'updated_at' in state:
    last_update = pd.to_datetime(state['updated_at']).replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    diff = (now - last_update).total_seconds()
    last_heartbeat_str = last_update.strftime("%H:%M:%S UTC")
    
    if diff < 120: # 2 minutes
        is_alive = True
        st.sidebar.success(f"🟢 System Online\n\nLast Beat: {last_heartbeat_str}")
    else:
        st.sidebar.error(f"🔴 System Stale\n\nLast Beat: {last_heartbeat_str}\n\n({int(diff)}s ago)")
else:
    st.sidebar.warning("⚪ No State Found")

# --- VIEWS ---

if view == "Cockpit":
    # --- METRICS ROW ---
    
    strat_map = load_strategy_names()
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    if state:
        pos = state.get("position", 0)
        entry = state.get("entry_price", 0.0)
        strat_idx = state.get("active_strat_idx", -1)
        balance = state.get("balance", 60000.0)
        realized = state.get("realized_pnl", 0.0)
        current_atr = state.get("current_atr", 0.0)
        
        # Resolve Strategy Name
        strat_name = "None"
        if strat_idx >= 0:
            strat_name = strat_map.get(strat_idx, f"#{strat_idx}")
        
        # Calculate Unrealized PnL
        bars = load_bars(1)
        current_price = bars.iloc[-1]["close"] if bars is not None and not bars.empty else entry
        
        pnl = 0.0
        if pos != 0 and entry > 0:
            if pos > 0:
                pnl = (current_price - entry) * abs(pos) * 100000 
            else:
                pnl = (entry - current_price) * abs(pos) * 100000

        with col1:
            st.metric("Balance", f"${balance:,.0f}")
        with col2:
            st.metric("Realized PnL", f"${realized:,.2f}", delta=realized, delta_color="normal")
        with col3:
            st.metric("Position", f"{pos} Lots", delta_color="off")
        with col4:
            st.metric("Entry Price", f"{entry:.5f}", delta=f"{current_price - entry:.5f}" if pos > 0 else f"{entry - current_price:.5f}")
        with col5:
            st.metric("Unrealized PnL", f"${pnl:.2f}", delta_color="normal")
        with col6:
            st.metric("ATR (Bars)", f"{current_atr:.6f}")
        with col7:
            st.metric("Active Strat", strat_name)
    else:
        col1.metric("Balance", "---")
        col2.metric("Realized PnL", "---")
        col3.metric("Position", "OFFLINE")
        col4.metric("Entry Price", "---")
        col5.metric("Unrealized PnL", "---")
        col6.metric("ATR", "---")
        col7.metric("Active Strat", "---")

    st.divider()

    # --- CHARTS ---
    st.subheader("Market View (EURUSD)")

    bars_df = load_bars(300)
    trade_events = parse_trades_from_logs(strat_map)

    if bars_df is not None and not bars_df.empty:
        fig = go.Figure(data=[go.Candlestick(x=bars_df['time_start'],
                        open=bars_df['open'],
                        high=bars_df['high'],
                        low=bars_df['low'],
                        close=bars_df['close'],
                        name="EURUSD")])

        # Add Entry Line if active
        if state and state.get("position", 0) != 0:
            entry_price = state.get("entry_price", 0.0)
            position = state.get("position", 0)
            
            # Entry Line
            color = "green" if position > 0 else "red"
            label = f"Active {'LONG' if position > 0 else 'SHORT'} Entry"
            fig.add_hline(y=entry_price, line_dash="solid", line_color=color, annotation_text=label)
            
            # SL/TP Lines
            current_sl = state.get("current_sl", 0.0)
            current_tp = state.get("current_tp", 0.0)
            
            if current_sl > 0:
                fig.add_hline(y=current_sl, line_dash="dash", line_color="red", annotation_text="Stop Loss")
            if current_tp > 0:
                fig.add_hline(y=current_tp, line_dash="dash", line_color="green", annotation_text="Take Profit")

        # Add Historical Trade Markers
        if not trade_events.empty:
            min_time = bars_df['time_start'].min()
            if trade_events['time'].dt.tz is None:
                 trade_events['time'] = trade_events['time'].dt.tz_localize('UTC') 
                 
            mask = trade_events['time'] >= min_time
            visible_events = trade_events[mask]
            
            entries = visible_events[visible_events['type'] == 'ENTRY']
            if not entries.empty:
                buys = entries[entries['side'] == 'BUY']
                fig.add_trace(go.Scatter(
                    x=buys['time'], y=buys['price'],
                    mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'),
                    name='Buy Entry'
                ))
                sells = entries[entries['side'] == 'SELL']
                fig.add_trace(go.Scatter(
                    x=sells['time'], y=sells['price'],
                    mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'),
                    name='Sell Entry'
                ))
                
            exits = visible_events[visible_events['type'] == 'EXIT']
            if not exits.empty:
                fig.add_trace(go.Scatter(
                    x=exits['time'], y=exits['price'],
                    mode='markers', marker=dict(symbol='x', size=10, color='orange'),
                    name='Exit'
                ))

        fig.update_layout(
            height=600, 
            xaxis_rangeslider_visible=False, 
            template="plotly_dark", 
            uirevision='constant',
            xaxis=dict(uirevision='constant'),
            yaxis=dict(uirevision='constant'),
        )
        st.plotly_chart(fig, use_container_width=True, key="main_chart")
    else:
        st.info("Waiting for bar data to generate chart...")

    # --- RECENT TRADES TABLE ---
    if not trade_events.empty:
        st.subheader("Recent Trade Events")
        st.dataframe(trade_events.sort_values("time", ascending=False).head(10), width="stretch")

    # --- LOGS ---
    st.subheader("System Logs")
    logs = load_logs(100)
    logs.reverse()  # Show most recent first
    log_text = "\n".join(logs)
    st.text_area("Live Logs", log_text, height=300, disabled=True)

elif view == "Strategy Inspector":
    st.header("🧬 Strategy Inspector")
    
    with st.spinner("Loading Strategy Data & Computing Live Features..."):
        strategies = load_strategies_full()
        bars_full = load_bars(full_history=True)
        
        if not strategies:
            st.warning("No strategies found in portfolio.")
        elif bars_full is None:
            st.warning("No live bar data available.")
        else:
            # Compute Context
            context = compute_live_features(strategies, bars_full)
            
            st.success(f"Loaded {len(strategies)} strategies. Context contains {len(context)} live features.")
            
            for i, strat in enumerate(strategies):
                with st.expander(f"Strat {i}: {strat.name} (H{strat.horizon})", expanded=(i==0)):
                    col1, col2 = st.columns(2)
                    
                    # LONG GENES
                    with col1:
                        st.subheader("Long Genes")
                        for gene in strat.long_genes:
                            key = get_gene_key(gene)
                            val = context.get(key, np.nan)
                            
                            # Determine Status
                            met = False
                            desc = f"{gene.type}"
                            
                            if gene.type == 'zscore' or gene.type == 'delta' or gene.type == 'persistence':
                                desc = f"{key} {gene.operator} {gene.threshold:.4f}"
                                if gene.operator == '>': met = val > gene.threshold
                                elif gene.operator == '<': met = val < gene.threshold
                                elif gene.operator == '>=': met = val >= gene.threshold
                                elif gene.operator == '<=': met = val <= gene.threshold
                                
                            elif gene.type == 'cross':
                                desc = f"{gene.feature_left} {gene.direction} {gene.feature_right}"
                                # Cross is hard to visualize with single value, usually requires history
                                # Simplified: Just show value of left?
                                val = "N/A" 
                                met = False # Cannot easily determine single-point cross without history logic
                                
                            # Display
                            icon = "✅" if met else "⬜"
                            val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
                            st.markdown(f"{icon} **{desc}** (Cur: `{val_str}`)")

                    # SHORT GENES
                    with col2:
                        st.subheader("Short Genes")
                        for gene in strat.short_genes:
                            key = get_gene_key(gene)
                            val = context.get(key, np.nan)
                            
                            # Determine Status
                            met = False
                            desc = f"{gene.type}"
                            
                            if gene.type == 'zscore' or gene.type == 'delta' or gene.type == 'persistence':
                                desc = f"{key} {gene.operator} {gene.threshold:.4f}"
                                if gene.operator == '>': met = val > gene.threshold
                                elif gene.operator == '<': met = val < gene.threshold
                                elif gene.operator == '>=': met = val >= gene.threshold
                                elif gene.operator == '<=': met = val <= gene.threshold
                                
                            # Display
                            icon = "✅" if met else "⬜"
                            val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
                            st.markdown(f"{icon} **{desc}** (Cur: `{val_str}`)")

elif view == "System Health":
    st.header("🏥 System Health & Status")
    
    health_data = []
    
    # 1. Core System
    age, timestamp = get_file_age(STATE_FILE)
    status = "🟢" if age and age < 120 else "🔴"
    health_data.append(["State File", status, format_age(age), timestamp, "Core Bot State"])
    
    age, timestamp = get_file_age(BARS_FILE)
    status = "🟢" if age and age < 120 else "🔴"
    health_data.append(["Live Bars", status, format_age(age), timestamp, "Market Data Persistence"])

    # 2. Ingestion
    age, timestamp = get_dir_latest_age(RAW_TICKS_DIR, "*.parquet")
    status = "🟢" if age and age < 600 else "🔴" # 10 mins tolerance
    health_data.append(["Raw Ticks", status, format_age(age), timestamp, "IBKR Ingestion Pipeline"])

    age, timestamp = get_dir_latest_age(GDELT_DIR, "*.parquet")
    status = "🟢" if age and age < 86400 else "🟡" # 24h tolerance
    health_data.append(["GDELT Parquet", status, format_age(age), timestamp, "News Sentiment Ingestion"])

    # 3. Macro
    age, timestamp = get_file_age(FRED_FILE)
    status = "🟢" if age and age < 86400 else "🟡"
    health_data.append(["FRED Macro", status, format_age(age), timestamp, "Economic Data"])
    
    age, timestamp = get_file_age(COT_FILE)
    status = "🟢" if age and age < 86400*7 else "🟡"
    health_data.append(["COT Data", status, format_age(age), timestamp, "Futures Positioning"])

    df_health = pd.DataFrame(health_data, columns=["Component", "Status", "Age", "Last Modified", "Description"])
    st.dataframe(df_health, width="stretch", height=400)

    # Disk Usage
    st.subheader("💾 Disk Usage")
    total, used, free = shutil.disk_usage(BASE_DIR)
    col1, col2 = st.columns(2)
    col1.metric("Free Space", f"{free // (2**30)} GB")
    col2.metric("Total Space", f"{total // (2**30)} GB")
    st.progress(used/total, text=f"Used: {int(used/total*100)}%")

# --- AUTO REFRESH LOGIC ---
if auto_refresh:
    time.sleep(5)
    st.rerun()
