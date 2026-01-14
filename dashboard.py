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
from feature_engine.pipeline import run_pipeline
from genome.strategy import Strategy
from backtest.strategy_loader import load_strategies as load_strategies_from_disk
from backtest.feature_computation import precompute_base_features, ensure_feature_context
import config as cfg

nest_asyncio.apply()

LOCAL_TZ = datetime.now().astimezone().tzinfo

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
MUTEX_FILE = cfg.MUTEX_PORTFOLIO_FILE

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
    try:
        strategies, _ = load_strategies_from_disk(source_type='mutex', load_metrics=False)
        return strategies
    except Exception as e:
        st.error(f"Error loading strategies: {e}")
        return []

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

    # 2. Run Pipeline (Using Shared Logic)
    data_cache = {
        'tnx': extract_series('TNX'),
        'us2y': extract_series('US2Y'),
        'zn': extract_series('ZN'),
        'vix': extract_series('VIX'),
        'ibit': extract_series('IBIT'),
        'tick_nyse': extract_series('TICK_NYSE'),
        'trin_nyse': extract_series('TRIN_NYSE'),
    }
    
    run_pipeline(engine, data_cache)

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
        return pd.DataFrame()
    
    events = []
    # Regex patterns
    entry_pattern = re.compile(r"VIRTUAL ENTRY: (BUY|SELL) ([\d\.]+) lots .*?Strat (.*?) @ ([\d\.]+)")
    exit_pattern = re.compile(r"VIRTUAL EXIT: (.*?) @ ([\d\.]+)")
    
    try:
        with open(LOG_FILE, "r") as f:
            for line in f:
                ts_str = line.split(",")[0] 
                try:
                    # Log timestamps are Local Time. Convert to UTC.
                    ts = pd.to_datetime(ts_str).tz_localize(LOCAL_TZ).tz_convert('UTC')
                except:
                    continue 
                
                # Check Entry
                m_entry = entry_pattern.search(line)
                if m_entry:
                    s_name = m_entry.group(3)
                    
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

def parse_closed_trades(strat_map=None):
    """Parses paper_trade.log to extract completed Trade Cycles."""
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    
    trades = []
    current_trade = None
    
    # Regex patterns
    entry_pattern = re.compile(r"VIRTUAL ENTRY: (BUY|SELL) ([\d\.]+) lots .*?Strat (.*?) @ ([\d\.]+)")
    exit_pattern = re.compile(r"VIRTUAL EXIT: (.*?) @ ([\d\.]+)")
    pnl_pattern = re.compile(r"Trade PnL: \$([-+]?[\d\.,]+)")
    
    try:
        with open(LOG_FILE, "r") as f:
            for line in f:
                ts_str = line.split(",")[0]
                try:
                    # Log timestamps are Local Time. Convert to UTC.
                    ts = pd.to_datetime(ts_str).tz_localize(LOCAL_TZ).tz_convert('UTC')
                except:
                    continue
                
                # Check Entry
                m_entry = entry_pattern.search(line)
                if m_entry:
                    s_name = m_entry.group(3)
                    
                    current_trade = {
                        "Entry Time": ts,
                        "Type": m_entry.group(1),
                        "Lots": float(m_entry.group(2)),
                        "Strat": s_name,
                        "Entry Price": float(m_entry.group(4)),
                        "Exit Time": pd.NaT,
                        "Exit Price": 0.0,
                        "Reason": "",
                        "PnL": 0.0
                    }
                    continue
                
                # Check Exit
                m_exit = exit_pattern.search(line)
                if m_exit and current_trade:
                    current_trade["Exit Time"] = ts
                    current_trade["Reason"] = m_exit.group(1).strip()
                    current_trade["Exit Price"] = float(m_exit.group(2))
                    trades.append(current_trade) # Add to list, but keep ref for PnL
                    # Don't clear current_trade yet, PnL line comes next
                    continue
                    
                # Check PnL
                m_pnl = pnl_pattern.search(line)
                if m_pnl and current_trade and current_trade.get("Exit Time") is not pd.NaT:
                    try:
                        pnl_str = m_pnl.group(1).replace(",", "")
                        current_trade["PnL"] = float(pnl_str)
                    except: pass
                    current_trade = None # Reset
                    
    except Exception as e:
        st.error(f"Error parsing trades: {e}")
        
    df = pd.DataFrame(trades)
    if not df.empty:
        # Sort by Entry Time Desc
        df = df.sort_values("Entry Time", ascending=False)
        # Ensure UTC for linking
        if df['Entry Time'].dt.tz is None: df['Entry Time'] = df['Entry Time'].dt.tz_localize('UTC')
        if df['Exit Time'].dt.tz is None: df['Exit Time'] = df['Exit Time'].dt.tz_localize('UTC')
        
    return df

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
view = st.sidebar.radio("View", ["Cockpit", "Strategy Inspector", "Trade Log", "System Health"])
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
                pnl = (current_price - entry) * abs(pos) * cfg.STANDARD_LOT_SIZE
            else:
                pnl = (entry - current_price) * abs(pos) * cfg.STANDARD_LOT_SIZE

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
    bars_df = load_bars(300)
    trade_events = parse_trades_from_logs(strat_map)

    if bars_df is not None and not bars_df.empty:
        last_row = bars_df.iloc[-1]
        st.markdown(f"### {cfg.PRIMARY_TICKER} (Live)")
        st.markdown(f"**Last:** {last_row['close']:.5f} | **Vol:** {last_row['volume']:.0f} | **Aggr:** {last_row['net_aggressor_vol']:.0f}")

        fig = go.Figure(data=[go.Candlestick(x=bars_df['time_start'],
                        open=bars_df['open'],
                        high=bars_df['high'],
                        low=bars_df['low'],
                        close=bars_df['close'],
                        name=cfg.PRIMARY_TICKER)])

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
        st.plotly_chart(fig, width='stretch', key="main_chart")
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

elif view == "Trade Log":
    st.header("📜 Trade Log")
    
    strat_map = load_strategy_names()
    trades_df = parse_closed_trades(strat_map)
    
    if trades_df.empty:
        st.info("No closed trades found in logs.")
    else:
        # Display Table
        st.dataframe(
            trades_df[["Entry Time", "Type", "Strat", "Entry Price", "Exit Time", "Exit Price", "Reason", "PnL"]],
            width='stretch',
            height=300
        )
        
        st.divider()
        st.subheader("Trade Replay")
        
        # Selector
        # Create a label for the selectbox
        trades_df["label"] = trades_df.apply(
            lambda x: f"{x['Entry Time'].strftime('%m-%d %H:%M')} | {x['Type']} {x['Strat']} | PnL: ${x['PnL']:.2f}", axis=1
        )
        
        selected_label = st.selectbox("Select Trade to Visualize", trades_df["label"].tolist())
        
        if selected_label:
            trade = trades_df[trades_df["label"] == selected_label].iloc[0]
            
            col_del, col_vis = st.columns([1, 4])
            with col_del:
                if st.button("🗑️ Delete Trade", type="primary", key=f"del_{trade['Entry Time']}"):
                    try:
                        entry_str = trade["Entry Time"].strftime("%Y-%m-%d %H:%M:%S")
                        exit_str = trade["Exit Time"].strftime("%Y-%m-%d %H:%M:%S")
                        
                        with open(LOG_FILE, "r") as f:
                            lines = f.readlines()
                            
                        new_lines = []
                        deleted_count = 0
                        for line in lines:
                            keep = True
                            # Match Entry Line
                            if entry_str in line and "VIRTUAL ENTRY" in line:
                                keep = False
                            
                            # Match Exit/PnL/Cool Lines (usually share same second)
                            if exit_str in line and ("VIRTUAL EXIT" in line or "Trade PnL" in line or "cooling down" in line):
                                keep = False
                                
                            if keep:
                                new_lines.append(line)
                            else:
                                deleted_count += 1
                        
                        if deleted_count > 0:
                            with open(LOG_FILE, "w") as f:
                                f.writelines(new_lines)
                            st.toast(f"Deleted {deleted_count} log lines.", icon="🗑️")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.warning("Could not find matching log lines to delete.")
                            
                    except Exception as e:
                        st.error(f"Deletion failed: {e}")

            # Load Bars
            bars = load_bars(full_history=True)
            if bars is not None and not bars.empty:
                # Time Window: Entry - 2h to Exit + 2h
                t_start = trade["Entry Time"] - timedelta(hours=2)
                t_end = trade["Exit Time"] + timedelta(hours=2)
                
                # Filter
                mask = (bars['time_start'] >= t_start) & (bars['time_start'] <= t_end)
                view_bars = bars[mask]
                
                if not view_bars.empty:
                    fig = go.Figure(data=[go.Candlestick(
                        x=view_bars['time_start'],
                        open=view_bars['open'],
                        high=view_bars['high'],
                        low=view_bars['low'],
                        close=view_bars['close'],
                        name=cfg.PRIMARY_TICKER
                    )])
                    
                    # Markers
                    # Entry
                    color = "green" if trade["Type"] == "BUY" else "red"
                    symbol = "triangle-up" if trade["Type"] == "BUY" else "triangle-down"
                    fig.add_trace(go.Scatter(
                        x=[trade["Entry Time"]], y=[trade["Entry Price"]],
                        mode='markers', marker=dict(symbol=symbol, size=15, color=color),
                        name='Entry'
                    ))
                    
                    # Exit
                    fig.add_trace(go.Scatter(
                        x=[trade["Exit Time"]], y=[trade["Exit Price"]],
                        mode='markers', marker=dict(symbol='x', size=12, color='orange'),
                        name=f'Exit ({trade["Reason"]})'
                    ))
                    
                    # Lines
                    fig.add_hline(y=trade["Entry Price"], line_dash="solid", line_color=color, annotation_text="Entry")
                    fig.add_hline(y=trade["Exit Price"], line_dash="dash", line_color="orange", annotation_text=f"Exit {trade['Reason']}")
                    
                    # Title
                    fig.update_layout(
                        title=f"{trade['Type']} {trade['Strat']} (PnL: ${trade['PnL']:.2f})",
                        height=600,
                        template="plotly_dark",
                        xaxis_rangeslider_visible=False
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.warning("No bar data found for this trade period (Data might be rotated out).")

elif view == "System Health":
    st.header("🏥 System Health & Status")
    
    health_data = []
    
    # 1. Core System
    age, timestamp = get_file_age(STATE_FILE)
    state_status = "🟢" if age and age < 120 else "🔴"
    health_data.append(["State File", state_status, format_age(age), timestamp, "Core Bot State"])
    
    age, timestamp = get_file_age(BARS_FILE)
    if age and age < 120:
        bar_status = "🟢"
    elif state_status == "🟢" and age and age < 260000: # ~72 hours (Weekend Tolerance)
        bar_status = "🟡" # Process is alive, but no bars (Market Closed/Low Vol)
    else:
        bar_status = "🔴"
        
    health_data.append(["Live Bars", bar_status, format_age(age), timestamp, "Market Data Persistence"])

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

    st.divider()
    st.subheader("🕵️ Advanced Diagnostics")

    d_col1, d_col2 = st.columns(2)

    # 1. Feature Parity Check
    with d_col1:
        st.markdown("**Feature Parity (Live Data)**")
        # Load enough bars for feature computation (need > 200 for most, > 100 for Event Decay)
        bars = load_bars(limit=500)
        
        if bars is not None and len(bars) >= 200:
            # --- COMPUTE FEATURES ON THE FLY ---
            # We must run the pipeline to see if features *can* be generated.
            # The raw parquet file does NOT contain them.
            try:
                engine = FeatureEngine(os.path.join(BASE_DIR, "data"))
                engine.bars = bars.copy()
                
                # Setup Cache (Extract correlators from the raw bars themselves)
                def extract_series(name):
                    if name in engine.bars.columns:
                         return engine.bars[[name, 'time_start']].rename(columns={name: 'close', 'time_start': 'ts_event'})
                    return None
                    
                data_cache = {
                    'tnx': extract_series('TNX'),
                    'us2y': extract_series('US2Y'),
                    'zn': extract_series('ZN'),
                    'vix': extract_series('VIX'),
                    'ibit': extract_series('IBIT'),
                    'tick_nyse': extract_series('TICK_NYSE'),
                    'trin_nyse': extract_series('TRIN_NYSE')
                }
                
                # Run Pipeline
                run_pipeline(engine, data_cache)
                enriched_cols = engine.bars.columns
                
                checks = {
                    "Seasonality": "hour_sin" in enriched_cols,
                    "FRED (Macro)": "net_liq_zscore_60d" in enriched_cols,
                    "COT (Positioning)": any("cot_" in c for c in enriched_cols if "btc" not in c), 
                    "Event Decay": "bars_since_high_100" in enriched_cols,
                    "Deltas": "delta_velocity_50_25" in enriched_cols
                }
                
                all_passed = True
                for name, passed in checks.items():
                    icon = "✅" if passed else "❌"
                    st.write(f"{icon} {name}")
                    if not passed:
                        all_passed = False
                        st.caption(f"Missing columns for {name}. Strategies using this will fail.")
            except Exception as e:
                st.error(f"Pipeline Error: {e}")

        else:
            if bars is None:
                st.warning("No live data found.")
            else:
                st.warning(f"Insufficient data for parity check ({len(bars)}/200 bars).")

    # 2. Log Analysis
    with d_col2:
        st.markdown("**Log Health (Last 200 Lines)**")
        logs = load_logs(200)
        error_count = sum(1 for line in logs if "ERROR" in line)
        critical_count = sum(1 for line in logs if "CRITICAL" in line)
        warn_count = sum(1 for line in logs if "WARNING" in line)
        
        if critical_count > 0:
            st.error(f"🚨 {critical_count} CRITICAL Errors")
        else:
            st.write("✅ No Critical Errors")
            
        if error_count > 0:
            st.error(f"❌ {error_count} Errors")
        else:
            st.write("✅ No Standard Errors")
            
        if warn_count > 0:
            st.warning(f"⚠️ {warn_count} Warnings")
        else:
            st.write("✅ No Warnings")
            
        # Check for Specific Alerts
        if any("Gap detected" in line for line in logs):
             st.info("ℹ️ Gap Filling Active/Triggered recently")
        if any("Market Closed" in line for line in logs):
             st.info("ℹ️ Market Closed Logic Active")

    # 3. Strategy Health
    st.divider()
    st.markdown("**Strategy Health**")
    strategies = load_strategies_full()
    if strategies:
        st.write(f"Loaded {len(strategies)} Strategies.")
        # Check for parameter anomalies
        valid_strat_count = 0
        for s in strategies:
            # Simple check: Horizon should be set (not None) and reasonable
            if s.horizon is not None and s.horizon > 0:
                valid_strat_count += 1
            else:
                st.error(f"Strategy {s.name} has invalid Horizon: {s.horizon}")
        
        if valid_strat_count == len(strategies):
            st.success("All Strategies have valid parameters.")
        else:
            st.warning(f"Only {valid_strat_count}/{len(strategies)} strategies appear valid.")
    else:
        st.error("No Strategies Loaded! Portfolio is empty.")

# --- AUTO REFRESH LOGIC ---
if auto_refresh:
    time.sleep(5)
    st.rerun()
