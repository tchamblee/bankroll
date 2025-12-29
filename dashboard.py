import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os
import time
import re
from datetime import datetime, timezone, timedelta

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

# --- HELPER FUNCTIONS ---

def load_state():
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading state: {e}")
        return None

def load_bars(limit=300):
    if not os.path.exists(BARS_FILE):
        return None
    try:
        df = pd.read_parquet(BARS_FILE)
        # Ensure UTC
        if 'time_start' in df.columns:
            df['time_start'] = pd.to_datetime(df['time_start'], utc=True)
        return df.tail(limit)
    except Exception as e:
        st.error(f"Error loading bars: {e}")
        return None

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

def parse_trades_from_logs():
    """Parses paper_trade.log to extract recent Entry/Exit events for plotting."""
    if not os.path.exists(LOG_FILE):
        return []
    
    events = []
    # Regex patterns
    # Log format: 2025-12-29 10:00:00,123 - PaperTrade - INFO - 📝 VIRTUAL ENTRY: BUY 1 lots (Strat 5) @ 1.05000
    entry_pattern = re.compile(r"VIRTUAL ENTRY: (BUY|SELL) (\d+) lots \(Strat (\d+)\) @ ([\d\.]+)")
    exit_pattern = re.compile(r"VIRTUAL EXIT: (.*?) @ ([\d\.]+)")
    
    try:
        with open(LOG_FILE, "r") as f:
            for line in f:
                ts_str = line.split(",")[0] # Naive timestamp extraction "2023-..."
                try:
                    ts = pd.to_datetime(ts_str) # Allow pandas to guess format
                except:
                    continue # Skip if no timestamp
                
                # Check Entry
                m_entry = entry_pattern.search(line)
                if m_entry:
                    events.append({
                        "time": ts,
                        "type": "ENTRY",
                        "side": m_entry.group(1),
                        "lots": m_entry.group(2),
                        "strat": m_entry.group(3),
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
        # Don't fail the UI if log parsing fails
        pass
        
    return pd.DataFrame(events)

# --- MAIN UI ---

st.title("💸 Bankroll Paper Trade Cockpit")

# Sidebar
st.sidebar.header("Controls")
auto_refresh = st.sidebar.checkbox("Auto-Refresh (5s)", value=False)
if st.sidebar.button("Manual Refresh"):
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown("### System Status")

# --- HEALTH CHECK ---
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

# --- METRICS ROW ---

col1, col2, col3, col4 = st.columns(4)

if state:
    pos = state.get("position", 0)
    entry = state.get("entry_price", 0.0)
    strat_idx = state.get("active_strat_idx", -1)
    
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
        st.metric("Position", f"{pos} Lots", delta_color="off")
    with col2:
        st.metric("Entry Price", f"{entry:.5f}", delta=f"{current_price - entry:.5f}" if pos > 0 else f"{entry - current_price:.5f}")
    with col3:
        st.metric("Unrealized PnL (Est)", f"${pnl:.2f}", delta_color="normal")
    with col4:
        st.metric("Active Strategy", f"Index {strat_idx}")
else:
    col1.metric("Position", "OFFLINE")
    col2.metric("Entry Price", "---")
    col3.metric("Unrealized PnL", "---")
    col4.metric("Active Strategy", "---")

st.divider()

# --- CHARTS ---
st.subheader("Market View (EURUSD)")

bars_df = load_bars(300)
trade_events = parse_trades_from_logs()

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
        fig.add_hline(y=entry_price, line_dash="dash", line_color="blue", annotation_text="Active Entry")

    # Add Historical Trade Markers
    if not trade_events.empty:
        # Filter events to chart window
        min_time = bars_df['time_start'].min()
        # Ensure timezones match (naive vs UTC can be tricky, assuming logs are consistent with bars)
        # Convert events time to UTC if naive
        if trade_events['time'].dt.tz is None:
             trade_events['time'] = trade_events['time'].dt.tz_localize('UTC') # Log timestamps usually match system
             
        mask = trade_events['time'] >= min_time
        visible_events = trade_events[mask]
        
        # Plot Entries
        entries = visible_events[visible_events['type'] == 'ENTRY']
        if not entries.empty:
            # Buy
            buys = entries[entries['side'] == 'BUY']
            fig.add_trace(go.Scatter(
                x=buys['time'], y=buys['price'],
                mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'),
                name='Buy Entry'
            ))
            # Sell
            sells = entries[entries['side'] == 'SELL']
            fig.add_trace(go.Scatter(
                x=sells['time'], y=sells['price'],
                mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'),
                name='Sell Entry'
            ))
            
        # Plot Exits
        exits = visible_events[visible_events['type'] == 'EXIT']
        if not exits.empty:
            fig.add_trace(go.Scatter(
                x=exits['time'], y=exits['price'],
                mode='markers', marker=dict(symbol='x', size=10, color='orange'),
                name='Exit'
            ))

    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Waiting for bar data to generate chart...")

# --- RECENT TRADES TABLE ---
if not trade_events.empty:
    st.subheader("Recent Trade Events")
    # Show last 10
    st.dataframe(trade_events.sort_values("time", ascending=False).head(10), use_container_width=True)

# --- LOGS ---
st.subheader("System Logs")
logs = load_logs(30)
logs.reverse()  # Show most recent first
log_text = "\n".join(logs)
st.text_area("Live Logs", log_text, height=300, disabled=True)

# --- AUTO REFRESH LOGIC ---
if auto_refresh:
    time.sleep(5)
    st.rerun()
