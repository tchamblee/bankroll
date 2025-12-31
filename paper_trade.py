import asyncio
import logging
import os
import sys
import json
import shutil
import tempfile
import glob
import time
import subprocess
from pathlib import Path
import datetime as dt
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from ib_insync import *
import nest_asyncio

import config as cfg
from utils import setup_logging
from genome import Strategy
from feature_engine.core import FeatureEngine
from feature_engine import loader
from backtest.feature_computation import precompute_base_features, ensure_feature_context

nest_asyncio.apply()

# --------------------------------------------------------------------------------------
# SETUP & LOGGING
# --------------------------------------------------------------------------------------

logger = setup_logging("PaperTrade", "paper_trade.log")

def play_sound(sound_file):
    """Plays a sound file using mpg123 in a non-blocking subprocess."""
    if os.path.exists(sound_file):
        try:
            subprocess.Popen(['mpg123', '-q', sound_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            logger.error(f"Failed to play sound {sound_file}: {e}")

# Capture ib_insync and asyncio logs to the same file
try:
    file_handler = [h for h in logger.handlers if isinstance(h, logging.FileHandler)][0]
    logging.getLogger("ib_insync").addHandler(file_handler)
    logging.getLogger("asyncio").addHandler(file_handler)
    # Ensure they are at least INFO
    logging.getLogger("ib_insync").setLevel(logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.WARNING) # Asyncio can be too noisy at INFO
except Exception as e:
    logger.warning(f"Failed to attach file handler to libraries: {e}")

# --------------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------------

CLIENT_ID = 2
WINDOW_SIZE = 4000 
WARMUP_DAYS = 5
LIVE_STATE_FILE = os.path.join(cfg.DIRS['OUTPUT_DIR'], "live_state.json")
LIVE_BARS_FILE = os.path.join(cfg.DIRS['PROCESSED_DIR'], "live_bars.parquet")

# --------------------------------------------------------------------------------------
# CLASSES
# --------------------------------------------------------------------------------------

class LiveDataManager:
    def __init__(self):
        self.ticks_buffer = []
        self.current_vol = 0
        self.primary_bars = pd.DataFrame()
        self.external_cols = [t['name'] for t in cfg.TARGETS if t['name'] != 'EURUSD']
        self.correlator_snapshot = {name: np.nan for name in self.external_cols}
        self.gdelt_df = None
        
    def load_persistence(self):
        """Loads locally saved bars to restore state after restart."""
        if os.path.exists(LIVE_BARS_FILE):
            try:
                df = pd.read_parquet(LIVE_BARS_FILE)
                # Sort and Dedup
                df = df.sort_values('time_start').drop_duplicates(subset=['time_start'])
                # Ensure UTC
                if 'time_start' in df.columns:
                    df['time_start'] = pd.to_datetime(df['time_start'], utc=True)
                
                # Keep only window size
                if len(df) > WINDOW_SIZE:
                    df = df.iloc[-WINDOW_SIZE:]
                
                self.primary_bars = df.reset_index(drop=True)
                logger.info(f"💾 Loaded {len(self.primary_bars)} bars from persistence.")
                
                # Update Snapshot from last bar
                if not self.primary_bars.empty:
                    last_row = self.primary_bars.iloc[-1]
                    for col in self.external_cols:
                        if col in last_row and not pd.isna(last_row[col]):
                            self.correlator_snapshot[col] = last_row[col]
            except Exception as e:
                logger.error(f"❌ Failed to load persistence: {e}")

    def save_bar(self, bar_df):
        """Appends a new bar to the persistent store."""
        try:
            # We append by reading, concat, writing (simple but inefficient for high freq, okay for 5-min bars)
            # Better: Append mode? Parquet doesn't support simple append easily without partitioning.
            # Given WINDOW_SIZE=4000, rewriting is fast enough (<100ms).
            
            # Combine with memory state (which is already updated)
            # Actually, self.primary_bars ALREADY has the new bar.
            # So just save self.primary_bars.
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(LIVE_BARS_FILE), exist_ok=True)
            self.primary_bars.to_parquet(LIVE_BARS_FILE, index=False)
        except Exception as e:
            logger.error(f"❌ Failed to save persistence: {e}")

    def warmup(self, ib=None):
        """Loads history from disk + fetches missing gap from IBKR."""
        logger.info("🔥 Warming up...")
        
        # 1. Load GDELT (Always full reload as it's small daily file)
        try:
            self.gdelt_df = loader.load_gdelt_v2_data(cfg.DIRS["DATA_GDELT"])
            if self.gdelt_df is None:
                 self.gdelt_df = loader.load_gdelt_data(cfg.DIRS["DATA_GDELT"])
        except: pass

        # 2. Load Local Persistence
        self.load_persistence()
        
        # 3. Gap Filling (from IBKR)
        if ib and not self.primary_bars.empty:
            last_dt = self.primary_bars['time_end'].max()
            if last_dt.tzinfo is None: last_dt = last_dt.replace(tzinfo=timezone.utc)
            
            now_dt = datetime.now(timezone.utc)
            delta = now_dt - last_dt
            
            if delta.total_seconds() > 300: # If gap > 5 mins
                logger.info(f"⏳ Gap detected: {delta}. Fetching missing data from {last_dt}...")
                # We need to fetch Ticks to build Volume Bars to ensure consistency
                # Fetching Historical Ticks is slow.
                # Only do this if gap is small (< 4 hours). If gap is HUGE, maybe fetch Bars?
                # No, we need Volume Bars.
                pass 
                # Ideally we implement logic here to fetch historical ticks from IBKR
                # and run them through add_tick.
                # For now, we rely on the fact that if we restart quickly, the gap is small.
                # If gap is large, we might miss bars.
        
        # If still empty (first run), try loading from Raw Data Lake (Backfill)
        if self.primary_bars.empty:
            data_dir = cfg.DIRS["DATA_RAW_TICKS"]
            if os.path.exists(data_dir):
                files = sorted([f for f in os.listdir(data_dir) if "EURUSD" in f and f.endswith(".parquet")])
                if files:
                    recent = files[-WARMUP_DAYS:]
                    dfs = []
                    for f in recent:
                        try: dfs.append(pd.read_parquet(os.path.join(data_dir, f)))
                        except: pass
                    if dfs:
                        full = pd.concat(dfs, ignore_index=True).sort_values('ts_event')
                        from feature_engine.bars import create_volume_bars
                        bars = create_volume_bars(full, volume_threshold=cfg.VOLUME_THRESHOLD)
                        if bars is not None:
                            self.primary_bars = bars.tail(WINDOW_SIZE).reset_index(drop=True)
                            self.save_bar(None) # Init file

        # Ensure External Cols exist
        for col in self.external_cols:
            if col not in self.primary_bars.columns:
                self.primary_bars[col] = np.nan
                
        logger.info(f"✅ Warmup complete. State: {len(self.primary_bars)} bars.")

    def add_tick(self, tick_obj, target_conf):
        name = target_conf['name']
        if name == 'EURUSD':
            ts = tick_obj.time.replace(tzinfo=timezone.utc)
            
            # Calculate Tick Volume
            tick_vol = 0
            if isinstance(tick_obj, TickByTickBidAsk):
                # Use sum of sizes as proxy
                tick_vol = tick_obj.bidSize + tick_obj.askSize
                self.ticks_buffer.append({'ts_event': ts, 'pricebid': tick_obj.bidPrice, 'priceask': tick_obj.askPrice, 'sizebid': tick_obj.bidSize, 'sizeask': tick_obj.askSize, 'last_price': np.nan, 'last_size': np.nan})
            elif isinstance(tick_obj, (TickByTickAllLast,)):
                tick_vol = tick_obj.size
                self.ticks_buffer.append({'ts_event': ts, 'pricebid': np.nan, 'priceask': np.nan, 'sizebid': np.nan, 'sizeask': np.nan, 'last_price': tick_obj.price, 'last_size': tick_obj.size})
            
            self.current_vol += tick_vol
            
            if self.current_vol >= cfg.VOLUME_THRESHOLD:
                return self._create_bar()
        else:
            price = 0.0
            if hasattr(tick_obj, 'last') and tick_obj.last and not np.isnan(tick_obj.last): price = tick_obj.last
            elif hasattr(tick_obj, 'close') and tick_obj.close and not np.isnan(tick_obj.close): price = tick_obj.close
            elif hasattr(tick_obj, 'price') and tick_obj.price and not np.isnan(tick_obj.price): price = tick_obj.price
            elif hasattr(tick_obj, 'bidPrice') and tick_obj.bidPrice and not np.isnan(tick_obj.bidPrice): 
                price = (tick_obj.bidPrice + tick_obj.askPrice) / 2
            
            if price > 0: self.correlator_snapshot[name] = price
        return None

    def _create_bar(self):
        df = pd.DataFrame(self.ticks_buffer)
        self.ticks_buffer = []
        self.current_vol = 0
        if 'pricebid' in df.columns and 'priceask' in df.columns:
            df['mid_price'] = (df['pricebid'] + df['priceask']) / 2
            df['mid_price'] = df['mid_price'].fillna(df['last_price'])
        else: df['mid_price'] = df['last_price']

        vol = (df['sizebid'].fillna(0) + df['sizeask'].fillna(0)) if 'sizebid' in df.columns else df['last_size'].fillna(1)
        df['volume'] = np.where(vol == 0, 1, vol)
        df['net_aggressor_vol'] = np.sign(df['mid_price'].diff().fillna(0)) * df['volume']

        new_bar = {
            'time_start': df['ts_event'].iloc[0], 'time_end': df['ts_event'].iloc[-1],
            'open': df['mid_price'].iloc[0], 'high': df['mid_price'].max(), 'low': df['mid_price'].min(), 'close': df['mid_price'].iloc[-1],
            'volume': df['volume'].sum(), 'net_aggressor_vol': df['net_aggressor_vol'].sum(), 'tick_count': len(df)
        }
        
        # Inject latest snapshot prices
        for name, price in self.correlator_snapshot.items(): 
            new_bar[name] = price

        bar_df = pd.DataFrame([new_bar])
        self.primary_bars = pd.concat([self.primary_bars, bar_df], ignore_index=True)
        
        # Forward Fill missing external data (in case some tickers were quiet)
        self.primary_bars[self.external_cols] = self.primary_bars[self.external_cols].ffill()
        
        if len(self.primary_bars) > WINDOW_SIZE: self.primary_bars = self.primary_bars.iloc[-WINDOW_SIZE:].reset_index(drop=True)
        
        # PERSIST TO DISK
        self.save_bar(bar_df)
        
        return bar_df

    def compute_features_full(self):
        if len(self.primary_bars) < 200: return None
        engine = FeatureEngine(cfg.DIRS["DATA_DIR"])
        engine.bars = self.primary_bars.copy()
        
        # Define Windows (Must match Training!)
        windows = [25, 50, 100, 200, 400, 800, 1600, 3200]
        
        # 1. Standard Features
        engine.add_features_to_bars(windows=windows)
        engine.add_physics_features()
        engine.add_advanced_physics_features(windows=windows)
        engine.add_microstructure_features()
        engine.add_delta_features(lookback=25)
        
        # 2. External Data Integration
        # Create separate DataFrames for engine methods that expect them
        # Note: Engine methods typically merge on time_start or index. 
        # Here we just want to ensure the COLUMNS exist or are generated.
        
        # Extract series from the monolithic dataframe for the engine helpers
        def extract_series(name):
            if name in engine.bars.columns:
                return engine.bars[[name, 'time_start']].rename(columns={name: 'close', 'time_start': 'ts_event'}) # Hack to mimic ticker DF
            return None

        # Macro Voltage
        # Requires: US2Y, SCHATZ, TNX, BUND
        engine.add_macro_voltage_features(
            us2y_df=extract_series('US2Y'), 
            schatz_df=extract_series('SCHATZ'), 
            tnx_df=extract_series('TNX'), 
            bund_df=extract_series('BUND'), 
            windows=[50, 100]
        )
        
        # Crypto
        engine.add_crypto_features(extract_series('IBIT'))
        
        # Intermarket (ES, ZN, 6E)
        intermarket_dfs = {
            '_es': extract_series('ES'),
            '_zn': extract_series('ZN'),
            '_6e': extract_series('6E')
        }
        # Filter Nones
        intermarket_dfs = {k: v for k, v in intermarket_dfs.items() if v is not None}
        if intermarket_dfs:
            engine.add_intermarket_features(intermarket_dfs)
            
        # 3. GDELT
        if self.gdelt_df is not None:
            engine.add_gdelt_features(self.gdelt_df)

        if 'time_start' in engine.bars.columns:
            engine.bars['time_hour'] = engine.bars['time_start'].dt.hour
            engine.bars['time_weekday'] = engine.bars['time_start'].dt.dayofweek
            
        return engine.bars

class ExecutionEngine:
    def __init__(self, strategies: list):
        self.strategies = strategies
        self.position = 0 
        self.entry_price = 0.0
        self.active_strat_idx = -1
        self.current_sl = 0.0
        self.current_tp = 0.0
        self.current_atr = 0.0
        self.cooldowns = np.zeros(len(strategies), dtype=int)
        self.lot_size = cfg.STANDARD_LOT_SIZE
        
        self.balance = cfg.ACCOUNT_SIZE
        self.realized_pnl = 0.0
        
        # Load State
        self.load_state()
        
    def load_state(self):
        if os.path.exists(LIVE_STATE_FILE):
            try:
                with open(LIVE_STATE_FILE, 'r') as f:
                    state = json.load(f)
                    self.position = state.get('position', 0)
                    self.entry_price = state.get('entry_price', 0.0)
                    self.active_strat_idx = state.get('active_strat_idx', -1)
                    self.current_sl = state.get('current_sl', 0.0)
                    self.current_tp = state.get('current_tp', 0.0)
                    self.current_atr = state.get('current_atr', 0.0)
                    
                    self.balance = state.get('balance', cfg.ACCOUNT_SIZE)
                    self.realized_pnl = state.get('realized_pnl', 0.0)
                    
                    saved_cool = state.get('cooldowns', [])
                    if len(saved_cool) == len(self.cooldowns):
                        self.cooldowns = np.array(saved_cool, dtype=int)
                        
                logger.info(f"💾 Restored State: Pos={self.position} @ {self.entry_price:.5f} (Strat {self.active_strat_idx}) | Bal=${self.balance:.0f}")
            except Exception as e:
                logger.error(f"❌ Failed to load state: {e}")

    def check_reset(self):
        """Checks for external reset signal and wipes state if found."""
        trigger_path = os.path.join(cfg.DIRS['OUTPUT_DIR'], "RESET_TRIGGER")
        if os.path.exists(trigger_path):
            logger.warning("⚠️ RESET TRIGGER RECEIVED: Wiping Internal State...")
            
            # Reset Memory
            self.position = 0
            self.entry_price = 0.0
            self.active_strat_idx = -1
            self.current_sl = 0.0
            self.current_tp = 0.0
            self.current_atr = 0.0
            self.cooldowns = np.zeros(len(self.strategies), dtype=int)
            self.balance = cfg.ACCOUNT_SIZE
            self.realized_pnl = 0.0
            
            # Delete Trigger
            try: os.remove(trigger_path)
            except: pass
            
            # Save Clean State
            self.save_state()
            logger.warning("✅ State Reset Complete.")
            return True
        return False

    def save_state(self):
        try:
            state = {
                'position': int(self.position),
                'entry_price': float(self.entry_price),
                'active_strat_idx': int(self.active_strat_idx),
                'current_sl': float(self.current_sl),
                'current_tp': float(self.current_tp),
                'current_atr': float(self.current_atr),
                'cooldowns': self.cooldowns.tolist(),
                'balance': float(self.balance),
                'realized_pnl': float(self.realized_pnl),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            # Atomic Write
            with tempfile.NamedTemporaryFile('w', dir=os.path.dirname(LIVE_STATE_FILE), delete=False) as tf:
                json.dump(state, tf)
                temp_name = tf.name
            shutil.move(temp_name, LIVE_STATE_FILE)
        except Exception as e:
            logger.error(f"❌ Failed to save state: {e}")

    def calculate_commission(self, price, lots):
        # IBKR Tiered/Fixed approx: 0.2 bps or min $2.00
        # Value = Price * (Lots * StandardLotSize)
        value = price * abs(lots) * cfg.STANDARD_LOT_SIZE
        comm = value * (cfg.COST_BPS / 10000.0)
        return max(cfg.MIN_COMMISSION, comm)

    async def check_intraday_exits(self, current_price):
        if self.position == 0 or self.active_strat_idx == -1: return
        strat = self.strategies[self.active_strat_idx]
        if not hasattr(self, 'last_atr'): return
        sl_dist, tp_dist = self.last_atr * strat.stop_loss_pct, self.last_atr * strat.take_profit_pct
        exit_signal, reason = False, ""
        
        # Check Long
        if self.position > 0:
            if current_price <= (self.entry_price - sl_dist): exit_signal, reason = True, "SL"
            elif current_price >= (self.entry_price + tp_dist): exit_signal, reason = True, "TP"
        # Check Short
        else:
            if current_price >= (self.entry_price + sl_dist): exit_signal, reason = True, "SL"
            elif current_price <= (self.entry_price - tp_dist): exit_signal, reason = True, "TP"
            
        if exit_signal:
            logger.info(f"📝 VIRTUAL EXIT: {reason} @ {current_price:.5f}")
            play_sound("resources/exit.mp3")
            
            # --- PnL Calculation ---
            try:
                raw_pnl = (current_price - self.entry_price) * self.position * cfg.STANDARD_LOT_SIZE
                comm = self.calculate_commission(self.entry_price, self.position) + \
                       self.calculate_commission(current_price, self.position)
                net_pnl = raw_pnl - comm
                
                self.balance += net_pnl
                self.realized_pnl += net_pnl
                logger.info(f"💰 Trade PnL: ${net_pnl:.2f} | Bal: ${self.balance:.2f}")
            except Exception as e:
                logger.error(f"❌ PnL Calculation Error: {e}")
            # -----------------------

            # Capture index BEFORE resetting
            idx_to_cool = self.active_strat_idx
            
            self.position, self.active_strat_idx, self.entry_price = 0, -1, 0.0
            self.current_sl, self.current_tp = 0.0, 0.0
            
            if reason == "SL":
                self.cooldowns[idx_to_cool] = cfg.STOP_LOSS_COOLDOWN_BARS
                logger.info(f"❄️ Strategy {idx_to_cool} cooling down for {cfg.STOP_LOSS_COOLDOWN_BARS} bars")
            
            self.save_state()

    def decrement_cooldowns(self):
        self.cooldowns = np.maximum(0, self.cooldowns - 1)
        # Only save if changed? Maybe too frequent. Save on trade/exit is most critical.
        # But if we crash during cooldown, we want to remember it.
        # Save every bar? It's cheap.
        # self.save_state() # Let's do it in process_new_bar instead

    async def execute_signal(self, signal, strat_idx, price, atr):
        self.last_atr = atr
        if signal == self.position:
            # Ensure SL/TP are hydrated for the dashboard even on restored positions
            if self.position != 0 and (self.current_sl == 0.0 or self.current_tp == 0.0):
                try:
                    strat = self.strategies[self.active_strat_idx]
                    sl_dist = atr * strat.stop_loss_pct
                    tp_dist = atr * strat.take_profit_pct
                    if self.position > 0:
                        self.current_sl = self.entry_price - sl_dist
                        self.current_tp = self.entry_price + tp_dist
                    else:
                        self.current_sl = self.entry_price + sl_dist
                        self.current_tp = self.entry_price - tp_dist
                    self.save_state()
                except Exception as e:
                    logger.error(f"❌ Failed to hydrate SL/TP on restore: {e}")
            return
        
        state_changed = False
        
        if self.position != 0:
            logger.info(f"📝 VIRTUAL EXIT: Signal Reversal @ {price:.5f}")
            play_sound("resources/exit.mp3")
            
            # --- PnL Calculation ---
            try:
                raw_pnl = (price - self.entry_price) * self.position * cfg.STANDARD_LOT_SIZE
                comm = self.calculate_commission(self.entry_price, self.position) + \
                       self.calculate_commission(price, self.position)
                net_pnl = raw_pnl - comm
                
                self.balance += net_pnl
                self.realized_pnl += net_pnl
                logger.info(f"💰 Trade PnL: ${net_pnl:.2f} | Bal: ${self.balance:.2f}")
            except Exception as e:
                logger.error(f"❌ PnL Calculation Error: {e}")
            # -----------------------

            self.position = 0
            self.current_sl, self.current_tp = 0.0, 0.0
            state_changed = True
            
        if signal != 0:
            action = 'BUY' if signal > 0 else 'SELL'
            logger.info(f"📝 VIRTUAL ENTRY: {action} {abs(signal)} lots (Strat {strat_idx}) @ {price:.5f}")
            play_sound("resources/entry.mp3")
            self.position, self.active_strat_idx, self.entry_price = signal, strat_idx, price
            
            # Calculate SL/TP
            strat = self.strategies[strat_idx]
            sl_dist = atr * strat.stop_loss_pct
            tp_dist = atr * strat.take_profit_pct
            
            if signal > 0: # Long
                self.current_sl = price - sl_dist
                self.current_tp = price + tp_dist
            else: # Short
                self.current_sl = price + sl_dist
                self.current_tp = price - tp_dist
                
            state_changed = True
            
        if state_changed:
            self.save_state()

class PaperTradeApp:
    def __init__(self):
        self.ib = IB()
        self.data_manager = LiveDataManager()
        self.strategies = []
        self.executor = None
        self.stop_event = asyncio.Event()
        self.contract_map = {}
        
        # Temp dir for JIT feature calculation (JIT Parity)
        self.temp_dir = tempfile.mkdtemp(prefix="paper_trade_jit_")
        self.existing_keys = set()
        logger.info(f"📁 Created Temp Context: {self.temp_dir}")
        
    def __del__(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def load_strategies(self):
        path = os.path.join(cfg.DIRS['STRATEGIES_DIR'], "mutex_portfolio.json")
        if not os.path.exists(path): return False
        with open(path, 'r') as f:
            data = json.load(f)
        for d in data:
            try:
                s = Strategy.from_dict(d)
                s.horizon, s.stop_loss_pct, s.take_profit_pct = d.get('horizon', 120), d.get('stop_loss_pct', 2.0), d.get('take_profit_pct', 4.0)
                self.strategies.append(s)
            except: pass
        return True

    async def run(self):
        try:
            if not self.load_strategies():
                logger.warning("⚠️ No strategies found. Running in DATA ONLY mode (Accumulating Bars).")
            else:
                logger.info(f"✅ Loaded {len(self.strategies)} strategies.")
                
            self.data_manager.warmup()
            await self.ib.connectAsync(cfg.IBKR_HOST, cfg.IBKR_PORT, clientId=CLIENT_ID)
            self.executor = ExecutionEngine(self.strategies)
            self.ib.reqMarketDataType(3)
            eur = Forex('EURUSD')
            await self.ib.qualifyContractsAsync(eur)
            self.ib.reqTickByTickData(eur, "BidAsk", 0, False)
            self.contract_map[eur.conId] = [t for t in cfg.TARGETS if t['name']=='EURUSD'][0]
            for t in cfg.TARGETS:
                if t['name'] == 'EURUSD': continue
                c = None
                if t['secType'] == 'IND': c = Index(t['symbol'], t['exchange'], t['currency'])
                elif t['secType'] == 'CONTFUT': c = ContFuture(t['symbol'], t['exchange'], t['currency'])
                elif t['secType'] == 'STK': c = Stock(t['symbol'], t['exchange'], t['currency'])
                elif t['secType'] == 'CASH': c = Forex(t['symbol'] + t['currency'])
                if c:
                    await self.ib.qualifyContractsAsync(c)
                    self.ib.reqMktData(c, "", False, False)
                    self.contract_map[c.conId] = t
            self.ib.pendingTickersEvent += self.on_pending_tickers
            logger.info("🟢 VIRTUAL PAPER TRADING ACTIVE (NO BROKER ORDERS)")
            
            disconnect_time = None
            last_state_update = datetime.now()
            
            while not self.stop_event.is_set():
                # Heartbeat
                Path("logs/paper_trade_heartbeat").touch()

                if self.executor: 
                    self.executor.check_reset()
                    
                    # Periodic State Flush (Keep Dashboard Green)
                    if (datetime.now() - last_state_update).total_seconds() > 60:
                        self.executor.save_state()
                        last_state_update = datetime.now()
                
                if not self.ib.isConnected():
                    if disconnect_time is None:
                        disconnect_time = datetime.now()
                        logger.warning("⚠️ IBKR Disconnected!")
                    elif (datetime.now() - disconnect_time).total_seconds() > 60:
                         logger.error("💀 Disconnected for > 60s. Exiting for restart.")
                         self.stop_event.set()
                         sys.exit(1)
                else:
                    disconnect_time = None
                    
                await asyncio.sleep(1)
        finally:
            self.ib.disconnect()
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info("🧹 Cleaned up temp dir")

    def on_pending_tickers(self, tickers):
        for t in tickers:
            conf = self.contract_map.get(t.contract.conId)
            if not conf: continue
            if conf['name'] == 'EURUSD':
                price = (t.bid + t.ask)/2 if t.bidSize > 0 and t.askSize > 0 else (t.last if t.last > 0 else 0)
                if price > 0: asyncio.create_task(self.executor.check_intraday_exits(price))
                if t.tickByTicks:
                    for tick in t.tickByTicks:
                        bar = self.data_manager.add_tick(tick, conf)
                        if bar is not None: asyncio.create_task(self.process_new_bar())
            else: self.data_manager.add_tick(t, conf)

    async def process_new_bar(self):
        # 1. Decrement cooldowns at the start of each new bar
        self.executor.decrement_cooldowns()
        self.executor.save_state()

        full_df = self.data_manager.compute_features_full()
        if full_df is None: return
        
        # --- JIT FEATURE COMPUTATION (PARITY FIX) ---
        # 1. Dump Base Features to Disk
        # We need to ensure existing_keys tracks what we have
        self.existing_keys.clear() # Reset for each bar to ensure fresh calculation on new data? 
        # Actually, if we use a temp dir, we overwrite files.
        # Efficient approach: Just clear tracking set and let precompute overwrite.
        
        precompute_base_features(full_df, self.temp_dir, self.existing_keys)
        
        # 2. Compute Derived Features (Z-Score, Slope, etc.) needed by strategies
        ensure_feature_context(self.strategies, self.temp_dir, self.existing_keys)
        
        # 3. Load Context for Signals
        # Strategies expect a dict: name -> np.array
        # We only load what's needed? Or everything?
        # Optimization: Only load what strategies use. But simpler to load keys from existing_keys
        context = {}
        for key in self.existing_keys:
            try:
                arr = np.load(os.path.join(self.temp_dir, f"{key}.npy"))
                if len(arr) == 0:
                    logger.error(f"⚠️ FOUND EMPTY FEATURE: {key} shape={arr.shape}")
                context[key] = arr
            except: pass
            
        context['__len__'] = len(full_df)
        if context['__len__'] > 0 and any(len(v) == 0 for v in context.values() if isinstance(v, np.ndarray)):
             logger.error(f"CRITICAL: Context has {context['__len__']} rows but contains empty features!")
            
        # Use precomputed ATR if available
        atr_vec = context.get('atr_base')
        if atr_vec is not None and len(atr_vec) > 0:
            atr = float(atr_vec[-1])
        else:
            atr = full_df['close'].iloc[-1] * 0.001 # Fallback 10 bps

        self.executor.current_atr = atr

        # --- ENFORCE TRADING HOURS ---
        last_ts = full_df['time_start'].iloc[-1]
        if not hasattr(last_ts, 'hour'): last_ts = pd.to_datetime(last_ts)
        
        current_hour = last_ts.hour
        current_weekday = last_ts.dayofweek
        
        # Market Open: Mon-Fri, StartHour <= H < EndHour
        is_market_open = (current_hour >= cfg.TRADING_START_HOUR) and \
                         (current_hour < cfg.TRADING_END_HOUR) and \
                         (current_weekday < 5)
                         
        if not is_market_open:
            if self.executor.position != 0:
                logger.info(f"🚫 Market Closed (Hour {current_hour}). Forcing Exit.")
                await self.executor.execute_signal(0, -1, full_df['close'].iloc[-1], atr)
            return # Skip signal generation
        # -----------------------------
        
        active_sig, active_idx = 0, -1
        
        for i, strat in enumerate(self.strategies):
            if self.executor.cooldowns[i] > 0: continue
            try:
                sig_vec = strat.generate_signal(context)
                if sig_vec[-1] != 0:
                    active_sig, active_idx = sig_vec[-1], i
                    break
            except Exception as e: 
                logger.error(f"Strategy {i} Error: {e}")
        
        if active_sig != 0:
            await self.executor.execute_signal(active_sig, active_idx, full_df['close'].iloc[-1], atr)

if __name__ == "__main__":
    app = PaperTradeApp()
    try: asyncio.run(app.run())
    except KeyboardInterrupt: logger.info("Exiting...")
