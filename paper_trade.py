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
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from ib_insync import *
import nest_asyncio

import config as cfg
from utils import setup_logging
from genome import Strategy
from backtest.strategy_loader import load_strategies as load_strategies_from_disk
from feature_engine.core import FeatureEngine
from feature_engine import loader
from feature_engine.pipeline import run_pipeline
from backtest.feature_computation import precompute_base_features, ensure_feature_context

nest_asyncio.apply()

# --------------------------------------------------------------------------------------
# SETUP & LOGGING
# --------------------------------------------------------------------------------------

logger = setup_logging("PaperTrade", "paper_trade.log")

def play_sound(sound_file):
    """Plays a sound file using mpg123 in a non-blocking subprocess if enabled."""
    if not cfg.ENABLE_SOUND:
        return
        
    if os.path.exists(sound_file):
        try:
            # Use a timeout-capable call or just ensure we don't spawn if not needed.
            # On Linux/WSL, mpg123 -q can hang if no audio device is found.
            subprocess.Popen(['mpg123', '-q', sound_file], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL,
                             start_new_session=True) # Decouple from parent
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

CLIENT_ID = cfg.IBKR_CLIENT_ID_PAPER
WINDOW_SIZE = 4000 
WARMUP_DAYS = 7
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
        self.external_cols = [t['name'] for t in cfg.TARGETS if t['name'] != cfg.PRIMARY_TICKER]
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

    async def fill_gaps(self, ib, contract):
        """Fetches missing ticks from IBKR to bridge the gap since last shutdown."""
        if self.primary_bars.empty: return

        last_dt = self.primary_bars['time_end'].max()
        if last_dt.tzinfo is None: last_dt = last_dt.replace(tzinfo=timezone.utc)
        
        now_dt = datetime.now(timezone.utc)
        delta = now_dt - last_dt
        
        if delta.total_seconds() < 300: return # No significant gap
        
        logger.info(f"⏳ Gap detected: {delta}. Fetching missing ticks from {last_dt}...")
        
        # Limit to last 2 hours to prevent massive delays
        start_time = last_dt
        if delta.total_seconds() > 7200:
            logger.warning("⚠️ Gap > 2 hours. Only backfilling last 2 hours.")
            start_time = now_dt - timedelta(hours=2)
            
        fetched_count = 0
        
        # Loop until caught up (within 30s of now)
        logger.info(f"Starting gap fill loop from {start_time} to {now_dt}...")
        while start_time < (now_dt - timedelta(seconds=30)):
            # Format: YYYYMMDD HH:MM:SS UTC (Explicit Timezone to prevent Future Data error)
            formatted_time = start_time.strftime("%Y%m%d %H:%M:%S") + " UTC"
            logger.info(f"  -> Requesting 1000 ticks from {formatted_time}...")
            try:
                # Use 'BidAsk' for ticks
                ticks = await ib.reqHistoricalTicksAsync(
                    contract, startDateTime=formatted_time, 
                    endDateTime="", numberOfTicks=1000, 
                    whatToShow='BID_ASK', useRth=False, ignoreSize=False
                )
            except Exception as e:
                logger.error(f"❌ Failed to fetch ticks batch: {e}")
                break
            except asyncio.CancelledError:
                logger.warning("⚠️ Tick fetch cancelled (Disconnect?). Stopping fill.")
                break
                
            if not ticks: 
                logger.info("  -> No more ticks returned.")
                break
            
            # Simple Wrapper to match TickByTickBidAsk interface
            class TickWrapper:
                def __init__(self, t):
                    self.time = t.time
                    self.bidPrice = t.priceBid
                    self.askPrice = t.priceAsk
                    self.bidSize = t.sizeBid
                    self.askSize = t.sizeAsk

            for t in ticks:
                wrapper = TickWrapper(t)
                self.add_tick(wrapper, {'name': cfg.PRIMARY_TICKER})
                
            fetched_count += len(ticks)
            # Update start_time to last tick time for next batch
            last_tick_time = ticks[-1].time
            if last_tick_time.tzinfo is None:
                last_tick_time = last_tick_time.replace(tzinfo=timezone.utc)
            start_time = last_tick_time
            
            logger.info(f"  -> Fetched {len(ticks)} ticks. Next start: {start_time}")
            
            # Brief pause to respect rate limits
            await asyncio.sleep(0.05)
            
        logger.info(f"✅ Gap Filled. Fetched {fetched_count} ticks. New Head: {self.primary_bars['time_end'].max()}")

    def warmup(self, ib=None):
        """Loads history from disk."""
        logger.info("🔥 Warming up...")
        
        # 1. Load GDELT (Always full reload as it's small daily file)
        try:
            self.gdelt_df = loader.load_gdelt_v2_data(cfg.DIRS["DATA_GDELT"])
            if self.gdelt_df is None:
                 self.gdelt_df = loader.load_gdelt_data(cfg.DIRS["DATA_GDELT"])
        except: pass

        # 2. Load Local Persistence
        self.load_persistence()
        
        # 3. Gap Filling (moved to fill_gaps called after connection)
        
        # If still empty (first run), try loading from Raw Data Lake (Backfill)
        if self.primary_bars.empty:
            data_dir = cfg.DIRS["DATA_RAW_TICKS"]
            if os.path.exists(data_dir):
                files = sorted([f for f in os.listdir(data_dir) if cfg.PRIMARY_TICKER in f and f.endswith(".parquet")])
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
        if name == cfg.PRIMARY_TICKER:
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
        
        # Build Data Cache from current bars (Live Mode)
        def extract(name):
            if name in engine.bars.columns:
                return engine.bars[[name, 'time_start']].rename(columns={name: 'close', 'time_start': 'ts_event'})
            return None
            
        data_cache = {
            'tnx': extract('TNX'),
            'usdchf': extract('USDCHF'),
            'bund': extract('BUND'),
            'us2y': extract('US2Y'),
            'schatz': extract('SCHATZ'),
            'es': extract('ES'),
            'zn': extract('ZN'),
            '6e': extract('6E'),
            'ibit': extract('IBIT'),
            # Note: TICK/TRIN usually don't have snapshot prices in paper_trade yet, but if they did:
            'tick_nyse': extract('TICK_NYSE'),
            'trin_nyse': extract('TRIN_NYSE'),
            'gdelt': self.gdelt_df
        }
        
        # Run Shared Pipeline
        run_pipeline(engine, data_cache)

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
            # Ensure SL/TP are hydrated
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
        
        # --- CLOSE EXISTING POSITION ---
        if self.position != 0:
            logger.info(f"📝 VIRTUAL EXIT: Signal Reversal @ {price:.5f}")
            play_sound("resources/exit.mp3")
            
            # --- PnL Calculation ---
            try:
                # Calculate PnL on the SPECIFIC SIZE held
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
            
        # --- OPEN NEW POSITION ---
        if signal != 0:
            # VOLATILITY TARGETING SIZING
            strat = self.strategies[strat_idx]
            eff_sl = strat.stop_loss_pct if strat.stop_loss_pct > 0 else 1.0
            eff_atr = atr if atr > 1e-6 else 1e-6
            
            target_risk_dollars = self.balance * cfg.RISK_PER_TRADE_PERCENT
            
            # Size = Risk / (LotSize * SL_Mult * ATR)
            calc_size = target_risk_dollars / (cfg.STANDARD_LOT_SIZE * eff_sl * eff_atr)
            
            # Clamp
            final_size = min(max(calc_size, cfg.MIN_LOTS), cfg.MAX_LOTS)
            
            # Apply Direction
            signed_size = np.sign(signal) * final_size
            
            action = 'BUY' if signal > 0 else 'SELL'
            logger.info(f"📝 VIRTUAL ENTRY: {action} {final_size:.2f} lots (Risk ${target_risk_dollars:.0f}) Strat {strat.name} @ {price:.5f}")
            play_sound("resources/entry.mp3")
            
            self.position = signed_size # Store actual size (e.g. 2.5 or -0.5)
            self.active_strat_idx = strat_idx
            self.entry_price = price
            
            # Calculate SL/TP
            sl_dist = atr * strat.stop_loss_pct
            tp_dist = atr * strat.take_profit_pct
            
            if signed_size > 0: # Long
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
        self.primary_contract = None
        self.last_eur_tick = datetime.now()
        
        # Temp dir for JIT feature calculation (JIT Parity)
        self.temp_dir = tempfile.mkdtemp(prefix="paper_trade_jit_")
        self.existing_keys = set()
        logger.info(f"📁 Created Temp Context: {self.temp_dir}")
        
    def __del__(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def load_strategies(self):
        try:
            # Shared loader handles path, json parsing, and hydration
            self.strategies, _ = load_strategies_from_disk(source_type='mutex', load_metrics=False)
            return bool(self.strategies)
        except Exception as e:
            logger.error(f"Failed to load strategies via shared loader: {e}")
            return False

    def _build_signal_context(self):
        """
        Helper: Computes features and builds the execution context (JIT).
        Returns: (full_df, context, atr) or (None, None, None)
        """
        full_df = self.data_manager.compute_features_full()
        if full_df is None: return None, None, None
        
        # --- JIT FEATURE COMPUTATION ---
        # 1. Dump Base Features to Disk
        self.existing_keys.clear()
        precompute_base_features(full_df, self.temp_dir, self.existing_keys)
        
        # 2. Compute Derived Features (Z-Score, Slope, etc.) needed by strategies
        ensure_feature_context(self.strategies, self.temp_dir, self.existing_keys)
        
        # 3. Load Context
        context = {}
        for key in self.existing_keys:
            try:
                arr = np.load(os.path.join(self.temp_dir, f"{key}.npy"))
                context[key] = arr
            except: pass
            
        context['__len__'] = len(full_df)
        
        # ATR
        atr_vec = context.get('atr_base')
        if atr_vec is not None and len(atr_vec) > 0:
            atr = float(atr_vec[-1])
        else:
            atr = full_df['close'].iloc[-1] * 0.001 

        # Enforce Floor (MIN_ATR_BPS)
        min_atr = full_df['close'].iloc[-1] * (cfg.MIN_ATR_BPS / 10000.0)
        atr = max(atr, min_atr)

        return full_df, context, atr

    def validate_strategies(self):
        """
        SMOKE TEST: Runs a dry pass of all strategies against the current feature set.
        Detects missing features (Parity Gap) before trading starts.
        """
        logger.info("🕵️ Running Strategy Feature Validation (Smoke Test)...")
        if len(self.data_manager.primary_bars) < 200:
            logger.warning("⚠️ Not enough data for validation (Need > 200 bars). Skipping.")
            return

        full_df, context, _ = self._build_signal_context()
        if context is None:
            logger.error("❌ Validation Failed: Could not build feature context.")
            return

        failing_strategies = []
        for i, strat in enumerate(self.strategies):
            try:
                # Dry Run
                strat.generate_signal(context)
            except KeyError as e:
                logger.critical(f"🚨 STRATEGY FAILURE: '{strat.name}' is missing feature {e}!")
                failing_strategies.append(i)
            except Exception as e:
                logger.error(f"❌ Strategy '{strat.name}' failed validation: {e}")
                failing_strategies.append(i)
        
        if failing_strategies:
            logger.critical(f"🛑 {len(failing_strategies)} STRATEGIES FAILED VALIDATION. REMOVING THEM.")
            # Remove backwards to keep indices valid
            for i in sorted(failing_strategies, reverse=True):
                self.strategies.pop(i)
                self.executor.strategies = self.strategies # Sync executor
                self.executor.cooldowns = np.delete(self.executor.cooldowns, i)
        else:
            logger.info("✅ All strategies passed feature validation.")

    async def run(self):
        exit_code = 0
        try:
            if not self.load_strategies():
                logger.warning("⚠️ No strategies found. Running in DATA ONLY mode (Accumulating Bars).")
            else:
                logger.info(f"✅ Loaded {len(self.strategies)} strategies.")
                
            self.data_manager.warmup()
            
            # --- VALIDATION ---
            self.validate_strategies()
            # ------------------
            
            await self.ib.connectAsync(cfg.IBKR_HOST, cfg.IBKR_PORT, clientId=CLIENT_ID)
            self.executor = ExecutionEngine(self.strategies) # Re-init in case strategies were dropped
            self.ib.reqMarketDataType(1)
            if not self.primary_contract:
                # Fallback if qualification failed or config issue
                logger.warning(f"⚠️ Primary Contract ({cfg.PRIMARY_TICKER}) not qualified. Attempting manual...")
                eur = Forex(cfg.PRIMARY_TICKER)
                await self.ib.qualifyContractsAsync(eur)
                self.primary_contract = eur
                self.contract_map[eur.conId] = [t for t in cfg.TARGETS if t['name']==cfg.PRIMARY_TICKER][0]
                
                # --- SUBSCRIBE PRIMARY TICKER ---
                self.ib.reqMktData(eur, "", False, False)
                self.ib.reqTickByTickData(eur, "BidAsk", numberOfTicks=0, ignoreSize=False)
                logger.info(f"   ✅ Subscribed Primary Ticker: {eur.localSymbol}")
                
                # --- GAP FILLING ---
                await self.data_manager.fill_gaps(self.ib, eur)
                # -------------------
            
            for t in cfg.TARGETS:
                if t['name'] == cfg.PRIMARY_TICKER: continue # Handled below
                
                if t['mode'] == 'BARS_TRADES_1MIN':
                    c = None
                    if t['secType'] == 'IND': c = Index(t['symbol'], t['exchange'], t['currency'])
                    elif t['secType'] == 'CONTFUT': c = ContFuture(t['symbol'], t['exchange'], t['currency'])
                    elif t['secType'] == 'STK': c = Stock(t['symbol'], t['exchange'], t['currency'])
                    elif t['secType'] == 'CASH': c = Forex(t['symbol'] + t['currency'])
                    if c:
                        try:
                            await self.ib.qualifyContractsAsync(c)
                            self.ib.reqMktData(c, "", False, False)
                            self.contract_map[c.conId] = t
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to qualify/request {t['name']}: {e}")
                        
            self.ib.pendingTickersEvent += self.on_pending_tickers
            logger.info("🟢 VIRTUAL PAPER TRADING ACTIVE (NO BROKER ORDERS)")
            
            disconnect_time = None
            last_state_update = datetime.now()
            self.last_eur_tick = datetime.now() # Reset on start
            
            while not self.stop_event.is_set():
                # Heartbeat
                Path("logs/paper_trade_heartbeat").touch()

                if self.executor: 
                    self.executor.check_reset()
                    
                    # Periodic State Flush (Keep Dashboard Green)
                    if (datetime.now() - last_state_update).total_seconds() > 60:
                        self.executor.save_state()
                        last_state_update = datetime.now()
                
                # Check for Data Staleness ({cfg.PRIMARY_TICKER} specifically)
                stall_delta = (datetime.now() - self.last_eur_tick).total_seconds()
                if stall_delta > 900: # 15 mins
                    logger.error(f"💀 {cfg.PRIMARY_TICKER} Data Stalled (>{stall_delta:.0f}s). Exiting for restart.")
                    sys.exit(1) # Supervisor will restart
                elif stall_delta > 60 and int(stall_delta) % 60 == 0:
                    logger.warning(f"⚠️ {cfg.PRIMARY_TICKER} Idle for {stall_delta:.0f}s (Market Closed/Slow?)")

                if not self.ib.isConnected():
                    if disconnect_time is None:
                        disconnect_time = datetime.now()
                        logger.warning("⚠️ IBKR Disconnected!")
                    elif (datetime.now() - disconnect_time).total_seconds() > 60:
                         logger.error("💀 Disconnected for > 60s. Exiting for restart.")
                         self.stop_event.set()
                         exit_code = 1
                         break
                else:
                    disconnect_time = None
                    
                await asyncio.sleep(1)
        except BaseException as e:
            logger.error(f"🔥 CRITICAL FAILURE in paper_trade loop: {e}", exc_info=True)
            exit_code = 1
        finally:
            self.ib.disconnect()
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info("🧹 Cleaned up temp dir")
            return exit_code

    def on_pending_tickers(self, tickers):
        for t in tickers:
            conf = self.contract_map.get(t.contract.conId)
            if not conf: continue
            if conf['name'] == cfg.PRIMARY_TICKER:
                self.last_eur_tick = datetime.now()
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

        full_df, context, atr = self._build_signal_context()
        if full_df is None or context is None: return

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
            except KeyError as e:
                logger.critical(f"🚨 RUNTIME MISSING FEATURE: {e} in Strategy {strat.name}")
            except Exception as e: 
                logger.error(f"Strategy {i} Error: {e}")
        
        if active_sig != 0:
            await self.executor.execute_signal(active_sig, active_idx, full_df['close'].iloc[-1], atr)

if __name__ == "__main__":
    app = PaperTradeApp()
    exit_code = 0
    try: 
        exit_code = asyncio.run(app.run())
    except KeyboardInterrupt: 
        logger.info("Exiting...")
    sys.exit(exit_code)
