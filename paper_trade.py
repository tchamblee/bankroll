import asyncio
import logging
import os
import sys
import json
import datetime as dt
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from ib_insync import *
import nest_asyncio

import config as cfg
from genome import Strategy
from feature_engine.core import FeatureEngine

nest_asyncio.apply()

# --------------------------------------------------------------------------------------
# SETUP & LOGGING
# --------------------------------------------------------------------------------------

LOG_DIR = cfg.DIRS.get("LOGS", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "paper_trade.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("PaperTrade")

# --------------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------------

CLIENT_ID = 2
VOLUME_THRESHOLD = 250
WINDOW_SIZE = 4000 
WARMUP_DAYS = 5

# --------------------------------------------------------------------------------------
# CLASSES
# --------------------------------------------------------------------------------------

class LiveDataManager:
    def __init__(self):
        self.ticks_buffer = []
        self.primary_bars = pd.DataFrame()
        self.correlator_snapshot = {t['name']: np.nan for t in cfg.TARGETS if t['name'] != 'EURUSD'}
        
    def warmup(self):
        """Loads historical ticks to pre-fill bars."""
        logger.info("🔥 Warming up with historical data...")
        data_dir = cfg.DIRS["DATA_RAW_TICKS"]
        if not os.path.exists(data_dir): return

        files = [f for f in os.listdir(data_dir) if "EURUSD" in f and f.endswith(".parquet")]
        files.sort()
        recent_files = files[-WARMUP_DAYS:]
        if not recent_files: return
            
        dfs = []
        for f in recent_files:
            try: dfs.append(pd.read_parquet(os.path.join(data_dir, f)))
            except: pass
            
        if not dfs: return
        full_ticks = pd.concat(dfs, ignore_index=True)
        if 'ts_event' in full_ticks.columns:
            full_ticks = full_ticks.sort_values('ts_event')
            
        from feature_engine.bars import create_volume_bars
        bars = create_volume_bars(full_ticks, volume_threshold=VOLUME_THRESHOLD)
        if bars is not None and not bars.empty:
            self.primary_bars = bars.tail(WINDOW_SIZE).reset_index(drop=True)
            logger.info(f"  ✅ Warmup complete. Loaded {len(self.primary_bars)} bars.")

    def add_tick(self, tick_obj, target_conf):
        name = target_conf['name']
        if name == 'EURUSD':
            ts = tick_obj.time.replace(tzinfo=timezone.utc)
            if isinstance(tick_obj, TickByTickBidAsk):
                self.ticks_buffer.append({'ts_event': ts, 'pricebid': tick_obj.bidPrice, 'priceask': tick_obj.askPrice, 'sizebid': tick_obj.bidSize, 'sizeask': tick_obj.askSize, 'last_price': np.nan, 'last_size': np.nan})
            elif isinstance(tick_obj, (TickByTickAllLast, TickByTickLast)):
                self.ticks_buffer.append({'ts_event': ts, 'pricebid': np.nan, 'priceask': np.nan, 'sizebid': np.nan, 'sizeask': np.nan, 'last_price': tick_obj.price, 'last_size': tick_obj.size})
            if len(self.ticks_buffer) >= VOLUME_THRESHOLD:
                return self._create_bar()
        else:
            price = 0.0
            if hasattr(tick_obj, 'last') and tick_obj.last: price = tick_obj.last
            elif hasattr(tick_obj, 'close') and tick_obj.close: price = tick_obj.close
            elif hasattr(tick_obj, 'price') and tick_obj.price: price = tick_obj.price
            elif hasattr(tick_obj, 'bidPrice') and tick_obj.bidPrice: price = (tick_obj.bidPrice + tick_obj.askPrice) / 2
            if price > 0: self.correlator_snapshot[name] = price
        return None

    def _create_bar(self):
        df = pd.DataFrame(self.ticks_buffer)
        self.ticks_buffer = []
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
        for name, price in self.correlator_snapshot.items(): new_bar[name] = price
        bar_df = pd.DataFrame([new_bar])
        self.primary_bars = pd.concat([self.primary_bars, bar_df], ignore_index=True)
        if len(self.primary_bars) > WINDOW_SIZE: self.primary_bars = self.primary_bars.iloc[-WINDOW_SIZE:].reset_index(drop=True)
        return bar_df

    def compute_features_full(self):
        if len(self.primary_bars) < 200: return None
        engine = FeatureEngine()
        engine.bars = self.primary_bars.copy()
        windows = [25, 50, 100, 200, 400, 800]
        engine.add_features_to_bars(windows=windows)
        engine.add_physics_features()
        engine.add_advanced_physics_features(windows=windows)
        engine.add_microstructure_features()
        engine.add_delta_features(lookback=25)
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
        self.cooldowns = np.zeros(len(strategies), dtype=int)
        self.lot_size = cfg.STANDARD_LOT_SIZE
        
    async def check_intraday_exits(self, current_price):
        if self.position == 0 or self.active_strat_idx == -1: return
        strat = self.strategies[self.active_strat_idx]
        if not hasattr(self, 'last_atr'): return
        sl_dist, tp_dist = self.last_atr * strat.stop_loss_pct, self.last_atr * strat.take_profit_pct
        exit_signal, reason = False, ""
        if self.position > 0:
            if current_price <= (self.entry_price - sl_dist): exit_signal, reason = True, "SL"
            elif current_price >= (self.entry_price + tp_dist): exit_signal, reason = True, "TP"
        else:
            if current_price >= (self.entry_price + sl_dist): exit_signal, reason = True, "SL"
            elif current_price <= (self.entry_price - tp_dist): exit_signal, reason = True, "TP"
        if exit_signal:
            logger.info(f"📝 VIRTUAL EXIT: {reason} @ {current_price:.5f}")
            self.position, self.active_strat_idx, self.entry_price = 0, -1, 0.0
            if reason == "SL": self.cooldowns[self.active_strat_idx] = cfg.STOP_LOSS_COOLDOWN_BARS

    async def execute_signal(self, signal, strat_idx, price, atr):
        self.last_atr = atr
        if signal == self.position: return
        if self.position != 0:
            logger.info(f"📝 VIRTUAL EXIT: Signal Reversal @ {price:.5f}")
            self.position = 0
        if signal != 0:
            action = 'BUY' if signal > 0 else 'SELL'
            logger.info(f"📝 VIRTUAL ENTRY: {action} {abs(signal)} lots (Strat {strat_idx}) @ {price:.5f}")
            self.position, self.active_strat_idx, self.entry_price = signal, strat_idx, price

class PaperTradeApp:
    def __init__(self):
        self.ib = IB()
        self.data_manager = LiveDataManager()
        self.strategies = []
        self.executor = None
        self.stop_event = asyncio.Event()
        self.contract_map = {}
        
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
        if not self.load_strategies(): return
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
        try:
            while not self.stop_event.is_set(): await asyncio.sleep(1)
        finally: self.ib.disconnect()

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
        full_df = self.data_manager.compute_features_full()
        if full_df is None: return
        context = {c: full_df[c].values for c in full_df.columns}
        atr = full_df['atr'].iloc[-1] if 'atr' in full_df.columns else full_df['close'].iloc[-1]*0.001
        active_sig, active_idx = 0, -1
        for i, strat in enumerate(self.strategies):
            if self.executor.cooldowns[i] > 0: continue
            try:
                sig_vec = strat.generate_signal(context)
                if sig_vec[-1] != 0:
                    active_sig, active_idx = sig_vec[-1], i
                    break
            except: pass
        await self.executor.execute_signal(active_sig, active_idx, full_df['close'].iloc[-1], atr)

if __name__ == "__main__":
    app = PaperTradeApp()
    try: asyncio.run(app.run())
    except KeyboardInterrupt: logger.info("Exiting...")
