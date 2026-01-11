"""
Paper Trade Module - Live Data Management
"""
import asyncio
import os
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from ib_insync import TickByTickBidAsk, TickByTickAllLast

import config as cfg
from feature_engine.core import FeatureEngine
from feature_engine import loader
from feature_engine.pipeline import run_pipeline

from .utils import logger, WINDOW_SIZE, WARMUP_DAYS, LIVE_BARS_FILE


class LiveDataManager:
    """Manages live tick data, bar creation, and feature computation."""

    def __init__(self):
        self.ticks_buffer = []
        self.current_vol = 0
        self.primary_bars = pd.DataFrame()
        self.external_cols = [t['name'] for t in cfg.TARGETS if t['name'] != cfg.PRIMARY_TICKER]
        self.correlator_snapshot = {name: np.nan for name in self.external_cols}
        self.gdelt_df = None

    def load_persistence(self):
        """Loads locally saved bars to restore state after restart."""
        if not os.path.exists(LIVE_BARS_FILE):
            return

        try:
            df = pd.read_parquet(LIVE_BARS_FILE)
            df = df.sort_values('time_start').drop_duplicates(subset=['time_start'])

            if 'time_start' in df.columns:
                df['time_start'] = pd.to_datetime(df['time_start'], utc=True)

            if len(df) > WINDOW_SIZE:
                df = df.iloc[-WINDOW_SIZE:]

            self.primary_bars = df.reset_index(drop=True)
            logger.info(f"Loaded {len(self.primary_bars)} bars from persistence.")

            # Update Snapshot from last bar
            if not self.primary_bars.empty:
                last_row = self.primary_bars.iloc[-1]
                for col in self.external_cols:
                    if col in last_row and not pd.isna(last_row[col]):
                        self.correlator_snapshot[col] = last_row[col]

        except Exception as e:
            logger.error(f"Failed to load persistence: {e}")

    def save_bar(self, bar_df=None):
        """Saves current bars to the persistent store."""
        try:
            os.makedirs(os.path.dirname(LIVE_BARS_FILE), exist_ok=True)
            self.primary_bars.to_parquet(LIVE_BARS_FILE, index=False)
        except Exception as e:
            logger.error(f"Failed to save persistence: {e}")

    async def fill_gaps(self, ib, contract):
        """Fetches missing ticks from IBKR to bridge the gap since last shutdown."""
        if self.primary_bars.empty:
            return

        last_dt = self.primary_bars['time_end'].max()
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)

        now_dt = datetime.now(timezone.utc)
        delta = now_dt - last_dt

        if delta.total_seconds() < 300:
            return  # No significant gap

        logger.info(f"Gap detected: {delta}. Fetching missing ticks from {last_dt}...")

        # Limit to last 2 hours to prevent massive delays
        start_time = last_dt
        if delta.total_seconds() > 7200:
            logger.warning("Gap > 2 hours. Only backfilling last 2 hours.")
            start_time = now_dt - timedelta(hours=2)

        fetched_count = 0
        logger.info(f"Starting gap fill loop from {start_time} to {now_dt}...")

        while start_time < (now_dt - timedelta(seconds=30)):
            formatted_time = start_time.strftime("%Y%m%d %H:%M:%S") + " UTC"
            logger.info(f"  -> Requesting 1000 ticks from {formatted_time}...")

            try:
                ticks = await ib.reqHistoricalTicksAsync(
                    contract, startDateTime=formatted_time,
                    endDateTime="", numberOfTicks=1000,
                    whatToShow='BID_ASK', useRth=False, ignoreSize=False
                )
            except asyncio.CancelledError:
                logger.warning("Tick fetch cancelled (Disconnect?). Stopping fill.")
                break
            except Exception as e:
                logger.error(f"Failed to fetch ticks batch: {e}")
                break

            if not ticks:
                logger.info("  -> No more ticks returned.")
                break

            # Wrapper to match TickByTickBidAsk interface
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
            last_tick_time = ticks[-1].time
            if last_tick_time.tzinfo is None:
                last_tick_time = last_tick_time.replace(tzinfo=timezone.utc)
            start_time = last_tick_time

            logger.info(f"  -> Fetched {len(ticks)} ticks. Next start: {start_time}")
            await asyncio.sleep(0.05)

        logger.info(f"Gap Filled. Fetched {fetched_count} ticks. New Head: {self.primary_bars['time_end'].max()}")

    def warmup(self, ib=None):
        """Loads history from disk."""
        logger.info("Warming up...")

        # 1. Load GDELT
        try:
            self.gdelt_df = loader.load_gdelt_v2_data(cfg.DIRS["DATA_GDELT"])
            if self.gdelt_df is None:
                self.gdelt_df = loader.load_gdelt_data(cfg.DIRS["DATA_GDELT"])
        except Exception as e:
            logger.warning(f"Failed to load GDELT data: {e}")

        # 2. Load Local Persistence
        self.load_persistence()

        # 3. If still empty (first run), try loading from Raw Data Lake
        if self.primary_bars.empty:
            data_dir = cfg.DIRS["DATA_RAW_TICKS"]
            if os.path.exists(data_dir):
                files = sorted([
                    f for f in os.listdir(data_dir)
                    if cfg.PRIMARY_TICKER in f and f.endswith(".parquet")
                ])
                if files:
                    recent = files[-WARMUP_DAYS:]
                    dfs = []
                    for f in recent:
                        try:
                            dfs.append(pd.read_parquet(os.path.join(data_dir, f)))
                        except Exception as e:
                            logger.warning(f"Failed to load {f}: {e}")
                    if dfs:
                        full = pd.concat(dfs, ignore_index=True).sort_values('ts_event')
                        from feature_engine.bars import create_volume_bars
                        bars = create_volume_bars(full, volume_threshold=cfg.VOLUME_THRESHOLD)
                        if bars is not None:
                            self.primary_bars = bars.tail(WINDOW_SIZE).reset_index(drop=True)
                            self.save_bar(None)

        # Ensure External Cols exist
        for col in self.external_cols:
            if col not in self.primary_bars.columns:
                self.primary_bars[col] = np.nan

        logger.info(f"Warmup complete. State: {len(self.primary_bars)} bars.")

    def add_tick(self, tick_obj, target_conf):
        """
        Process an incoming tick.
        Returns bar DataFrame if a new bar was created, None otherwise.
        """
        name = target_conf['name']

        if name == cfg.PRIMARY_TICKER:
            ts = tick_obj.time.replace(tzinfo=timezone.utc)

            tick_vol = 0
            if isinstance(tick_obj, TickByTickBidAsk):
                tick_vol = tick_obj.bidSize + tick_obj.askSize
                self.ticks_buffer.append({
                    'ts_event': ts,
                    'pricebid': tick_obj.bidPrice,
                    'priceask': tick_obj.askPrice,
                    'sizebid': tick_obj.bidSize,
                    'sizeask': tick_obj.askSize,
                    'last_price': np.nan,
                    'last_size': np.nan
                })
            elif isinstance(tick_obj, TickByTickAllLast):
                tick_vol = tick_obj.size
                self.ticks_buffer.append({
                    'ts_event': ts,
                    'pricebid': np.nan,
                    'priceask': np.nan,
                    'sizebid': np.nan,
                    'sizeask': np.nan,
                    'last_price': tick_obj.price,
                    'last_size': tick_obj.size
                })
            else:
                # TickWrapper from gap fill
                if hasattr(tick_obj, 'bidPrice'):
                    tick_vol = tick_obj.bidSize + tick_obj.askSize
                    self.ticks_buffer.append({
                        'ts_event': ts,
                        'pricebid': tick_obj.bidPrice,
                        'priceask': tick_obj.askPrice,
                        'sizebid': tick_obj.bidSize,
                        'sizeask': tick_obj.askSize,
                        'last_price': np.nan,
                        'last_size': np.nan
                    })

            self.current_vol += tick_vol

            if self.current_vol >= cfg.VOLUME_THRESHOLD:
                return self._create_bar()
        else:
            # Correlator tick - extract price
            price = 0.0
            if hasattr(tick_obj, 'last') and tick_obj.last and not np.isnan(tick_obj.last):
                price = tick_obj.last
            elif hasattr(tick_obj, 'close') and tick_obj.close and not np.isnan(tick_obj.close):
                price = tick_obj.close
            elif hasattr(tick_obj, 'price') and tick_obj.price and not np.isnan(tick_obj.price):
                price = tick_obj.price
            elif hasattr(tick_obj, 'bidPrice') and tick_obj.bidPrice and not np.isnan(tick_obj.bidPrice):
                price = (tick_obj.bidPrice + tick_obj.askPrice) / 2

            if price > 0:
                self.correlator_snapshot[name] = price

        return None

    def _create_bar(self):
        """Create a volume bar from accumulated ticks."""
        df = pd.DataFrame(self.ticks_buffer)
        self.ticks_buffer = []
        self.current_vol = 0

        if 'pricebid' in df.columns and 'priceask' in df.columns:
            df['mid_price'] = (df['pricebid'] + df['priceask']) / 2
            df['mid_price'] = df['mid_price'].fillna(df['last_price'])
        else:
            df['mid_price'] = df['last_price']

        vol = (df['sizebid'].fillna(0) + df['sizeask'].fillna(0)) if 'sizebid' in df.columns else df['last_size'].fillna(1)
        df['volume'] = np.where(vol == 0, 1, vol)
        df['net_aggressor_vol'] = np.sign(df['mid_price'].diff().fillna(0)) * df['volume']

        new_bar = {
            'time_start': df['ts_event'].iloc[0],
            'time_end': df['ts_event'].iloc[-1],
            'open': df['mid_price'].iloc[0],
            'high': df['mid_price'].max(),
            'low': df['mid_price'].min(),
            'close': df['mid_price'].iloc[-1],
            'volume': df['volume'].sum(),
            'net_aggressor_vol': df['net_aggressor_vol'].sum(),
            'tick_count': len(df)
        }

        # Inject latest snapshot prices
        for name, price in self.correlator_snapshot.items():
            new_bar[name] = price

        bar_df = pd.DataFrame([new_bar])
        self.primary_bars = pd.concat([self.primary_bars, bar_df], ignore_index=True)

        # Forward Fill missing external data
        self.primary_bars[self.external_cols] = self.primary_bars[self.external_cols].ffill()

        if len(self.primary_bars) > WINDOW_SIZE:
            self.primary_bars = self.primary_bars.iloc[-WINDOW_SIZE:].reset_index(drop=True)

        # Persist to disk
        self.save_bar(bar_df)

        return bar_df

    def compute_features_full(self):
        """Compute all features on current bars."""
        if len(self.primary_bars) < 200:
            return None

        engine = FeatureEngine(cfg.DIRS["DATA_DIR"])
        engine.bars = self.primary_bars.copy()

        # Build Data Cache dynamically from config.TARGETS
        def extract(name):
            if name in engine.bars.columns:
                return engine.bars[[name, 'time_start']].rename(
                    columns={name: 'close', 'time_start': 'ts_event'}
                )
            return None

        data_cache = {}
        for target in cfg.TARGETS:
            if target['name'] != cfg.PRIMARY_TICKER:
                # Map target name to cache key (lowercase)
                key = target['name'].lower()
                # Handle special cases
                if target['name'] == 'TICK_NYSE':
                    key = 'tick_nyse'
                elif target['name'] == 'TRIN_NYSE':
                    key = 'trin_nyse'
                elif target['name'] == '6E':
                    key = '6e'
                data_cache[key] = extract(target['name'])

        # Add GDELT
        data_cache['gdelt'] = self.gdelt_df

        # Run Shared Pipeline
        run_pipeline(engine, data_cache)

        if 'time_start' in engine.bars.columns:
            engine.bars['time_hour'] = engine.bars['time_start'].dt.hour
            engine.bars['time_weekday'] = engine.bars['time_start'].dt.dayofweek

        return engine.bars
