"""
Paper Trade Module - Live Data Management

Uses 1-minute bar subscriptions (consistent with backfill) to build volume bars.
"""
import asyncio
import os
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

import config as cfg
from feature_engine.core import FeatureEngine
from feature_engine import loader
from feature_engine.pipeline import run_pipeline

from .utils import logger, WINDOW_SIZE, WARMUP_DAYS, LIVE_BARS_FILE


class LiveDataManager:
    """Manages live bar data, volume bar creation, and feature computation."""

    def __init__(self):
        self.minute_bars_buffer = []  # Buffer of 1-minute bars for volume bar creation
        self.current_vol = 0
        self.primary_bars = pd.DataFrame()  # Volume bars
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
        """Fetches missing 1-minute bars from IBKR to bridge the gap since last shutdown."""
        if self.primary_bars.empty:
            return

        last_dt = self.primary_bars['time_end'].max()
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)

        now_dt = datetime.now(timezone.utc)
        delta = now_dt - last_dt

        if delta.total_seconds() < 300:
            return  # No significant gap

        logger.info(f"Gap detected: {delta}. Fetching missing bars from {last_dt}...")

        # Calculate duration string for IBKR (limit to 48 hours)
        # IBKR format: S=seconds, D=days, W=weeks, M=months, Y=years (NO hours/minutes!)
        gap_seconds = min(delta.total_seconds(), 48 * 3600)
        if gap_seconds <= 86400:  # <= 1 day
            duration_str = f"{int(gap_seconds) + 300} S"  # Add 5 min buffer
        else:
            duration_str = f"{int(gap_seconds / 86400) + 1} D"

        try:
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=duration_str,
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=2,  # UTC timezone
            )
        except Exception as e:
            logger.error(f"Failed to fetch historical bars for gap fill: {e}")
            return

        if not bars:
            logger.info("No bars returned for gap fill.")
            return

        # Filter bars newer than our last bar and process them
        fetched_count = 0
        for bar in bars:
            bar_time = bar.date
            if bar_time.tzinfo is None:
                bar_time = bar_time.replace(tzinfo=timezone.utc)

            if bar_time > last_dt:
                self.add_minute_bar(bar, {'name': cfg.PRIMARY_TICKER})
                fetched_count += 1

        logger.info(f"Gap Filled. Processed {fetched_count} bars. New Head: {self.primary_bars['time_end'].max() if not self.primary_bars.empty else 'N/A'}")

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

        # 3. If still empty (first run), try loading from Clean Data
        if self.primary_bars.empty:
            clean_dir = cfg.DIRS["DATA_CLEAN_TICKS"]
            clean_file = os.path.join(clean_dir, f"CLEAN_{cfg.PRIMARY_TICKER}.parquet")
            if os.path.exists(clean_file):
                try:
                    # Load clean 1-minute bar data
                    clean_df = pd.read_parquet(clean_file)
                    if 'ts_event' in clean_df.columns:
                        clean_df = clean_df.sort_values('ts_event')

                    # Create volume bars from 1-minute bars (consistent with backfill)
                    from feature_engine.bars import create_volume_bars_from_1min
                    bars = create_volume_bars_from_1min(clean_df, volume_threshold=cfg.VOLUME_THRESHOLD)
                    if bars is not None and len(bars) > 0:
                        self.primary_bars = bars.tail(WINDOW_SIZE).reset_index(drop=True)
                        self.save_bar(None)
                        logger.info(f"Loaded {len(self.primary_bars)} volume bars from clean data.")
                except Exception as e:
                    logger.warning(f"Failed to load clean data: {e}")

        # Ensure External Cols exist
        for col in self.external_cols:
            if col not in self.primary_bars.columns:
                self.primary_bars[col] = np.nan

        logger.info(f"Warmup complete. State: {len(self.primary_bars)} bars.")

    def add_minute_bar(self, bar_obj, target_conf):
        """
        Process an incoming 1-minute bar (from live subscription or gap fill).
        Returns volume bar DataFrame if a new volume bar was created, None otherwise.
        """
        name = target_conf['name']

        if name == cfg.PRIMARY_TICKER:
            # Extract bar time
            bar_time = bar_obj.date if hasattr(bar_obj, 'date') else bar_obj.time
            if bar_time.tzinfo is None:
                bar_time = bar_time.replace(tzinfo=timezone.utc)

            # Add 1-minute bar to buffer
            self.minute_bars_buffer.append({
                'ts_event': bar_time,
                'open': bar_obj.open,
                'high': bar_obj.high,
                'low': bar_obj.low,
                'close': bar_obj.close,
                'volume': bar_obj.volume,
            })

            self.current_vol += bar_obj.volume

            # Check if we've accumulated enough volume
            if self.current_vol >= cfg.VOLUME_THRESHOLD:
                return self._create_volume_bar()
        else:
            # Correlator bar - extract close price
            price = bar_obj.close if hasattr(bar_obj, 'close') else 0.0
            if price > 0:
                self.correlator_snapshot[name] = price

        return None

    def update_correlator(self, bar_obj, target_conf):
        """Update correlator snapshot from a bar update (no volume bar creation)."""
        name = target_conf['name']
        if name != cfg.PRIMARY_TICKER:
            price = bar_obj.close if hasattr(bar_obj, 'close') else 0.0
            if price > 0:
                self.correlator_snapshot[name] = price

    def _create_volume_bar(self):
        """Create a volume bar from accumulated 1-minute bars."""
        if not self.minute_bars_buffer:
            return None

        df = pd.DataFrame(self.minute_bars_buffer)
        self.minute_bars_buffer = []
        self.current_vol = 0

        # Aggregate 1-minute bars into volume bar (same logic as backfill)
        new_bar = {
            'time_start': df['ts_event'].iloc[0],
            'time_end': df['ts_event'].iloc[-1] + timedelta(minutes=1),  # End of last 1-min bar
            'open': df['open'].iloc[0],
            'high': df['high'].max(),
            'low': df['low'].min(),
            'close': df['close'].iloc[-1],
            'volume': df['volume'].sum(),
            'tick_count': len(df),  # Number of 1-min bars aggregated
        }

        # Calculate net aggressor volume from price direction
        df['price_change'] = df['close'].diff().fillna(0)
        df['net_aggressor'] = np.sign(df['price_change']) * df['volume']
        new_bar['net_aggressor_vol'] = df['net_aggressor'].sum()

        # Inject latest correlator snapshot prices
        for name, price in self.correlator_snapshot.items():
            new_bar[name] = price

        bar_df = pd.DataFrame([new_bar])
        self.primary_bars = pd.concat([self.primary_bars, bar_df], ignore_index=True)

        # Forward fill missing external data
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
                data_cache[key] = extract(target['name'])

        # Add GDELT
        data_cache['gdelt'] = self.gdelt_df

        # Run Shared Pipeline
        run_pipeline(engine, data_cache)

        if 'time_start' in engine.bars.columns:
            engine.bars['time_hour'] = engine.bars['time_start'].dt.hour
            engine.bars['time_weekday'] = engine.bars['time_start'].dt.dayofweek

        return engine.bars
