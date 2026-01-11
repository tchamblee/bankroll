"""
Paper Trade Module - Main Application

Contains the PaperTradeApp class and entry point.
"""
import asyncio
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from ib_insync import IB, Forex
from ingest.ibkr_utils import build_contract_from_config, qualify_contract
import nest_asyncio

import config as cfg
from genome import Strategy
from backtest.strategy_loader import load_strategies as load_strategies_from_disk
from backtest.feature_computation import precompute_base_features, ensure_feature_context

from .utils import logger, configure_library_logging, CLIENT_ID
from .data_manager import LiveDataManager
from .execution import ExecutionEngine

nest_asyncio.apply()
configure_library_logging()


class PaperTradeApp:
    """
    Main paper trading application.

    Handles IBKR connection, data streaming, strategy execution,
    and virtual position management.
    """

    def __init__(self):
        self.ib = IB()
        self.data_manager = LiveDataManager()
        self.strategies = []
        self.executor = None
        self.stop_event = asyncio.Event()
        self.contract_map = {}
        self.primary_contract = None
        self.last_eur_tick = datetime.now()

        # Temp dir for JIT feature calculation
        self.temp_dir = tempfile.mkdtemp(prefix="paper_trade_jit_")
        self.existing_keys = set()
        logger.info(f"Created Temp Context: {self.temp_dir}")

    def __del__(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def load_strategies(self):
        """Load strategies from the mutex portfolio."""
        try:
            self.strategies, _ = load_strategies_from_disk(source_type='mutex', load_metrics=False)
            return bool(self.strategies)
        except Exception as e:
            logger.error(f"Failed to load strategies via shared loader: {e}")
            return False

    def _build_signal_context(self):
        """
        Compute features and build the execution context.
        Returns: (full_df, context, atr) or (None, None, None)
        """
        full_df = self.data_manager.compute_features_full()
        if full_df is None:
            return None, None, None

        # JIT Feature Computation
        self.existing_keys.clear()
        precompute_base_features(full_df, self.temp_dir, self.existing_keys)
        ensure_feature_context(self.strategies, self.temp_dir, self.existing_keys)

        # Load Context
        context = {}
        for key in self.existing_keys:
            try:
                arr = np.load(os.path.join(self.temp_dir, f"{key}.npy"))
                context[key] = arr
            except Exception:
                pass

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
        Smoke test: run a dry pass of all strategies against current features.
        Detects missing features before trading starts.
        """
        logger.info("Running Strategy Feature Validation (Smoke Test)...")

        if len(self.data_manager.primary_bars) < 200:
            logger.warning("Not enough data for validation (Need > 200 bars). Skipping.")
            return

        full_df, context, _ = self._build_signal_context()
        if context is None:
            logger.error("Validation Failed: Could not build feature context.")
            return

        failing_indices = []
        for i, strat in enumerate(self.strategies):
            try:
                strat.generate_signal(context)
            except KeyError as e:
                logger.critical(f"STRATEGY FAILURE: '{strat.name}' is missing feature {e}!")
                failing_indices.append(i)
            except Exception as e:
                logger.error(f"Strategy '{strat.name}' failed validation: {e}")
                failing_indices.append(i)

        if failing_indices:
            logger.critical(f"{len(failing_indices)} STRATEGIES FAILED VALIDATION. REMOVING THEM.")
            # Remove backwards to keep indices valid
            for i in sorted(failing_indices, reverse=True):
                removed = self.strategies.pop(i)
                logger.warning(f"Removed failing strategy: {removed.name}")

            # Re-initialize executor with updated strategy list
            if self.executor is not None:
                self.executor = ExecutionEngine(self.strategies)
        else:
            logger.info("All strategies passed feature validation.")

    async def run(self):
        """Main async run loop."""
        exit_code = 0

        try:
            if not self.load_strategies():
                logger.warning("No strategies found. Running in DATA ONLY mode (Accumulating Bars).")
            else:
                logger.info(f"Loaded {len(self.strategies)} strategies.")

            self.data_manager.warmup()

            # Initialize executor BEFORE validation
            self.executor = ExecutionEngine(self.strategies)

            # Validation (may modify strategy list and re-init executor)
            self.validate_strategies()

            await self.ib.connectAsync(cfg.IBKR_HOST, cfg.IBKR_PORT, clientId=CLIENT_ID)
            self.ib.reqMarketDataType(1)

            if not self.primary_contract:
                logger.info(f"Qualifying primary contract: {cfg.PRIMARY_TICKER}")
                eur = Forex(cfg.PRIMARY_TICKER)
                await self.ib.qualifyContractsAsync(eur)
                self.primary_contract = eur
                self.contract_map[eur.conId] = [t for t in cfg.TARGETS if t['name'] == cfg.PRIMARY_TICKER][0]

                # Subscribe primary ticker
                self.ib.reqMktData(eur, "", False, False)
                self.ib.reqTickByTickData(eur, "BidAsk", numberOfTicks=0, ignoreSize=False)
                logger.info(f"Subscribed Primary Ticker: {eur.localSymbol}")

                # Gap filling
                await self.data_manager.fill_gaps(self.ib, eur)

            # Subscribe correlator tickers
            for t in cfg.TARGETS:
                if t['name'] == cfg.PRIMARY_TICKER:
                    continue

                if t['mode'] == 'BARS_TRADES_1MIN':
                    try:
                        c = build_contract_from_config(t)
                        c = await qualify_contract(self.ib, c, logger)
                        if c:
                            self.ib.reqMktData(c, "", False, False)
                            self.contract_map[c.conId] = t
                    except Exception as e:
                        logger.warning(f"Failed to qualify/request {t['name']}: {e}")

            self.ib.pendingTickersEvent += self.on_pending_tickers
            logger.info("VIRTUAL PAPER TRADING ACTIVE (NO BROKER ORDERS)")

            disconnect_time = None
            last_state_update = datetime.now()
            self.last_eur_tick = datetime.now()

            while not self.stop_event.is_set():
                # Heartbeat
                Path("logs/paper_trade_heartbeat").touch()

                if self.executor:
                    self.executor.check_reset()

                    # Periodic state flush
                    if (datetime.now() - last_state_update).total_seconds() > 60:
                        self.executor.save_state()
                        last_state_update = datetime.now()

                # Check for data staleness
                stall_delta = (datetime.now() - self.last_eur_tick).total_seconds()
                if stall_delta > 900:  # 15 mins
                    logger.error(f"{cfg.PRIMARY_TICKER} Data Stalled (>{stall_delta:.0f}s). Exiting for restart.")
                    sys.exit(1)
                elif stall_delta > 60 and int(stall_delta) % 60 == 0:
                    logger.warning(f"{cfg.PRIMARY_TICKER} Idle for {stall_delta:.0f}s (Market Closed/Slow?)")

                # Check connection
                if not self.ib.isConnected():
                    if disconnect_time is None:
                        disconnect_time = datetime.now()
                        logger.warning("IBKR Disconnected!")
                    elif (datetime.now() - disconnect_time).total_seconds() > 60:
                        logger.error("Disconnected for > 60s. Exiting for restart.")
                        self.stop_event.set()
                        exit_code = 1
                        break
                else:
                    disconnect_time = None

                await asyncio.sleep(1)

        except BaseException as e:
            logger.error(f"CRITICAL FAILURE in paper_trade loop: {e}", exc_info=True)
            exit_code = 1

        finally:
            self.ib.disconnect()
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info("Cleaned up temp dir")

        return exit_code

    def on_pending_tickers(self, tickers):
        """Handle pending ticker updates from IBKR."""
        for t in tickers:
            conf = self.contract_map.get(t.contract.conId)
            if not conf:
                continue

            if conf['name'] == cfg.PRIMARY_TICKER:
                self.last_eur_tick = datetime.now()

                # Calculate mid price for exit checks
                price = 0
                if t.bidSize > 0 and t.askSize > 0:
                    price = (t.bid + t.ask) / 2
                elif t.last > 0:
                    price = t.last

                if price > 0 and self.executor:
                    asyncio.create_task(self.executor.check_intraday_exits(price))

                # Process tick-by-tick data for bar creation
                if t.tickByTicks:
                    for tick in t.tickByTicks:
                        bar = self.data_manager.add_tick(tick, conf)
                        if bar is not None:
                            asyncio.create_task(self.process_new_bar())
            else:
                self.data_manager.add_tick(t, conf)

    async def process_new_bar(self):
        """Process a newly completed bar."""
        if not self.executor:
            return

        # Decrement cooldowns at the start of each new bar
        self.executor.decrement_cooldowns()
        self.executor.save_state()

        full_df, context, atr = self._build_signal_context()
        if full_df is None or context is None:
            return

        self.executor.current_atr = atr

        # Enforce trading hours
        last_ts = full_df['time_start'].iloc[-1]
        if not hasattr(last_ts, 'hour'):
            last_ts = pd.to_datetime(last_ts)

        current_hour = last_ts.hour
        current_weekday = last_ts.dayofweek

        is_market_open = (
            current_hour >= cfg.TRADING_START_HOUR and
            current_hour < cfg.TRADING_END_HOUR and
            current_weekday < 5
        )

        if not is_market_open:
            if self.executor.position != 0:
                logger.info(f"Market Closed (Hour {current_hour}). Forcing Exit.")
                await self.executor.execute_signal(0, -1, full_df['close'].iloc[-1], atr)
            return

        # Generate signals
        active_sig, active_idx = 0, -1

        for i, strat in enumerate(self.strategies):
            if self.executor.cooldowns[i] > 0:
                continue
            try:
                sig_vec = strat.generate_signal(context)
                if sig_vec[-1] != 0:
                    active_sig, active_idx = sig_vec[-1], i
                    break
            except KeyError as e:
                logger.critical(f"RUNTIME MISSING FEATURE: {e} in Strategy {strat.name}")
            except Exception as e:
                logger.error(f"Strategy {i} Error: {e}")

        if active_sig != 0:
            await self.executor.execute_signal(active_sig, active_idx, full_df['close'].iloc[-1], atr)


def main():
    """Entry point for paper trading."""
    app = PaperTradeApp()
    exit_code = 0
    try:
        exit_code = asyncio.run(app.run())
    except KeyboardInterrupt:
        logger.info("Exiting...")
    sys.exit(exit_code)
