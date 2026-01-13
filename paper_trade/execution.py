"""
Paper Trade Module - Execution Engine

Handles position management, barrier checking, and trade execution.
"""
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone

import numpy as np

import config as cfg
from utils.trading import check_barrier_long, check_barrier_short

from .utils import logger, play_sound, LIVE_STATE_FILE


class ExecutionEngine:
    """
    Manages virtual trading positions with proper barrier handling.

    Supports:
    - Directional barriers (sl_long, sl_short, tp_long, tp_short)
    - Volatility scaling (tightens SL when vol spikes)
    - Position sizing based on risk percentage
    """

    def __init__(self, strategies: list):
        self.strategies = strategies
        self.position = 0
        self.entry_price = 0.0
        self.active_strat_idx = -1
        self.current_sl = 0.0
        self.current_tp = 0.0
        self.current_atr = 0.0
        self.entry_atr = 0.0  # ATR at position entry (for vol scaling)
        self.last_atr = 0.0   # Most recent ATR value
        self.cooldowns = np.zeros(len(strategies), dtype=int)
        self.lot_size = cfg.STANDARD_LOT_SIZE

        self.balance = cfg.ACCOUNT_SIZE
        self.realized_pnl = 0.0

        self.load_state()

    def load_state(self):
        """Load persisted state from disk."""
        if not os.path.exists(LIVE_STATE_FILE):
            return

        try:
            with open(LIVE_STATE_FILE, 'r') as f:
                state = json.load(f)
                self.position = state.get('position', 0)
                self.entry_price = state.get('entry_price', 0.0)
                self.active_strat_idx = state.get('active_strat_idx', -1)
                self.current_sl = state.get('current_sl', 0.0)
                self.current_tp = state.get('current_tp', 0.0)
                self.current_atr = state.get('current_atr', 0.0)
                self.entry_atr = state.get('entry_atr', 0.0)
                self.last_atr = state.get('last_atr', 0.0)

                self.balance = state.get('balance', cfg.ACCOUNT_SIZE)
                self.realized_pnl = state.get('realized_pnl', 0.0)

                saved_cool = state.get('cooldowns', [])
                if len(saved_cool) == len(self.cooldowns):
                    self.cooldowns = np.array(saved_cool, dtype=int)

            # Validate active_strat_idx is within bounds
            if self.active_strat_idx >= len(self.strategies):
                logger.warning(
                    f"Restored active_strat_idx={self.active_strat_idx} is out of bounds "
                    f"(only {len(self.strategies)} strategies). Resetting position state."
                )
                self.position = 0
                self.active_strat_idx = -1
                self.entry_price = 0.0
                self.current_sl = 0.0
                self.current_tp = 0.0

            logger.info(
                f"Restored State: Pos={self.position} @ {self.entry_price:.5f} "
                f"(Strat {self.active_strat_idx}) | Bal=${self.balance:.0f}"
            )
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    def save_state(self):
        """Persist current state to disk atomically."""
        try:
            state = {
                'position': int(self.position) if isinstance(self.position, (int, np.integer)) else float(self.position),
                'entry_price': float(self.entry_price),
                'active_strat_idx': int(self.active_strat_idx),
                'current_sl': float(self.current_sl),
                'current_tp': float(self.current_tp),
                'current_atr': float(self.current_atr),
                'entry_atr': float(self.entry_atr),
                'last_atr': float(self.last_atr),
                'cooldowns': self.cooldowns.tolist(),
                'balance': float(self.balance),
                'realized_pnl': float(self.realized_pnl),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            # Atomic write
            os.makedirs(os.path.dirname(LIVE_STATE_FILE), exist_ok=True)
            with tempfile.NamedTemporaryFile('w', dir=os.path.dirname(LIVE_STATE_FILE), delete=False) as tf:
                json.dump(state, tf)
                temp_name = tf.name
            shutil.move(temp_name, LIVE_STATE_FILE)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def check_reset(self):
        """Check for external reset signal and wipe state if found."""
        trigger_path = os.path.join(cfg.DIRS['OUTPUT_DIR'], "RESET_TRIGGER")
        if not os.path.exists(trigger_path):
            return False

        logger.warning("RESET TRIGGER RECEIVED: Wiping Internal State...")

        self.position = 0
        self.entry_price = 0.0
        self.active_strat_idx = -1
        self.current_sl = 0.0
        self.current_tp = 0.0
        self.current_atr = 0.0
        self.entry_atr = 0.0
        self.last_atr = 0.0
        self.cooldowns = np.zeros(len(self.strategies), dtype=int)
        self.balance = cfg.ACCOUNT_SIZE
        self.realized_pnl = 0.0

        try:
            os.remove(trigger_path)
        except Exception:
            pass

        self.save_state()
        logger.warning("State Reset Complete.")
        return True

    def calculate_commission(self, price, lots):
        """Calculate commission for a trade."""
        value = price * abs(lots) * cfg.STANDARD_LOT_SIZE
        comm = value * (cfg.COST_BPS / 10000.0)
        return max(cfg.MIN_COMMISSION, comm)

    async def check_intraday_exits(self, current_price):
        """
        Check if current price has hit SL or TP barriers.

        Uses directional barriers and volatility scaling for consistency
        with the backtester and trade_simulator.
        """
        if self.position == 0 or self.active_strat_idx == -1:
            return

        # Bounds check for strategies list
        if not self.strategies or self.active_strat_idx >= len(self.strategies):
            return

        # Need ATR for barrier checking
        if self.entry_atr <= 0:
            # Fallback: use current_atr if entry_atr not set
            if self.current_atr > 0:
                self.entry_atr = self.current_atr
            else:
                return

        strat = self.strategies[self.active_strat_idx]

        # Get directional barriers
        direction = 'long' if self.position > 0 else 'short'
        sl_mult = strat.get_effective_sl(direction)
        tp_mult = strat.get_effective_tp(direction)

        # Use JIT barrier check with volatility scaling
        # For tick-level checking, use current_price as both low and high
        if self.position > 0:
            hit, exit_price, code = check_barrier_long(
                self.entry_price, self.entry_atr,
                current_price, current_price,
                sl_mult, tp_mult,
                self.current_atr,
                cfg.VOL_SCALE_THRESHOLD, cfg.VOL_SCALE_TIGHTEN
            )
        else:
            hit, exit_price, code = check_barrier_short(
                self.entry_price, self.entry_atr,
                current_price, current_price,
                sl_mult, tp_mult,
                self.current_atr,
                cfg.VOL_SCALE_THRESHOLD, cfg.VOL_SCALE_TIGHTEN
            )

        if not hit:
            return

        reason = "SL" if code == 1 else "TP"
        logger.info(f"VIRTUAL EXIT: {reason} @ {current_price:.5f} (barrier @ {exit_price:.5f})")
        play_sound("resources/exit.mp3")

        # PnL Calculation
        try:
            raw_pnl = (current_price - self.entry_price) * self.position * cfg.STANDARD_LOT_SIZE
            comm = (self.calculate_commission(self.entry_price, abs(self.position)) +
                    self.calculate_commission(current_price, abs(self.position)))
            net_pnl = raw_pnl - comm

            self.balance += net_pnl
            self.realized_pnl += net_pnl
            logger.info(f"Trade PnL: ${net_pnl:.2f} | Bal: ${self.balance:.2f}")
        except Exception as e:
            logger.error(f"PnL Calculation Error: {e}")

        # Apply cooldown if stopped out
        idx_to_cool = self.active_strat_idx

        self.position = 0
        self.active_strat_idx = -1
        self.entry_price = 0.0
        self.entry_atr = 0.0
        self.current_sl = 0.0
        self.current_tp = 0.0

        if reason == "SL":
            self.cooldowns[idx_to_cool] = cfg.STOP_LOSS_COOLDOWN_BARS
            logger.info(f"Strategy {idx_to_cool} cooling down for {cfg.STOP_LOSS_COOLDOWN_BARS} bars")

        self.save_state()

    def decrement_cooldowns(self):
        """Decrement all cooldown counters by 1."""
        self.cooldowns = np.maximum(0, self.cooldowns - 1)

    async def execute_signal(self, signal, strat_idx, price, atr):
        """
        Execute a trading signal.

        Handles:
        - Closing existing positions (reversal or flatten)
        - Opening new positions with directional barriers
        - Volatility-targeted position sizing
        """
        self.last_atr = atr
        self.current_atr = atr

        # No change - same direction (compare signs, not values)
        if np.sign(signal) == np.sign(self.position):
            # Hydrate SL/TP if needed (restored from state but barriers not computed)
            if self.position != 0 and self.active_strat_idx >= 0 and self.active_strat_idx < len(self.strategies) and (self.current_sl == 0.0 or self.current_tp == 0.0):
                try:
                    strat = self.strategies[self.active_strat_idx]
                    direction = 'long' if self.position > 0 else 'short'
                    sl_mult = strat.get_effective_sl(direction)
                    tp_mult = strat.get_effective_tp(direction)

                    # Use entry_atr if available, else current
                    barrier_atr = self.entry_atr if self.entry_atr > 0 else atr

                    if self.position > 0:
                        self.current_sl = self.entry_price - (barrier_atr * sl_mult)
                        self.current_tp = self.entry_price + (barrier_atr * tp_mult)
                    else:
                        self.current_sl = self.entry_price + (barrier_atr * sl_mult)
                        self.current_tp = self.entry_price - (barrier_atr * tp_mult)
                    self.save_state()
                except Exception as e:
                    logger.error(f"Failed to hydrate SL/TP on restore: {e}")
            return

        state_changed = False

        # Close existing position
        if self.position != 0:
            logger.info(f"VIRTUAL EXIT: Signal Reversal @ {price:.5f}")
            play_sound("resources/exit.mp3")

            try:
                raw_pnl = (price - self.entry_price) * self.position * cfg.STANDARD_LOT_SIZE
                comm = (self.calculate_commission(self.entry_price, abs(self.position)) +
                        self.calculate_commission(price, abs(self.position)))
                net_pnl = raw_pnl - comm

                self.balance += net_pnl
                self.realized_pnl += net_pnl
                logger.info(f"Trade PnL: ${net_pnl:.2f} | Bal: ${self.balance:.2f}")
            except Exception as e:
                logger.error(f"PnL Calculation Error: {e}")

            self.position = 0
            self.entry_atr = 0.0
            self.current_sl = 0.0
            self.current_tp = 0.0
            state_changed = True

        # Open new position
        if signal != 0:
            strat = self.strategies[strat_idx]

            # Get directional barriers
            direction = 'long' if signal > 0 else 'short'
            sl_mult = strat.get_effective_sl(direction)
            tp_mult = strat.get_effective_tp(direction)

            # Volatility targeting position sizing
            eff_sl = sl_mult if sl_mult > 0 else 1.0
            eff_atr = atr if atr > 1e-6 else 1e-6

            target_risk_dollars = self.balance * cfg.RISK_PER_TRADE_PERCENT
            calc_size = target_risk_dollars / (cfg.STANDARD_LOT_SIZE * eff_sl * eff_atr)
            final_size = min(max(calc_size, cfg.MIN_LOTS), cfg.MAX_LOTS)
            signed_size = np.sign(signal) * final_size

            action = 'BUY' if signal > 0 else 'SELL'
            logger.info(
                f"VIRTUAL ENTRY: {action} {final_size:.2f} lots (Risk ${target_risk_dollars:.0f}) "
                f"Strat {strat.name} @ {price:.5f} | SL:{sl_mult:.1f}x TP:{tp_mult:.1f}x ATR"
            )
            play_sound("resources/entry.mp3")

            self.position = signed_size
            self.active_strat_idx = strat_idx
            self.entry_price = price
            self.entry_atr = atr  # Store for vol scaling

            # Calculate actual SL/TP prices
            sl_dist = atr * sl_mult
            tp_dist = atr * tp_mult

            if signed_size > 0:  # Long
                self.current_sl = price - sl_dist
                self.current_tp = price + tp_dist
            else:  # Short
                self.current_sl = price + sl_dist
                self.current_tp = price - tp_dist

            state_changed = True

        if state_changed:
            self.save_state()
