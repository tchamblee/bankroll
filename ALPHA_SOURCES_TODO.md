# Alpha Sources TODO - EUR/USD Enhancement

## Overview
Novel data sources and features to enhance EUR/USD prediction at 1.5-6 hour horizons.
These gaps were identified from comprehensive codebase analysis on 2026-01-11.

---

## 1. Economic Surprise Data
**Status:** Pending
**Priority:** High
**Complexity:** Medium

**Problem:** System tracks FRED macro but NOT consensus vs actual surprises. EUR/USD moves 20-50 pips on NFP/CPI surprises - not capturing the magnitude of the miss.

**Solution:**
- Scrape ForexFactory or Investing.com economic calendar
- Capture: event name, forecast, actual, previous, timestamp
- Calculate: `surprise_z = (actual - forecast) / historical_std`

**Data Source:** ForexFactory.com, Investing.com (Chrome scraping - blocks headless)

**Features to Generate:**
- `econ_surprise_usd` - Latest USD release surprise z-score
- `econ_surprise_eur` - Latest EUR release surprise z-score
- `econ_surprise_diff` - Relative surprise (EUR - USD)
- `surprise_decay_2h`, `surprise_decay_4h` - Decaying impact

---

## 2. Pre/Post Event Tension Windows
**Status:** COMPLETED (2026-01-11)
**Priority:** High
**Complexity:** Low

**Problem:** `calendar.py` flags event days but doesn't capture the 2-hour pre-announcement vol compression or post-release digestion pattern.

**Solution:**
- Ingest detailed event calendar with exact release times
- Calculate bars until next high-impact event
- Track bars since last high-impact event

**Features Generated:**
- `bars_to_event` - Countdown to next high-impact release (0-500 bars)
- `bars_since_event` - Bars elapsed since last release (0-500 bars)
- `pre_event_tension` - Ramps 0→1 in 2 hours before event
- `post_event_phase` - Binary flag for first 30 bars after release
- `pending_event_type` - Which event approaching (0=none, 1=NFP, 2=CPI)

---

## 3. Cross-Currency Basis
**Status:** Pending
**Priority:** Medium
**Complexity:** High

**Problem:** Have yield spreads but NOT EUR/USD basis - the synthetic dollar shortage indicator. When basis blows out, EUR rallies hard.

**Solution:**
- Track FX forward points (available from IBKR)
- Calculate implied rate differential
- Compare to actual rate differential (TNX - BUND proxy)
- Basis = Forward implied rate - Actual rate differential

**Data Source:** IBKR FX forwards, or Bloomberg terminal scrape

**Features to Generate:**
- `eur_usd_basis` - Current basis level
- `basis_z_60d` - Basis z-score vs 60-day history
- `basis_widening` - Rate of change (stress acceleration)

---

## 4. Yield Curve Acceleration
**Status:** COMPLETED (2026-01-11)
**Priority:** Medium
**Complexity:** Low

**Problem:** Have TNX-BUND spread but only as level. The velocity of flattening/steepening predicts reversals.

**Solution:**
- Add delta/momentum features to existing yield spread calculations
- Track acceleration (second derivative)

**Features to Generate:**
- `tnx_bund_delta_50` - 50-bar change in spread
- `tnx_bund_delta_100` - 100-bar change in spread
- `tnx_bund_accel` - Acceleration (delta of delta)
- `us_curve_delta_50` - TNX-US2Y momentum
- `eu_curve_delta_50` - BUND-SCHATZ momentum

**Implementation:** Modify `feature_engine/intermarket.py`

---

## 5. Option Gamma Clustering
**Status:** Pending
**Priority:** Medium
**Complexity:** High

**Problem:** Major EUR/USD strikes act as magnets. When spot approaches high-gamma strikes, dealers hedge and create mean reversion.

**Solution:**
- Ingest CME 6E options chain from IBKR
- Calculate gamma at each strike
- Identify strikes with highest open interest / gamma concentration
- Track distance to nearest high-gamma strike

**Data Source:** IBKR options chain for 6E (EUR futures)

**Features to Generate:**
- `nearest_gamma_strike` - Closest high-OI strike price
- `dist_to_gamma_strike` - Distance in ATR units
- `gamma_clustering_score` - Concentration measure
- `gamma_magnet_zone` - Binary flag when within 0.5 ATR of major strike

---

## 6. Treasury Auction Quality
**Status:** Pending
**Priority:** Low
**Complexity:** Medium

**Problem:** Weak US bond auctions (low bid-to-cover, high tail) precede EUR strength. Not tracking this.

**Solution:**
- Scrape TreasuryDirect.gov auction results
- Track 2Y, 5Y, 10Y, 30Y auctions
- Calculate quality metrics

**Data Source:** TreasuryDirect.gov, or financial news scrape

**Features to Generate:**
- `auction_btc_z` - Bid-to-cover ratio z-score vs history
- `auction_tail_z` - Tail (high yield - avg yield) z-score
- `auction_quality` - Composite score
- `auction_decay_24h` - Decaying impact after auction

**Schedule:**
- 2Y: Monthly
- 5Y: Monthly
- 10Y: Monthly (mid-month)
- 30Y: Monthly

---

## Implementation Order (Recommended)

1. **Yield Curve Acceleration** - Easiest, just add deltas to existing features
2. **Pre/Post Event Tension** - Medium, need calendar data with times
3. **Economic Surprise Data** - Chrome scraper needed, high impact
4. **Treasury Auction Quality** - Chrome scraper, lower frequency
5. **Cross-Currency Basis** - Need to verify IBKR data availability
6. **Option Gamma Clustering** - Most complex, need options data pipeline

---

## Notes

- Chrome DevTools MCP is set up and working for scraping
- All features should be added to `feature_engine/` modules
- New genes may be needed in `genome/genes/` for these features
- Backfill historical data where possible before live deployment
