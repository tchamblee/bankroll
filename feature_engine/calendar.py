import pandas as pd
import numpy as np
import os
import pytz
import config

class CalendarFeature:
    # Event type encoding (US events only for ES trading)
    EVENT_TYPES = {'NFP': 1, 'CPI': 2, 'FOMC': 3}

    def __init__(self, calendar_path="resources/economic_calendar.csv"):
        self.calendar_path = calendar_path
        self.events = self.load_calendar()
        # Average bar duration in minutes (from config)
        self.avg_bar_mins = getattr(config, 'AVG_BAR_MINS', 1.5)

    def load_calendar(self):
        if not os.path.exists(self.calendar_path):
            print(f"⚠️ Calendar file not found: {self.calendar_path}")
            return None

        try:
            df = pd.read_csv(self.calendar_path)
            df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
            df = df.sort_values('timestamp_utc')
            # Add event type encoding
            df['event_code'] = df['event'].map(self.EVENT_TYPES).fillna(0).astype(int)
            return df
        except Exception as e:
            print(f"❌ Error loading calendar: {e}")
            return None

    def add_features(self, bars_df):
        """
        Adds calendar-based features:
        - minutes_to_shock: minutes to/from nearest event (legacy)
        - shock_decay: exponential decay from event (legacy)
        - bars_to_event: estimated bars until next event
        - bars_since_event: estimated bars since last event
        - pre_event_tension: tension score building before events (0-1)
        - post_event_phase: binary flag for first 30 bars after release
        - pending_event_type: which event is approaching (0=none, 1=NFP, 2=CPI, etc.)
        """
        if self.events is None or bars_df is None:
            return bars_df

        # Ensure bars are sorted and datetime
        if 'time_end' not in bars_df.columns:
            return bars_df  # Can't align

        bars_df = bars_df.sort_values('time_end')

        # 1. Find NEXT event (Forward Search)
        events_fwd = self.events[['timestamp_utc', 'event', 'event_code']].copy()
        events_fwd = events_fwd.rename(columns={
            'timestamp_utc': 'next_event_time',
            'event': 'next_event_type',
            'event_code': 'next_event_code'
        })

        merged_fwd = pd.merge_asof(
            bars_df[['time_end']],
            events_fwd,
            left_on='time_end',
            right_on='next_event_time',
            direction='forward'
        )

        # 2. Find PREVIOUS event (Backward Search)
        events_bwd = self.events[['timestamp_utc', 'event_code']].copy()
        events_bwd = events_bwd.rename(columns={
            'timestamp_utc': 'prev_event_time',
            'event_code': 'prev_event_code'
        })

        merged_bwd = pd.merge_asof(
            bars_df[['time_end']],
            events_bwd,
            left_on='time_end',
            right_on='prev_event_time',
            direction='backward'
        )

        # 3. Calculate Time Deltas (in minutes)
        time_to_next = (merged_fwd['next_event_time'] - bars_df['time_end']).dt.total_seconds() / 60.0
        time_since_prev = (bars_df['time_end'] - merged_bwd['prev_event_time']).dt.total_seconds() / 60.0

        # 4. Convert to Bars (using avg bar duration)
        bars_to_event = (time_to_next / self.avg_bar_mins).fillna(9999)
        bars_since_event = (time_since_prev / self.avg_bar_mins).fillna(9999)

        # Clamp to reasonable range
        bars_df['bars_to_event'] = bars_to_event.clip(0, 500).astype(int)
        bars_df['bars_since_event'] = bars_since_event.clip(0, 500).astype(int)

        # 5. Legacy: minutes_to_shock (signed, clamped)
        dist_next = time_to_next.fillna(9999)
        dist_prev = time_since_prev.fillna(9999)
        closer_to_prev = dist_prev < dist_next

        final_dist = pd.Series(999.0, index=bars_df.index)
        final_dist[closer_to_prev] = dist_prev[closer_to_prev]
        final_dist[~closer_to_prev] = -dist_next[~closer_to_prev]
        final_dist = final_dist.clip(-180, 180)

        bars_df['minutes_to_shock'] = final_dist
        bars_df['shock_decay'] = 1.0 / (1.0 + final_dist.abs() / 10.0)

        # 6. Pre-Event Tension (builds in 2 hours before event)
        # 0 when > 120 mins away, ramps to 1.0 at event time
        # tension = max(0, 1 - (mins_to_event / 120))
        pre_tension = (1.0 - time_to_next / 120.0).clip(0, 1).fillna(0)
        bars_df['pre_event_tension'] = pre_tension

        # 7. Post-Event Phase (first 30 bars after event = ~45 mins)
        # Binary flag for immediate post-release period
        post_event_bars_threshold = 30
        bars_df['post_event_phase'] = (bars_since_event <= post_event_bars_threshold).astype(int)

        # 8. Pending Event Type (what's coming next)
        # 0 = no event within 4 hours, 1 = NFP, 2 = CPI, etc.
        pending_threshold_mins = 240  # 4 hours
        pending_type = merged_fwd['next_event_code'].where(time_to_next <= pending_threshold_mins, 0)
        bars_df['pending_event_type'] = pending_type.fillna(0).astype(int)

        # Fill any remaining NaNs
        bars_df['minutes_to_shock'] = bars_df['minutes_to_shock'].fillna(180)
        bars_df['shock_decay'] = bars_df['shock_decay'].fillna(0.0)
        bars_df['pre_event_tension'] = bars_df['pre_event_tension'].fillna(0.0)

        return bars_df

def add_calendar_features(df):
    feature = CalendarFeature()
    return feature.add_features(df)
