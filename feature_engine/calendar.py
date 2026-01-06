import pandas as pd
import numpy as np
import os
import pytz

class CalendarFeature:
    def __init__(self, calendar_path="resources/economic_calendar.csv"):
        self.calendar_path = calendar_path
        self.events = self.load_calendar()

    def load_calendar(self):
        if not os.path.exists(self.calendar_path):
            print(f"⚠️ Calendar file not found: {self.calendar_path}")
            return None
        
        try:
            df = pd.read_csv(self.calendar_path)
            df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
            df = df.sort_values('timestamp_utc')
            return df
        except Exception as e:
            print(f"❌ Error loading calendar: {e}")
            return None

    def add_features(self, bars_df):
        """
        Adds 'minutes_to_shock' and 'shock_decay' features.
        """
        if self.events is None or bars_df is None:
            return bars_df
        
        # Ensure bars are sorted and datetime
        if 'time_end' not in bars_df.columns:
            return bars_df # Can't align
            
        bars_df = bars_df.sort_values('time_end')
            
        # 1. Find NEXT event (Forward Search)
        # merge_asof direction='forward' finds the first event timestamp >= bar_time
        # Note: merge_asof requires both to be sorted.
        
        # Prepare events for merging
        events_df = self.events[['timestamp_utc', 'event']].copy()
        events_df = events_df.rename(columns={'timestamp_utc': 'next_event_time', 'event': 'next_event_type'})
        
        # We merge on 'time_end' of the bar.
        # direction='forward': match with the *next* available event.
        merged_fwd = pd.merge_asof(
            bars_df[['time_end']], 
            events_df, 
            left_on='time_end', 
            right_on='next_event_time', 
            direction='forward'
        )
        
        # 2. Find PREVIOUS event (Backward Search)
        events_prev = self.events[['timestamp_utc']].copy()
        events_prev = events_prev.rename(columns={'timestamp_utc': 'prev_event_time'})
        
        merged_bwd = pd.merge_asof(
            bars_df[['time_end']],
            events_prev,
            left_on='time_end',
            right_on='prev_event_time',
            direction='backward'
        )
        
        # 3. Calculate Deltas
        # Time to Next (Negative minutes)
        time_to_next = (merged_fwd['next_event_time'] - bars_df['time_end']).dt.total_seconds() / 60.0
        
        # Time Since Prev (Positive minutes)
        time_since_prev = (bars_df['time_end'] - merged_bwd['prev_event_time']).dt.total_seconds() / 60.0
        
        # 4. Construct 'minutes_to_shock'
        # Logic: If we are closer to the previous event (and it was recent), show positive time.
        # If we are approaching the next event (and it is soon), show negative time.
        # Threshold: 60 minutes.
        
        # Initialize with NaN or 999
        minutes_col = pd.Series(999.0, index=bars_df.index)
        
        # Pending Shock (< 120 mins)
        mask_approaching = (time_to_next <= 120) & (time_to_next >= 0)
        minutes_col[mask_approaching] = -time_to_next[mask_approaching]
        
        # Recent Shock (< 120 mins)
        mask_recent = (time_since_prev <= 120) & (time_since_prev >= 0)
        
        # Conflict resolution: If inside a sandwich (e.g. 1 hour after NFP, 1 hour before something else),
        # prioritize the one we are *closer* to.
        # Actually, let's just use the minimum absolute distance.
        
        dist_next = time_to_next.fillna(9999)
        dist_prev = time_since_prev.fillna(9999)
        
        # Find which is closer
        closer_to_prev = dist_prev < dist_next
        
        final_dist = pd.Series(999.0, index=bars_df.index)
        final_dist[closer_to_prev] = dist_prev[closer_to_prev] # Positive
        final_dist[~closer_to_prev] = -dist_next[~closer_to_prev] # Negative
        
        # Clamp to +/- 180 mins to avoid infinite values blowing up scales
        final_dist = final_dist.clip(-180, 180)
        
        bars_df['minutes_to_shock'] = final_dist
        
        # 5. Decay Feature (0.0 to 1.0)
        # 1.0 = At the event (time=0)
        # 0.5 = 15 mins away?
        # Function: 1 / (1 + abs(t)/10)
        # At t=0 -> 1.0
        # At t=10 -> 0.5
        # At t=60 -> 0.14
        bars_df['shock_decay'] = 1.0 / (1.0 + final_dist.abs() / 10.0)
        
        # Fill NaNs (no events found)
        bars_df['minutes_to_shock'] = bars_df['minutes_to_shock'].fillna(180)
        bars_df['shock_decay'] = bars_df['shock_decay'].fillna(0.0)
        
        return bars_df

def add_calendar_features(df):
    feature = CalendarFeature()
    return feature.add_features(df)
