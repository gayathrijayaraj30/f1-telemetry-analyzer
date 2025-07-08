import os
import logging
from collections import defaultdict
from typing import Tuple, List, Dict

import fastf1
import pandas as pd
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

# Enable FastF1 cache
CACHE_DIR = '/tmp/fastf1_cache'
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

def get_latest_available_year(start_year: int = 2018) -> int:
    current_year = datetime.now().year
    for year in reversed(range(start_year, current_year + 1)):
        try:
            calendar = fastf1.get_event_schedule(year)
            if not calendar.empty:
                return year
        except Exception as e:
            logger.warning(f"Failed to load calendar for {year}: {e}")
    raise ValueError("No available year found with race data.")

def load_telemetry_data(year: int, gp: str, session_type: str) -> pd.DataFrame:
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    laps = session.laps.reset_index(drop=True).copy()
    laps['Driver'] = laps['Driver'].astype(str)
    return laps

def get_available_race_data(start_year: int = 2018):
    current_year = datetime.now().year
    years = []
    gps_by_year = defaultdict(list)
    sessions_by_year_gp = defaultdict(list)
    common_sessions = ['FP1', 'FP2', 'FP3', 'Q', 'R', 'S']

    for year in range(start_year, current_year + 1):
        try:
            calendar = fastf1.get_event_schedule(year)
            if not calendar.empty:
                years.append(year)
                for event in calendar.itertuples():
                    gps_by_year[year].append(event.EventName)
                    sessions_by_year_gp[(year, event.EventName)] = common_sessions
        except Exception as e:
            logger.warning(f"Skipping {year} due to error: {e}")
            continue

    return years, gps_by_year, sessions_by_year_gp
