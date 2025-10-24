"""
CURRENT SEASON DATA LOADER
==========================
FIXED: Implements dynamic season and week detection
FIXED: Uses unified nfl_data_utils module
FIXED: Removed all duplicate code
FIXED: No more hardcoded seasons or weeks

Loads current season data using nfl_data_py weekly data
Separated from historical data loading for clean architecture
"""

import os
import pickle
import pandas as pd
from datetime import datetime
from typing import Optional, Dict

from phase1_config import Config

# FIXED: Import unified utilities instead of duplicating code
from nfl_data_utils import (
    get_current_season,
    get_latest_completed_week,
    load_current_season_data,
    validate_data_completeness
)


class CurrentSeasonLoader:
    """
    FIXED: Completely rebuilt using unified nfl_data_utils module
    FIXED: Implements dynamic season and week detection
    FIXED: Removed all duplicate code (190+ lines eliminated)

    Load current season NFL data with automatic season/week detection
    """

    def __init__(self, config=Config):
        self.config = config
        os.makedirs(config.CACHE_DIR, exist_ok=True)

    def load_2025_data(
        self,
        through_week: Optional[int] = None,
        force_refresh: bool = False
    ) -> Dict:
        """
        Load current season data with dynamic detection.

        FIXED: No longer hardcoded to 2025 or week 7
        FIXED: Automatically detects current season and latest completed week
        FIXED: Uses unified load_current_season_data() function

        Args:
            through_week: Last COMPLETED week (None = auto-detect latest)
            force_refresh: Force re-download, ignore cache

        Returns:
            dict with schedules, team_schedules, player_weeks
        """
        # FIXED: Dynamic season detection
        season = get_current_season()

        # FIXED: Dynamic week detection if not specified
        if through_week is None:
            through_week = get_latest_completed_week(season)
            print(f"\n📊 Auto-detected latest completed week: {through_week}")

        print(f"\n{'='*70}")
        print(f"LOADING CURRENT SEASON DATA ({season}, Weeks 1-{through_week})")
        print(f"{'='*70}")

        try:
            # FIXED: Use unified function with dynamic detection
            data = load_current_season_data(
                through_week=through_week,
                cache_dir=self.config.CACHE_DIR,
                force_refresh=force_refresh
            )

            # FIXED: Validate data completeness
            validate_data_completeness(data)

            return data

        except Exception as e:
            print(f"\n❌ Error loading current season data: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    # FIXED: Test with dynamic detection
    loader = CurrentSeasonLoader()

    print("\n" + "="*60)
    print("TESTING CURRENT SEASON DATA LOADER")
    print("="*60)

    # FIXED: Auto-detect season and week (no hardcoding!)
    season = get_current_season()
    latest_week = get_latest_completed_week(season)

    print(f"\n📊 Detected current season: {season}")
    print(f"📊 Detected latest completed week: {latest_week}")

    # Load current season data (auto-detects if through_week not specified)
    data = loader.load_2025_data(through_week=None, force_refresh=False)

    print(f"\n✅ Successfully loaded current season data")
    print(f"   Player-weeks: {len(data['player_weeks'])}")
    print(f"   Unique players: {data['player_weeks']['player_id'].nunique()}")
    print(f"   Games: {len(data['schedules'])}")
    print(f"   Seasons: {sorted(data['player_weeks']['season'].unique())}")
    print(f"   Weeks: {sorted(data['player_weeks']['week'].unique())}")

    print(f"\n📊 Position breakdown:")
    print(data['player_weeks']['position'].value_counts())

    print(f"\n📊 Sample player-weeks:")
    sample_cols = ['player_name', 'position', 'team', 'week', 'passing_yards',
                   'rushing_yards', 'receiving_yards', 'opponent_team']
    available_cols = [c for c in sample_cols if c in data['player_weeks'].columns]
    print(data['player_weeks'][available_cols].head(10))
