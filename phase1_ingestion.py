"""
PHASE 1 DATA INGESTION - HISTORICAL DATA LOADER
===============================================
FIXED: Updated to use unified nfl_data_utils module
FIXED: Removed duplicate code and functions
FIXED: Implements dynamic season detection

Loads historical seasons using nfl_data_py weekly data
Works with current season loader for complete data pipeline
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

# FIXED: Import unified utilities instead of duplicating code
from nfl_data_utils import (
    get_current_season,
    get_historical_seasons,
    load_historical_data,
    validate_data_completeness
)


class HistoricalDataIngestion:
    """
    FIXED: Simplified class using unified nfl_data_utils module
    Load historical NFL data using nfl_data_py weekly stats
    """

    def __init__(self, config):
        self.config = config
        os.makedirs(config.CACHE_DIR, exist_ok=True)

    def load_historical_seasons(self, force_refresh: bool = False) -> Dict:
        """
        Load historical seasons using dynamic detection.

        FIXED: Now uses unified load_historical_data() function
        FIXED: Removed all duplicate code

        Args:
            force_refresh: Force re-download

        Returns:
            dict with schedules, team_schedules, player_weeks
        """
        print(f"\n{'='*70}")
        print(f"LOADING HISTORICAL DATA")
        print(f"{'='*70}")

        # FIXED: Use unified function with dynamic season detection
        try:
            # Load using unified utilities (automatically uses config.HISTORICAL_SEASONS)
            data = load_historical_data(
                lookback_years=3,
                cache_dir=self.config.CACHE_DIR,
                force_refresh=force_refresh
            )

            # FIXED: Validate data completeness
            validate_data_completeness(data)

            return data

        except Exception as e:
            print(f"\n   ❌ Error loading historical seasons: {e}")
            import traceback
            traceback.print_exc()
            raise

