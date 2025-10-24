"""
PHASE 1 DATA INGESTION - HISTORICAL DATA ONLY
==============================================
Loads historical seasons (2022-2024) using nfl_data_py weekly data
Current season (2025) loaded separately by load_2025_data.py
"""

import os
import pickle
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from datetime import datetime


class HistoricalDataIngestion:
    """Load historical NFL data (2022-2024) using nfl_data_py weekly stats"""

    def __init__(self, config):
        self.config = config
        os.makedirs(config.CACHE_DIR, exist_ok=True)

    def load_historical_seasons(self, force_refresh=False):
        """
        Load historical seasons (2022, 2023, 2024) using weekly data

        Args:
            force_refresh: Force re-download

        Returns:
            dict with schedules, team_schedules, player_weeks
        """
        seasons = self.config.HISTORICAL_SEASONS
        cache_file = f"{self.config.CACHE_DIR}/historical_{'_'.join(map(str, seasons))}.pkl"

        # Try cache first
        if not force_refresh and os.path.exists(cache_file):
            print("   ✅ Loading from cache...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"      Player-weeks: {len(data['player_weeks'])}")
            print(f"      Games: {len(data['schedules'])}")
            return data

        print("   📥 Fetching from nfl_data_py...")

        try:
            # Load weekly player stats (already aggregated!)
            print(f"      [1/3] Weekly player data for {seasons}...")
            player_weeks = nfl.import_weekly_data(seasons)
            print(f"            ✓ {len(player_weeks)} player-weeks")

            # Load schedules
            print(f"      [2/3] Schedules...")
            schedules = nfl.import_schedules(seasons)

            # Map game_type to season_type for compatibility
            if 'game_type' in schedules.columns:
                schedules['season_type'] = schedules['game_type'].map({
                    'REG': 'REG',
                    'WC': 'POST',
                    'DIV': 'POST',
                    'CON': 'POST',
                    'SB': 'POST'
                }).fillna('REG')

            print(f"            ✓ {len(schedules)} games")

            # Create team-level schedules
            print(f"      [3/3] Team schedules...")
            team_schedules = self._create_team_schedules(schedules)
            print(f"            ✓ {len(team_schedules)} team-games")

            # Standardize column names
            player_weeks = self._standardize_columns(player_weeks)

            # Add game context (opponent, home/away, etc.)
            player_weeks = self._add_game_context(player_weeks, team_schedules)

            # Package data
            data = {
                'schedules': schedules,
                'team_schedules': team_schedules,
                'player_weeks': player_weeks
            }

            # Cache
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print("      ✅ Historical data cached")

            return data

        except Exception as e:
            print(f"\n   ❌ Error loading historical seasons: {e}")
            import traceback
            traceback.print_exc()
            raise


    def _create_team_schedules(self, schedules):
        """Explode schedules to team-level rows (home + away)"""
        # Home team rows
        home = schedules.copy()
        home['team'] = home['home_team']
        home['opponent'] = home['away_team']
        home['is_home'] = 1

        # Away team rows
        away = schedules.copy()
        away['team'] = away['away_team']
        away['opponent'] = away['home_team']
        away['is_home'] = 0

        # Combine
        return pd.concat([home, away], ignore_index=True)

    def _standardize_columns(self, player_weeks):
        """
        Standardize column names from nfl_data_py weekly data to our format

        Args:
            player_weeks: DataFrame from import_weekly_data()

        Returns:
            Standardized DataFrame
        """
        # Map nfl_data_py weekly columns to our expected format
        rename_map = {
            'player_id': 'player_id',
            'player_name': 'player_name',
            'player_display_name': 'player_name',
            'position': 'position',
            'position_group': 'position',
            'recent_team': 'team',
            'season': 'season',
            'week': 'week',

            # Passing
            'attempts': 'attempts',
            'completions': 'completions',
            'passing_yards': 'passing_yards',
            'passing_tds': 'passing_tds',
            'interceptions': 'interceptions',
            'sacks': 'sacks',

            # Receiving
            'targets': 'targets',
            'receptions': 'receptions',
            'receiving_yards': 'receiving_yards',
            'receiving_tds': 'receiving_tds',

            # Rushing
            'carries': 'carries',
            'rushing_yards': 'rushing_yards',
            'rushing_tds': 'rushing_tds'
        }

        # Only rename columns that exist
        existing_renames = {k: v for k, v in rename_map.items() if k in player_weeks.columns}
        df = player_weeks.rename(columns=existing_renames)

        # Ensure required columns exist
        required_cols = ['player_id', 'player_name', 'position', 'team', 'season', 'week',
                        'attempts', 'completions', 'passing_yards', 'passing_tds', 'interceptions', 'sacks',
                        'targets', 'receptions', 'receiving_yards', 'receiving_tds',
                        'carries', 'rushing_yards', 'rushing_tds']

        for col in required_cols:
            if col not in df.columns:
                df[col] = 0 if col not in ['player_id', 'player_name', 'position', 'team'] else ''

        # Fill NaN with 0 for numeric columns
        numeric_cols = ['attempts', 'completions', 'passing_yards', 'passing_tds', 'interceptions', 'sacks',
                       'targets', 'receptions', 'receiving_yards', 'receiving_tds',
                       'carries', 'rushing_yards', 'rushing_tds']

        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df

    def _add_game_context(self, player_weeks, team_schedules):
        """
        Add game context to player_weeks (opponent, home/away, game_id)

        Args:
            player_weeks: Player-week DataFrame
            team_schedules: Team schedules DataFrame

        Returns:
            player_weeks with added game context columns
        """
        # Merge with team schedules to get opponent and home/away info
        # Match on team, season, week
        merged = player_weeks.merge(
            team_schedules[['team', 'season', 'week', 'opponent', 'is_home', 'game_id']],
            on=['team', 'season', 'week'],
            how='left'
        )

        # Rename opponent to opponent_team for consistency
        merged = merged.rename(columns={'opponent': 'opponent_team'})

        # Fill missing values
        if 'opponent_team' not in merged.columns:
            merged['opponent_team'] = ''
        if 'is_home' not in merged.columns:
            merged['is_home'] = 0
        if 'game_id' not in merged.columns:
            merged['game_id'] = ''

        merged['opponent_team'] = merged['opponent_team'].fillna('')
        merged['is_home'] = merged['is_home'].fillna(0)
        merged['game_id'] = merged['game_id'].fillna('')

        return merged

