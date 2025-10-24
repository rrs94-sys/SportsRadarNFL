"""
CURRENT SEASON DATA LOADER (2025)
==================================
Loads 2025 season data using nfl_data_py weekly data
Separated from historical data loading for clean architecture
"""

import os
import pickle
import pandas as pd
import nfl_data_py as nfl
from datetime import datetime
from phase1_config import Config


class CurrentSeasonLoader:
    """Load current season (2025) NFL data"""

    def __init__(self, config=Config):
        self.config = config
        os.makedirs(config.CACHE_DIR, exist_ok=True)

    def load_2025_data(self, through_week=7, force_refresh=False):
        """
        Load 2025 season data

        Args:
            through_week: Last COMPLETED week (default 7 - update as weeks complete)
            force_refresh: Force re-download, ignore cache

        Returns:
            dict with schedules, team_schedules, player_weeks
        """
        print(f"\n{'='*70}")
        print(f"LOADING 2025 SEASON DATA (Weeks 1-{through_week})")
        print(f"{'='*70}")

        cache_file = f"{self.config.CACHE_DIR}/current_2025_w{through_week}.pkl"

        # Try cache first
        if not force_refresh and os.path.exists(cache_file):
            print("✅ Loading from cache...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)

            print(f"   Player-weeks: {len(data['player_weeks'])}")
            print(f"   Games: {len(data['schedules'])}")
            return data

        print("📥 Fetching from nfl_data_py...")

        try:
            # Load weekly player stats (already aggregated!)
            print("   [1/3] Weekly player data...")
            player_weeks = nfl.import_weekly_data([2025])
            player_weeks = player_weeks[player_weeks['week'] <= through_week].copy()
            print(f"         ✓ {len(player_weeks)} player-weeks")

            # Load schedules
            print("   [2/3] Schedules...")
            schedules = nfl.import_schedules([2025])
            schedules = schedules[schedules['week'] <= through_week].copy()

            # Map game_type to season_type for compatibility
            if 'game_type' in schedules.columns:
                schedules['season_type'] = schedules['game_type'].map({
                    'REG': 'REG',
                    'WC': 'POST',
                    'DIV': 'POST',
                    'CON': 'POST',
                    'SB': 'POST'
                }).fillna('REG')

            print(f"         ✓ {len(schedules)} games")

            # Create team-level schedules
            print("   [3/3] Team schedules...")
            team_schedules = self._create_team_schedules(schedules)
            print(f"         ✓ {len(team_schedules)} team-games")

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

            # Cache for next time
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print("\n✅ 2025 data cached for future use")

            return data

        except Exception as e:
            print(f"\n❌ Error loading 2025 data: {e}")
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


if __name__ == "__main__":
    # Test the loader
    loader = CurrentSeasonLoader()

    print("\n" + "="*60)
    print("TESTING 2025 DATA LOADER")
    print("="*60)

    # Load 2025 data through week 7
    data = loader.load_2025_data(through_week=7, force_refresh=True)

    print(f"\n✅ Successfully loaded 2025 data")
    print(f"   Player-weeks: {len(data['player_weeks'])}")
    print(f"   Unique players: {data['player_weeks']['player_id'].nunique()}")
    print(f"   Games: {len(data['schedules'])}")

    print(f"\n📊 Position breakdown:")
    print(data['player_weeks']['position'].value_counts())

    print(f"\n📊 Sample player-weeks:")
    sample_cols = ['player_name', 'position', 'team', 'week', 'passing_yards',
                   'rushing_yards', 'receiving_yards', 'opponent_team']
    available_cols = [c for c in sample_cols if c in data['player_weeks'].columns]
    print(data['player_weeks'][available_cols].head(10))
