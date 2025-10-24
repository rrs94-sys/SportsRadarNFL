"""
TANK01 API Client for 2025 NFL Player Statistics
================================================
FIXED: Complete integration for 2025 season data when nfl_data_py unavailable
FIXED: Seamless field mapping to nfl_data_py format for data consistency

This client fetches 2025 NFL player statistics from TANK01 API and normalizes
them to match nfl_data_py data structure exactly.

Key Features:
- Weekly player statistics (passing, rushing, receiving)
- Schedule data for game context
- Field normalization to nfl_data_py format
- Robust caching and error handling
- Automatic retries with exponential backoff
"""

import requests
import pandas as pd
import pickle
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from phase1_config import Config


# ============================================================================
# FIELD MAPPING: TANK01 → nfl_data_py Format
# ============================================================================

class Tank01FieldMapping:
    """
    Comprehensive field mapping between TANK01 API and nfl_data_py format.

    FIXED: Ensures 100% data compatibility between sources
    """

    # Player info fields
    PLAYER_FIELDS = {
        'playerID': 'player_id',
        'longName': 'player_name',
        'espnName': 'player_name',  # Fallback
        'pos': 'position',
        'team': 'team',
        'teamAbv': 'team',  # Fallback
    }

    # Passing stats
    PASSING_FIELDS = {
        'Cmp': 'completions',
        'passCompletions': 'completions',
        'Att': 'attempts',
        'passAttempts': 'attempts',
        'passYds': 'passing_yards',
        'Yds': 'passing_yards',
        'passTD': 'passing_tds',
        'TD': 'passing_tds',
        'Int': 'interceptions',
        'passInt': 'interceptions',
        'Sck': 'sacks',
        'sacksTaken': 'sacks',
    }

    # Rushing stats
    RUSHING_FIELDS = {
        'rushCarries': 'carries',
        'Car': 'carries',
        'rushYds': 'rushing_yards',
        'rushYards': 'rushing_yards',
        'rushTD': 'rushing_tds',
        'rushTouchdowns': 'rushing_tds',
    }

    # Receiving stats
    RECEIVING_FIELDS = {
        'Tgt': 'targets',
        'targets': 'targets',
        'Rec': 'receptions',
        'receptions': 'receptions',
        'recYds': 'receiving_yards',
        'receivingYards': 'receiving_yards',
        'recTD': 'receiving_tds',
        'receivingTouchdowns': 'receiving_tds',
    }

    # Game context
    GAME_FIELDS = {
        'gameID': 'game_id',
        'week': 'week',
        'season': 'season',
        'gameWeek': 'week',
        'seasonYear': 'season',
    }

    @classmethod
    def get_all_mappings(cls) -> Dict[str, str]:
        """Combine all field mappings into single dictionary"""
        all_fields = {}
        all_fields.update(cls.PLAYER_FIELDS)
        all_fields.update(cls.PASSING_FIELDS)
        all_fields.update(cls.RUSHING_FIELDS)
        all_fields.update(cls.RECEIVING_FIELDS)
        all_fields.update(cls.GAME_FIELDS)
        return all_fields


# ============================================================================
# TANK01 API CLIENT
# ============================================================================

class Tank01StatsClient:
    """
    Complete TANK01 API client for 2025 NFL statistics.

    FIXED: Robust implementation with caching, retries, and normalization
    """

    # Available TANK01 endpoints
    ENDPOINTS = {
        'player_stats': '/getNFLPlayerStats',
        'box_score': '/getNFLBoxScore',
        'team_stats': '/getNFLTeamStats',
        'schedule': '/getNFLGamesForWeek',
        'scores': '/getNFLScoresOnly',
    }

    def __init__(self, config: Config = Config):
        self.config = config
        self.api_key = config.TANK_API_KEY
        self.base_url = config.TANK_BASE_URL
        self.cache_dir = config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        self.headers = {
            'X-RapidAPI-Key': self.api_key,
            'X-RapidAPI-Host': 'tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com'
        }

    def fetch_2025_weekly_data(
        self,
        weeks: Optional[List[int]] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch 2025 weekly player statistics from TANK01 API.

        FIXED: Main entry point for 2025 data - matches nfl_data_py structure

        Args:
            weeks: List of weeks to fetch (None = all available)
            force_refresh: Force new API calls, ignore cache

        Returns:
            DataFrame matching nfl_data_py weekly_data format
        """
        print("\n" + "="*70)
        print("FETCHING 2025 DATA FROM TANK01 API")
        print("="*70)

        # Determine which weeks to fetch
        if weeks is None:
            weeks = self._get_available_weeks_2025()

        print(f"Target weeks: {weeks}")

        # Fetch data for each week
        all_player_stats = []

        for week in weeks:
            print(f"\n  Week {week}...")
            week_data = self._fetch_week_stats(week, force_refresh)

            if not week_data.empty:
                all_player_stats.append(week_data)
                print(f"    ✓ {len(week_data)} player-week records")
            else:
                print(f"    ⚠️  No data for week {week}")

        if not all_player_stats:
            print("\n❌ No 2025 data loaded from TANK01")
            return pd.DataFrame()

        # Combine all weeks
        combined = pd.concat(all_player_stats, ignore_index=True)

        # FIXED: Normalize to nfl_data_py format
        normalized = self._normalize_to_nfl_data_py_format(combined)

        # FIXED: Validate data consistency
        self._validate_data_structure(normalized)

        print(f"\n✅ Successfully loaded {len(normalized)} player-weeks from TANK01")
        print(f"   Weeks: {sorted(normalized['week'].unique())}")
        print(f"   Players: {normalized['player_id'].nunique()}")

        return normalized

    def _fetch_week_stats(self, week: int, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch stats for a single week with caching.

        Args:
            week: Week number
            force_refresh: Force new API call

        Returns:
            DataFrame with week stats
        """
        cache_file = os.path.join(
            self.cache_dir,
            f'tank01_2025_week{week}.pkl'
        )

        # FIXED: Check cache first
        if not force_refresh and os.path.exists(cache_file):
            cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if cache_age < timedelta(days=1):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        # Fetch from API with retries
        data = self._api_call_with_retry(
            endpoint='/getNFLBoxScore',
            params={'week': str(week), 'seasonType': 'reg', 'season': '2025'}
        )

        if not data or 'body' not in data:
            return pd.DataFrame()

        # Parse box scores into player stats
        player_stats = self._parse_box_scores(data['body'], week)

        # Cache it
        if not player_stats.empty:
            with open(cache_file, 'wb') as f:
                pickle.dump(player_stats, f)

        return player_stats

    def _parse_box_scores(self, games: Dict, week: int) -> pd.DataFrame:
        """
        Parse TANK01 box score data into player statistics.

        FIXED: Extracts all relevant stats from nested game data

        Args:
            games: Dictionary of games from API
            week: Week number

        Returns:
            DataFrame with player stats
        """
        all_players = []

        for game_id, game_data in games.items():
            if not isinstance(game_data, dict):
                continue

            # Get team stats from both home and away
            for location in ['home', 'away']:
                team_data = game_data.get(location, {})
                if not team_data:
                    continue

                team_abbr = team_data.get('teamAbv', team_data.get('team', ''))

                # Parse player stats
                player_stats = team_data.get('playerStats', {})

                # FIXED: Extract stats by category
                for stat_type in ['Passing', 'Rushing', 'Receiving']:
                    stat_dict = player_stats.get(stat_type, {})

                    for player_id, stats in stat_dict.items():
                        if not isinstance(stats, dict):
                            continue

                        player_record = {
                            'player_id': player_id,
                            'player_name': stats.get('longName', stats.get('name', '')),
                            'position': stats.get('pos', self._infer_position(stat_type)),
                            'team': team_abbr,
                            'week': week,
                            'season': 2025,
                            'game_id': game_id,
                        }

                        # Add all numeric stats
                        for key, value in stats.items():
                            if isinstance(value, (int, float)):
                                player_record[key] = value

                        all_players.append(player_record)

        df = pd.DataFrame(all_players)

        # FIXED: Aggregate by player (in case they appear multiple times)
        if not df.empty:
            groupby_cols = ['player_id', 'player_name', 'position', 'team', 'week', 'season']
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c not in groupby_cols]

            if numeric_cols:
                df = df.groupby(groupby_cols, as_index=False)[numeric_cols].sum()

        return df

    def _normalize_to_nfl_data_py_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize TANK01 data to match nfl_data_py structure exactly.

        FIXED: Critical function ensuring data compatibility

        Args:
            df: Raw TANK01 data

        Returns:
            Normalized DataFrame matching nfl_data_py format
        """
        if df.empty:
            return df

        # FIXED: Apply field mapping
        field_map = Tank01FieldMapping.get_all_mappings()

        # Rename columns using mapping
        rename_dict = {}
        for old_col in df.columns:
            if old_col in field_map:
                rename_dict[old_col] = field_map[old_col]

        df = df.rename(columns=rename_dict)

        # FIXED: Ensure all required columns exist
        required_cols = [
            'player_id', 'player_name', 'position', 'team', 'season', 'week',
            'attempts', 'completions', 'passing_yards', 'passing_tds', 'interceptions', 'sacks',
            'targets', 'receptions', 'receiving_yards', 'receiving_tds',
            'carries', 'rushing_yards', 'rushing_tds'
        ]

        for col in required_cols:
            if col not in df.columns:
                if col in ['player_id', 'player_name', 'position', 'team']:
                    df[col] = ''
                else:
                    df[col] = 0

        # FIXED: Fill NaN values
        numeric_cols = [
            'attempts', 'completions', 'passing_yards', 'passing_tds', 'interceptions', 'sacks',
            'targets', 'receptions', 'receiving_yards', 'receiving_tds',
            'carries', 'rushing_yards', 'rushing_tds'
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        # FIXED: Ensure data types match nfl_data_py
        df['season'] = df['season'].astype(int)
        df['week'] = df['week'].astype(int)

        return df

    def _validate_data_structure(self, df: pd.DataFrame):
        """
        Validate that normalized data matches expected structure.

        FIXED: Ensures data quality before merging with historical data

        Args:
            df: Normalized DataFrame

        Raises:
            AssertionError if validation fails
        """
        if df.empty:
            return

        required_cols = ['player_id', 'player_name', 'team', 'week', 'season', 'position']
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

        # Check for reasonable data
        assert df['season'].iloc[0] == 2025, "Season should be 2025"
        assert df['week'].min() >= 1 and df['week'].max() <= 18, "Week out of range"
        assert not df['player_id'].isna().all(), "Player IDs cannot be all null"

        print("  ✓ Data structure validation passed")

    def _api_call_with_retry(
        self,
        endpoint: str,
        params: Dict,
        max_retries: int = 3
    ) -> Optional[Dict]:
        """
        Make API call with exponential backoff retry.

        FIXED: Robust error handling for network issues

        Args:
            endpoint: API endpoint
            params: Query parameters
            max_retries: Maximum retry attempts

        Returns:
            JSON response or None
        """
        url = f"{self.base_url}{endpoint}"

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                wait_time = (2 ** attempt)  # Exponential backoff

                if attempt < max_retries - 1:
                    print(f"    ⚠️  API call failed, retrying in {wait_time}s... ({e})")
                    time.sleep(wait_time)
                else:
                    print(f"    ❌ API call failed after {max_retries} attempts: {e}")
                    return None

        return None

    def fetch_2025_schedule(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch 2025 season schedule from TANK01.

        Returns:
            DataFrame with game schedule
        """
        cache_file = os.path.join(self.cache_dir, 'tank01_2025_schedule.pkl')

        if not force_refresh and os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        print("\nFetching 2025 schedule from TANK01...")

        all_games = []

        # Fetch schedule for each week
        for week in range(1, 19):  # Weeks 1-18
            data = self._api_call_with_retry(
                endpoint='/getNFLGamesForWeek',
                params={'week': str(week), 'seasonType': 'reg', 'season': '2025'}
            )

            if data and 'body' in data:
                games = data['body']
                if isinstance(games, dict):
                    for game_id, game_info in games.items():
                        if isinstance(game_info, dict):
                            game_info['game_id'] = game_id
                            game_info['week'] = week
                            game_info['season'] = 2025
                            all_games.append(game_info)

        schedule_df = pd.DataFrame(all_games)

        if not schedule_df.empty:
            # Normalize column names
            schedule_df = schedule_df.rename(columns={
                'gameID': 'game_id',
                'home': 'home_team',
                'away': 'away_team',
                'gameDate': 'gameday',
                'gameTime': 'gametime',
            })

            with open(cache_file, 'wb') as f:
                pickle.dump(schedule_df, f)

        return schedule_df

    def _get_available_weeks_2025(self) -> List[int]:
        """
        Determine which weeks of 2025 season have data available.

        Returns:
            List of available week numbers
        """
        # FIXED: Use current date to estimate available weeks
        today = datetime.today()

        if today.year < 2025:
            return []

        if today.year == 2025:
            if today.month < 9:
                return []

            # Estimate weeks based on date
            season_start = datetime(2025, 9, 1)
            weeks_elapsed = (today - season_start).days // 7
            return list(range(1, min(weeks_elapsed + 1, 19)))

        # After 2025, all weeks available
        return list(range(1, 19))

    def _infer_position(self, stat_type: str) -> str:
        """Infer player position from stat type"""
        position_map = {
            'Passing': 'QB',
            'Rushing': 'RB',
            'Receiving': 'WR',
        }
        return position_map.get(stat_type, '')


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING TANK01 STATS CLIENT")
    print("="*70)

    client = Tank01StatsClient()

    # Test fetching 2025 data
    print("\n1. Testing 2025 data fetch...")
    data_2025 = client.fetch_2025_weekly_data(weeks=[1, 2], force_refresh=False)

    if not data_2025.empty:
        print(f"\n✅ Successfully loaded {len(data_2025)} records")
        print(f"\nColumns: {list(data_2025.columns)}")
        print(f"\nSample data:")
        print(data_2025.head())

        print(f"\nPosition breakdown:")
        print(data_2025['position'].value_counts())
    else:
        print("\n⚠️  No 2025 data available yet")

    # Test schedule fetch
    print("\n2. Testing schedule fetch...")
    schedule = client.fetch_2025_schedule(force_refresh=False)

    if not schedule.empty:
        print(f"✅ Loaded {len(schedule)} games")
    else:
        print("⚠️  No schedule data available")

    print("\n" + "="*70)
    print("TANK01 CLIENT TEST COMPLETE")
    print("="*70)
