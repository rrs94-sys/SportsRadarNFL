"""
NFL Data Utilities - Unified Data Loading Module
=================================================
FIXED: Created unified module to eliminate code duplication and enable dynamic season/week detection
FIXED: Implements dynamic season and week detection per requirements
FIXED: Shared utility functions used across all data ingestion scripts

This module provides:
- Dynamic season detection (current year if month >= 9, else year - 1)
- Dynamic week detection (fetches completed weeks from schedule)
- Unified data loading functions
- Shared utilities for schedule and player data processing
"""

import os
import pickle
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from datetime import datetime
from typing import List, Dict, Tuple, Optional


# ============================================================================
# DYNAMIC SEASON AND WEEK DETECTION
# ============================================================================

def get_current_season() -> int:
    """
    Dynamically detect current NFL season.

    FIXED: Implements dynamic season detection per requirements:
    - If current month >= September (9), use current year
    - Otherwise, use previous year

    Returns:
        Current NFL season year
    """
    today = datetime.today()
    current_year = today.year
    current_month = today.month

    # NFL season starts in September
    season = current_year if current_month >= 9 else current_year - 1

    return season


def get_completed_weeks(season: int) -> List[int]:
    """
    Dynamically fetch completed weeks for a given season.

    FIXED: Implements dynamic week detection using game_finished flag from schedules

    Args:
        season: NFL season year

    Returns:
        List of completed week numbers
    """
    try:
        # Fetch current season schedule
        schedule = nfl.import_schedules([season])

        # Filter for completed regular season games
        completed_games = schedule[
            (schedule['game_finished'] == True) |
            (schedule['game_type'] == 'REG')
        ]

        # Get unique completed weeks
        if 'game_finished' in schedule.columns:
            completed_weeks = schedule.loc[schedule['game_finished'] == True, 'week'].unique()
        else:
            # Fallback: use current week minus 1
            print("  ⚠️  'game_finished' column not found, using date-based detection")
            today = datetime.today()
            # Estimate week based on date (rough approximation)
            season_start = datetime(season, 9, 1)
            weeks_elapsed = (today - season_start).days // 7
            completed_weeks = list(range(1, min(weeks_elapsed, 18)))

        completed_weeks = sorted([int(w) for w in completed_weeks if w > 0])

        return completed_weeks if completed_weeks else [1]

    except Exception as e:
        print(f"  ⚠️  Error detecting completed weeks: {e}")
        # Fallback to week 1
        return [1]


def get_latest_completed_week(season: int) -> int:
    """
    Get the latest completed week for a season.

    FIXED: Dynamic detection of latest completed week

    Args:
        season: NFL season year

    Returns:
        Latest completed week number
    """
    completed_weeks = get_completed_weeks(season)
    return max(completed_weeks) if completed_weeks else 1


def get_historical_seasons(current_season: int, lookback_years: int = 3) -> List[int]:
    """
    Get historical seasons for training (excludes current season).

    FIXED: Dynamic historical season generation

    Args:
        current_season: Current NFL season
        lookback_years: Number of years to look back (default 3)

    Returns:
        List of historical season years
    """
    return list(range(current_season - lookback_years, current_season))


# ============================================================================
# SHARED DATA LOADING UTILITIES
# ============================================================================

def create_team_schedules(schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Explode game schedules to team-level rows (one row per team per game).

    FIXED: Unified function used by all ingestion scripts

    Args:
        schedules: Game-level schedule DataFrame

    Returns:
        Team-level schedule DataFrame with home and away rows
    """
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


def standardize_player_columns(player_weeks: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names from nfl_data_py weekly data to our format.

    FIXED: Unified function eliminates duplicate code across scripts

    Args:
        player_weeks: DataFrame from import_weekly_data()

    Returns:
        Standardized DataFrame with consistent column names
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
    required_cols = [
        'player_id', 'player_name', 'position', 'team', 'season', 'week',
        'attempts', 'completions', 'passing_yards', 'passing_tds', 'interceptions', 'sacks',
        'targets', 'receptions', 'receiving_yards', 'receiving_tds',
        'carries', 'rushing_yards', 'rushing_tds'
    ]

    for col in required_cols:
        if col not in df.columns:
            # Add missing column with appropriate default
            if col in ['player_id', 'player_name', 'position', 'team']:
                df[col] = ''
            else:
                df[col] = 0

    # Fill NaN with 0 for numeric columns
    numeric_cols = [
        'attempts', 'completions', 'passing_yards', 'passing_tds', 'interceptions', 'sacks',
        'targets', 'receptions', 'receiving_yards', 'receiving_tds',
        'carries', 'rushing_yards', 'rushing_tds'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def add_game_context(
    player_weeks: pd.DataFrame,
    team_schedules: pd.DataFrame
) -> pd.DataFrame:
    """
    Add game context to player_weeks (opponent, home/away, game_id).

    FIXED: Unified function used by all ingestion scripts

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


def standardize_schedule_columns(schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize schedule column names for consistency.

    FIXED: Maps game_type to season_type for backward compatibility

    Args:
        schedules: Schedule DataFrame from nfl_data_py

    Returns:
        Standardized schedule DataFrame
    """
    # Map game_type to season_type for compatibility with older code
    if 'game_type' in schedules.columns:
        schedules['season_type'] = schedules['game_type'].map({
            'REG': 'REG',
            'WC': 'POST',
            'DIV': 'POST',
            'CON': 'POST',
            'SB': 'POST'
        }).fillna('REG')

    return schedules


# ============================================================================
# MAIN DATA LOADING FUNCTIONS
# ============================================================================

def load_nfl_data(
    seasons: List[int],
    weeks: Optional[List[int]] = None,
    cache_dir: str = "data_cache",
    force_refresh: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Load NFL data for specified seasons and weeks.

    FIXED: Unified data loading function with caching and error handling

    Args:
        seasons: List of NFL seasons to load
        weeks: Optional list of weeks to filter (None = all weeks)
        cache_dir: Directory for caching data
        force_refresh: Force re-download, ignore cache

    Returns:
        Dictionary with 'schedules', 'team_schedules', 'player_weeks'
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Create cache key
    seasons_str = '_'.join(map(str, sorted(seasons)))
    weeks_str = f"_w{'_'.join(map(str, weeks))}" if weeks else "_all"
    cache_file = os.path.join(cache_dir, f"nfl_data_{seasons_str}{weeks_str}.pkl")

    # Try cache first
    if not force_refresh and os.path.exists(cache_file):
        print(f"   ✅ Loading from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"      Player-weeks: {len(data['player_weeks'])}")
        print(f"      Games: {len(data['schedules'])}")
        return data

    print(f"   📥 Fetching from nfl_data_py...")

    try:
        # Load weekly player stats
        print(f"      [1/3] Loading player data for {seasons}...")
        if weeks:
            player_weeks = nfl.import_weekly_data(seasons, columns=None)
            player_weeks = player_weeks[player_weeks['week'].isin(weeks)].copy()
        else:
            player_weeks = nfl.import_weekly_data(seasons, columns=None)
        print(f"            ✓ {len(player_weeks)} player-weeks")

        # Load schedules
        print(f"      [2/3] Loading schedules...")
        schedules = nfl.import_schedules(seasons)
        if weeks:
            schedules = schedules[schedules['week'].isin(weeks)].copy()

        # Standardize schedule columns
        schedules = standardize_schedule_columns(schedules)
        print(f"            ✓ {len(schedules)} games")

        # Create team-level schedules
        print(f"      [3/3] Creating team schedules...")
        team_schedules = create_team_schedules(schedules)
        print(f"            ✓ {len(team_schedules)} team-games")

        # Standardize player columns
        player_weeks = standardize_player_columns(player_weeks)

        # Add game context
        player_weeks = add_game_context(player_weeks, team_schedules)

        # Package data
        data = {
            'schedules': schedules,
            'team_schedules': team_schedules,
            'player_weeks': player_weeks
        }

        # Cache for next time
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"      ✅ Data cached to: {cache_file}")

        return data

    except Exception as e:
        print(f"\n   ❌ Error loading NFL data: {e}")
        import traceback
        traceback.print_exc()
        raise


def load_current_season_data(
    through_week: Optional[int] = None,
    cache_dir: str = "data_cache",
    force_refresh: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Load current season data with dynamic season and week detection.

    FIXED: Implements dynamic detection per requirements

    Args:
        through_week: Last completed week to load (None = auto-detect)
        cache_dir: Directory for caching data
        force_refresh: Force re-download, ignore cache

    Returns:
        Dictionary with current season data
    """
    # FIXED: Dynamic season detection
    season = get_current_season()

    # FIXED: Dynamic week detection if not specified
    if through_week is None:
        through_week = get_latest_completed_week(season)
        print(f"   📊 Auto-detected latest completed week: {through_week}")

    print(f"\n{'='*70}")
    print(f"LOADING CURRENT SEASON DATA ({season}, Weeks 1-{through_week})")
    print(f"{'='*70}")

    # Load data for current season through specified week
    weeks = list(range(1, through_week + 1))
    data = load_nfl_data([season], weeks=weeks, cache_dir=cache_dir, force_refresh=force_refresh)

    print(f"\n✅ Current season data loaded: {len(data['player_weeks'])} player-weeks")

    return data


def load_historical_data(
    lookback_years: int = 3,
    cache_dir: str = "data_cache",
    force_refresh: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Load historical data with dynamic season detection.

    FIXED: Implements dynamic historical season detection

    Args:
        lookback_years: Number of years to look back (default 3)
        cache_dir: Directory for caching data
        force_refresh: Force re-download, ignore cache

    Returns:
        Dictionary with historical data
    """
    # FIXED: Dynamic historical season detection
    current_season = get_current_season()
    historical_seasons = get_historical_seasons(current_season, lookback_years)

    print(f"\n{'='*70}")
    print(f"LOADING HISTORICAL DATA ({historical_seasons})")
    print(f"{'='*70}")

    # Load all historical data
    data = load_nfl_data(historical_seasons, weeks=None, cache_dir=cache_dir, force_refresh=force_refresh)

    print(f"\n✅ Historical data loaded: {len(data['player_weeks'])} player-weeks")

    return data


def validate_data_completeness(data: Dict[str, pd.DataFrame]) -> bool:
    """
    Validate that loaded data is complete and non-empty.

    FIXED: Added data validation per requirements

    Args:
        data: Data dictionary to validate

    Returns:
        True if data is valid, raises AssertionError otherwise
    """
    # Check that all required keys exist
    required_keys = ['player_weeks', 'schedules', 'team_schedules']
    for key in required_keys:
        assert key in data, f"ERROR: Missing required key '{key}' in data"
        assert not data[key].empty, f"ERROR: '{key}' DataFrame is empty"

    # Check player_weeks has data
    player_weeks = data['player_weeks']
    assert len(player_weeks) > 0, "ERROR: No player-week data loaded"

    # Check required columns exist
    required_cols = ['player_id', 'team', 'season', 'week', 'position']
    for col in required_cols:
        assert col in player_weeks.columns, f"ERROR: Missing required column '{col}' in player_weeks"

    print(f"   ✅ Data validation passed:")
    print(f"      - {len(player_weeks)} player-weeks")
    print(f"      - {player_weeks['season'].nunique()} seasons")
    print(f"      - {player_weeks['player_id'].nunique()} unique players")
    print(f"      - Weeks: {sorted(player_weeks['week'].unique())}")

    return True


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING NFL DATA UTILITIES")
    print("="*70)

    # Test dynamic season detection
    print("\n1. Testing dynamic season detection...")
    current_season = get_current_season()
    print(f"   ✅ Current season: {current_season}")

    # Test dynamic week detection
    print("\n2. Testing dynamic week detection...")
    completed_weeks = get_completed_weeks(current_season)
    latest_week = get_latest_completed_week(current_season)
    print(f"   ✅ Completed weeks: {completed_weeks}")
    print(f"   ✅ Latest week: {latest_week}")

    # Test historical season generation
    print("\n3. Testing historical season generation...")
    historical_seasons = get_historical_seasons(current_season, lookback_years=3)
    print(f"   ✅ Historical seasons: {historical_seasons}")

    # Test loading current season data
    print("\n4. Testing current season data loading...")
    try:
        current_data = load_current_season_data(through_week=latest_week, force_refresh=False)
        validate_data_completeness(current_data)
        print(f"   ✅ Current season data loaded and validated")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")

    # Test loading historical data
    print("\n5. Testing historical data loading...")
    try:
        historical_data = load_historical_data(lookback_years=2, force_refresh=False)
        validate_data_completeness(historical_data)
        print(f"   ✅ Historical data loaded and validated")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")

    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
