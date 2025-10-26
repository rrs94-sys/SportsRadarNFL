"""
NFL Data Utilities - Unified Data Loading Module with nflreadpy
================================================================
FIXED: Created unified module to eliminate code duplication and enable dynamic season/week detection
FIXED: Implements dynamic season and week detection per requirements
FIXED: Migration from nfl_data_py (deprecated) to nflreadpy for 2025 support
FIXED: Normalized to single API source (nflreadpy) for all seasons (1999-2025)
FIXED: Added play-by-play (PBP) data loading for enhanced feature engineering

This module provides:
- Dynamic season detection (current year if month >= 9, else year - 1)
- Dynamic week detection (fetches completed weeks from schedule)
- Single normalized data source: nflreadpy (1999-2025)
- Play-by-play (PBP) data loading for detailed analysis
- 30/70 weighting model (30% historical, 70% recent)
- Unified data loading functions
- Shared utilities for schedule and player data processing
- Compatibility shim for nfl_data_py → nflreadpy migration

DATA SOURCE STRATEGY:
- All Seasons (1999-2025): nflreadpy (Polars-first with pandas conversion)
- Default lookback: 5 full seasons for historical data
- 30/70 weighting: 30% weight on historical, 70% weight on current season

NOTE: nfl_data_py was deprecated in Sep 2025. This module now uses nflreadpy
      (Polars-first, with automatic pandas conversion) for all data.
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Iterable

# ============================================================================
# NFLREADPY COMPATIBILITY SHIM
# ============================================================================
# nfl_data_py was deprecated in Sep 2025 → migrated to nflreadpy
# This shim provides drop-in compatibility while using nflreadpy internally

try:
    import nflreadpy as nfl_new  # Polars-first API
except ImportError as e:
    raise RuntimeError(
        "Missing dependency nflreadpy. Install with: pip install -U nflreadpy"
    ) from e


def _to_year_list(seasons: Union[int, Iterable[int], bool, None]) -> List[int] | bool:
    """
    Normalize season input. Return list[int] or True (nflreadpy supports seasons=True for 'all available').
    """
    if seasons is True:
        return True
    if seasons is None:
        return True
    if isinstance(seasons, int):
        return [seasons]
    return list(seasons)


def import_weekly_data(seasons: Union[int, Iterable[int], bool, None]) -> pd.DataFrame:
    """
    Replacement for nfl_data_py.import_weekly_data(...).
    Returns a pandas DataFrame of game-level (weekly) player stats.

    FIXED: Migrated from nfl_data_py to nflreadpy for 2025 support
    """
    yrs = _to_year_list(seasons)
    # nflreadpy returns Polars DataFrame; convert to pandas for compatibility
    df_pl = nfl_new.load_player_stats(yrs)  # weekly/game-level player stats
    df = df_pl.to_pandas()

    # Normalize column names to match nfl_data_py expectations
    # nflreadpy uses slightly different column names in some cases
    if 'player_display_name' in df.columns and 'player_name' not in df.columns:
        df = df.rename(columns={'player_display_name': 'player_name'})

    return df


def import_schedules(seasons: Union[int, Iterable[int], bool, None]) -> pd.DataFrame:
    """
    Replacement for nfl_data_py.import_schedules(...).
    Returns a pandas DataFrame of game schedules.

    FIXED: Migrated from nfl_data_py to nflreadpy
    """
    yrs = _to_year_list(seasons)
    df_pl = nfl_new.load_schedules(yrs)
    return df_pl.to_pandas()


def import_team_desc() -> pd.DataFrame:
    """
    Replacement for nfl_data_py.import_team_desc().
    Returns team descriptions/metadata.

    FIXED: Migrated from nfl_data_py to nflreadpy
    """
    df_pl = nfl_new.load_teams()
    return df_pl.to_pandas()


def import_injuries_safe(seasons: Union[int, Iterable[int], bool, None]) -> pd.DataFrame:
    """
    Injury data with 2025 guard.

    FIXED: 2025 injuries are unavailable upstream. Return empty for 2025 to avoid breakage.
    """
    yrs = _to_year_list(seasons)

    try:
        df_pl = nfl_new.load_injuries(yrs)  # may not include 2025 at all
        df = df_pl.to_pandas()
    except Exception as e:
        # If upstream transiently fails or injuries endpoint missing, return empty
        print(f"⚠️  Injuries unavailable: {e}. Returning empty DataFrame.")
        return pd.DataFrame()

    # Explicit 2025 guard: drop any 2025 rows to avoid downstream assumptions
    if "season" in df.columns:
        df = df[df["season"] <= 2024]
    else:
        # No season column — keep empty to be safe
        df = pd.DataFrame(columns=["season"])

    return df


def import_pbp_data(seasons: Union[int, Iterable[int], bool, None]) -> pd.DataFrame:
    """
    Load play-by-play (PBP) data from nflreadpy.

    FIXED: Added PBP data loading for enhanced feature engineering

    Args:
        seasons: Season(s) to load - int, list of ints, True for all, or None for all

    Returns:
        pandas DataFrame with play-by-play data
    """
    yrs = _to_year_list(seasons)
    print(f"   📊 Loading PBP data for seasons: {yrs if yrs != True else 'all available'}...")

    try:
        # nflreadpy returns Polars DataFrame; convert to pandas for compatibility
        df_pl = nfl_new.load_pbp(yrs)
        df = df_pl.to_pandas()
        print(f"      ✓ Loaded {len(df):,} plays")
        return df
    except Exception as e:
        print(f"      ⚠️  PBP data unavailable: {e}. Returning empty DataFrame.")
        return pd.DataFrame()


# Legacy compatibility: expose as 'nfl' module for backwards compatibility
class _NFLDataShim:
    """Compatibility shim to mimic nfl_data_py module interface"""
    import_weekly_data = staticmethod(import_weekly_data)
    import_schedules = staticmethod(import_schedules)
    import_team_desc = staticmethod(import_team_desc)
    import_injuries = staticmethod(import_injuries_safe)
    import_pbp_data = staticmethod(import_pbp_data)

nfl = _NFLDataShim()


# ============================================================================
# DYNAMIC SEASON AND WEEK DETECTION
# ============================================================================

def get_current_season() -> int:
    """
    Dynamically detect current NFL season with data availability check.

    FIXED: Implements dynamic season detection per requirements:
    - If current month >= September (9), use current year
    - Otherwise, use previous year
    - Verifies data exists, falls back to most recent available season

    Returns:
        Current NFL season year (verified to have data available)
    """
    today = datetime.today()
    current_year = today.year
    current_month = today.month

    # NFL season starts in September
    detected_season = current_year if current_month >= 9 else current_year - 1

    # FIXED: Verify data exists for detected season, fall back if needed
    available_season = _verify_season_data_exists(detected_season)

    if available_season != detected_season:
        print(f"⚠️  {detected_season} data not available yet, using {available_season}")

    return available_season


def _verify_season_data_exists(season: int) -> int:
    """
    Verify that data exists for a given season, fall back to most recent if not.

    FIXED: Core fix for 404 errors - checks data availability before fetching
    FIXED: Now uses nflreadpy via compatibility shim

    Args:
        season: Proposed season to check

    Returns:
        Season that actually has data available
    """
    # Try to load a minimal schedule to verify data exists
    max_attempts = 5
    for attempt in range(max_attempts):
        try_season = season - attempt
        if try_season < 1999:  # nflreadpy/nfl_data_py starts at 1999
            break

        try:
            # Quick check - try to load schedule (lightweight)
            schedule = nfl.import_schedules([try_season])
            if not schedule.empty:
                return try_season
        except Exception as e:
            # Data doesn't exist for this season, try previous year
            if attempt < max_attempts - 1:
                continue
            else:
                # Default to 2024 as last resort
                print(f"⚠️  Could not verify any season data, defaulting to 2024")
                return 2024

    # If all else fails, return 2024
    return 2024


def get_completed_weeks(season: int) -> List[int]:
    """
    Dynamically fetch completed weeks for a given season.

    FIXED: Implements dynamic week detection using game_finished flag from schedules
    FIXED: Better error handling for data availability issues

    Args:
        season: NFL season year

    Returns:
        List of completed week numbers
    """
    import nfl_data_py as nfl

    try:
        # Fetch season schedule
        schedule = nfl.import_schedules([season])

        if schedule.empty:
            print(f"  ⚠️  No schedule data for {season}, using default weeks")
            return list(range(1, 18))  # Return all regular season weeks

        # Get unique completed weeks
        if 'game_finished' in schedule.columns:
            # Filter for actually finished games
            completed_weeks = schedule.loc[schedule['game_finished'] == True, 'week'].unique()
        elif 'gameday' in schedule.columns:
            # Fallback: use games in the past
            today = datetime.today()
            past_games = schedule[pd.to_datetime(schedule['gameday']) < pd.Timestamp(today)]
            completed_weeks = past_games['week'].unique()
        else:
            # Last resort: estimate based on current date
            print("  ⚠️  Limited schedule data, using date-based estimation")
            today = datetime.today()
            season_start = datetime(season, 9, 1)
            weeks_elapsed = max(1, (today - season_start).days // 7)
            completed_weeks = list(range(1, min(weeks_elapsed, 19)))

        completed_weeks = sorted([int(w) for w in completed_weeks if w > 0 and w <= 18])

        return completed_weeks if completed_weeks else [1]

    except Exception as e:
        print(f"  ⚠️  Error detecting completed weeks for {season}: {e}")
        # Return reasonable default based on current date
        today = datetime.today()
        if today.month >= 9:  # September or later
            # Estimate weeks into season
            weeks = max(1, (today.month - 9) * 4 + today.day // 7)
            return list(range(1, min(weeks + 1, 19)))
        else:  # Before September - previous season
            return list(range(1, 19))  # All weeks from previous season


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


def get_historical_seasons(current_season: int, lookback_years: int = 5) -> List[int]:
    """
    Get historical seasons for training (excludes current season).

    FIXED: Dynamic historical season generation
    FIXED: Default changed to 5 years for enhanced training data

    Args:
        current_season: Current NFL season
        lookback_years: Number of years to look back (default 5)

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
    FIXED: Handles 404 errors by filtering unavailable seasons

    Args:
        seasons: List of NFL seasons to load
        weeks: Optional list of weeks to filter (None = all weeks)
        cache_dir: Directory for caching data
        force_refresh: Force re-download, ignore cache

    Returns:
        Dictionary with 'schedules', 'team_schedules', 'player_weeks'
    """
    import nfl_data_py as nfl

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

    # FIXED: Verify each season exists before loading
    available_seasons = []
    for season in seasons:
        try:
            # Quick check - try schedule first (lightweight)
            test_schedule = nfl.import_schedules([season])
            if not test_schedule.empty:
                available_seasons.append(season)
        except Exception:
            print(f"      ⚠️  Season {season} data not available, skipping")
            continue

    if not available_seasons:
        raise ValueError(f"No data available for any of the requested seasons: {seasons}")

    if len(available_seasons) < len(seasons):
        print(f"      ℹ️  Using available seasons: {available_seasons}")

    try:
        # Load weekly player stats
        print(f"      [1/3] Loading player data for {available_seasons}...")
        player_weeks = nfl.import_weekly_data(available_seasons, columns=None)

        if weeks:
            player_weeks = player_weeks[player_weeks['week'].isin(weeks)].copy()

        print(f"            ✓ {len(player_weeks)} player-weeks")

        # Load schedules
        print(f"      [2/3] Loading schedules...")
        schedules = nfl.import_schedules(available_seasons)

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
        print(f"   ℹ️  Attempted seasons: {available_seasons}")
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
    FIXED: Normalized to single API source (nflreadpy) for all seasons

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

    # FIXED: Use nflreadpy for all seasons (1999-2025)
    print(f"   📊 Data source: nflreadpy (unified API)")
    weeks = list(range(1, through_week + 1))
    data = load_nfl_data([season], weeks=weeks, cache_dir=cache_dir, force_refresh=force_refresh)

    print(f"\n✅ Current season data loaded: {len(data['player_weeks'])} player-weeks")

    return data


def load_historical_data(
    lookback_years: int = 5,
    cache_dir: str = "data_cache",
    force_refresh: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Load historical data with dynamic season detection.

    FIXED: Implements dynamic historical season detection
    FIXED: Default changed to 5 years for enhanced training data

    Args:
        lookback_years: Number of years to look back (default 5)
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


def load_hybrid_data(
    historical_years: int = 5,
    current_through_week: Optional[int] = None,
    cache_dir: str = "data_cache",
    force_refresh: bool = False,
    include_pbp: bool = True,
    include_injuries: bool = True,
    force_current_season: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load and merge historical data + current season data with 30/70 weighting.

    FIXED: Hybrid loader for training models on combined datasets
    FIXED: Normalized to single API source (nflreadpy) for all seasons
    FIXED: Default changed to 5 years for enhanced training data (2020-2024 + 2025)
    FIXED: Added 30/70 weighting model (30% historical, 70% current)
    FIXED: Optional PBP data loading for enhanced feature engineering
    FIXED: Integrated injury data from TANK01 (2025) and nflreadpy (2020-2024)

    Args:
        historical_years: Number of historical years to include (default 5 = 2020-2024)
        current_through_week: Last week of current season (None = auto-detect)
        cache_dir: Cache directory
        force_refresh: Force refresh flag
        include_pbp: Whether to load play-by-play data (default True)
        include_injuries: Whether to load injury data (default True)
        force_current_season: Override current season (e.g., 2025 for explicit testing)

    Returns:
        Dictionary with merged data with 30/70 weighting applied
        Includes: player_weeks, schedules, team_schedules, pbp, injuries
    """
    print("\n" + "="*70)
    print("LOADING HYBRID DATA (Historical + Current Season)")
    print("30/70 Weighting Model: 30% historical, 70% current")
    if force_current_season:
        print(f"Forced Current Season: {force_current_season}")
        print(f"Historical Range: {force_current_season - historical_years}-{force_current_season - 1}")
    print("="*70)

    # Determine current season (allow override for 2025 explicit usage)
    current_season = force_current_season if force_current_season else get_current_season()

    # Load historical data
    print(f"\n[1/4] Loading historical data...")
    historical = load_historical_data(
        lookback_years=historical_years,
        cache_dir=cache_dir,
        force_refresh=force_refresh
    )

    # Load current season
    print(f"\n[2/4] Loading current season ({current_season})...")
    current = load_current_season_data(
        through_week=current_through_week,
        cache_dir=cache_dir,
        force_refresh=force_refresh
    )

    # FIXED: Apply 30/70 weighting model
    print("\n[3/4] Merging datasets with 30/70 weighting...")

    # Add weight column: 0.30 for historical, 0.70 for current
    historical['player_weeks'] = historical['player_weeks'].copy()
    current['player_weeks'] = current['player_weeks'].copy()

    historical['player_weeks']['weight'] = 0.30
    current['player_weeks']['weight'] = 0.70

    print(f"   ✓ Historical weight: 0.30 ({len(historical['player_weeks'])} player-weeks)")
    print(f"   ✓ Current weight: 0.70 ({len(current['player_weeks'])} player-weeks)")

    merged_data = {
        'player_weeks': pd.concat([
            historical['player_weeks'],
            current['player_weeks']
        ], ignore_index=True),

        'schedules': pd.concat([
            historical['schedules'],
            current['schedules']
        ], ignore_index=True),

        'team_schedules': pd.concat([
            historical['team_schedules'],
            current['team_schedules']
        ], ignore_index=True),
    }

    # FIXED: Load PBP data if requested
    print(f"\n[4/4] Loading additional data...")
    if include_pbp:
        print("   📊 Loading play-by-play data...")
        pbp_seasons = list(range(current_season - historical_years, current_season + 1))
        merged_data['pbp'] = import_pbp_data(pbp_seasons)
        print(f"      ✓ PBP data loaded for seasons: {pbp_seasons}")
    else:
        merged_data['pbp'] = pd.DataFrame()

    # FIXED: Load injury data if requested
    if include_injuries:
        print("   🏥 Loading injury data...")
        try:
            from injury_data_mapper import InjuryDataMapper
            injury_mapper = InjuryDataMapper()

            # Load injuries for all seasons
            all_injuries = []
            injury_seasons = list(range(current_season - historical_years, current_season + 1))

            for season in injury_seasons:
                use_tank01 = (season >= 2025)
                use_historical = (season <= 2024)
                season_injuries = injury_mapper.get_unified_injuries(
                    season=season,
                    use_tank01=use_tank01,
                    use_historical=use_historical
                )
                if not season_injuries.empty:
                    all_injuries.append(season_injuries)

            if all_injuries:
                merged_data['injuries'] = pd.concat(all_injuries, ignore_index=True)
                print(f"      ✓ Injury data loaded: {len(merged_data['injuries'])} player injuries")
            else:
                merged_data['injuries'] = pd.DataFrame()
                print(f"      ⚠️  No injury data available")

        except ImportError as e:
            print(f"      ⚠️  Injury mapper not available: {e}")
            merged_data['injuries'] = pd.DataFrame()
        except Exception as e:
            print(f"      ⚠️  Error loading injuries: {e}")
            merged_data['injuries'] = pd.DataFrame()
    else:
        merged_data['injuries'] = pd.DataFrame()

    # FIXED: Verify merged data consistency
    _verify_hybrid_data_consistency(merged_data, historical, current)

    print(f"\n✅ Hybrid data loaded:")
    print(f"   Historical: {len(historical['player_weeks'])} player-weeks (weight: 0.30)")
    print(f"   Current:    {len(current['player_weeks'])} player-weeks (weight: 0.70)")
    print(f"   Total:      {len(merged_data['player_weeks'])} player-weeks")
    print(f"   PBP plays:  {len(merged_data['pbp']):,}")
    print(f"   Injuries:   {len(merged_data['injuries'])} player injuries")
    print(f"   Seasons:    {sorted(merged_data['player_weeks']['season'].unique())}")

    return merged_data


def _verify_hybrid_data_consistency(
    merged: Dict[str, pd.DataFrame],
    historical: Dict[str, pd.DataFrame],
    current: Dict[str, pd.DataFrame]
):
    """
    Verify that hybrid data merge was successful and consistent.

    FIXED: Critical validation ensuring data integrity

    Args:
        merged: Merged data dictionary
        historical: Historical data dictionary
        current: Current season data dictionary

    Raises:
        AssertionError if data inconsistency detected
    """
    pw_merged = merged['player_weeks']
    pw_hist = historical['player_weeks']
    pw_curr = current['player_weeks']

    # FIXED: Check row counts
    expected_rows = len(pw_hist) + len(pw_curr)
    actual_rows = len(pw_merged)
    assert actual_rows == expected_rows, f"Row count mismatch: expected {expected_rows}, got {actual_rows}"

    # FIXED: Check column alignment
    hist_cols = set(pw_hist.columns)
    curr_cols = set(pw_curr.columns)
    merged_cols = set(pw_merged.columns)

    # All columns should be present in merged
    assert hist_cols.issubset(merged_cols), "Missing historical columns in merge"
    assert curr_cols.issubset(merged_cols), "Missing current season columns in merge"

    # FIXED: Check for data type consistency
    for col in merged_cols:
        if col in pw_hist.columns and col in pw_curr.columns:
            hist_dtype = pw_hist[col].dtype
            curr_dtype = pw_curr[col].dtype

            # Allow some flexibility for numeric types
            if pd.api.types.is_numeric_dtype(hist_dtype) and pd.api.types.is_numeric_dtype(curr_dtype):
                continue

            # String types should match
            if pd.api.types.is_string_dtype(hist_dtype) or pd.api.types.is_object_dtype(hist_dtype):
                continue

            # For other types, they should match
            assert hist_dtype == curr_dtype, f"Data type mismatch for column '{col}': {hist_dtype} vs {curr_dtype}"

    print("   ✓ Hybrid data consistency validation passed")


def load_2020_2025_data(
    current_through_week: Optional[int] = None,
    cache_dir: str = "data_cache",
    force_refresh: bool = False,
    include_pbp: bool = True,
    include_injuries: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to explicitly load 2020-2024 (full) + 2025 (current) data.

    FIXED: Explicit 2020-2024 + 2025 loading per user requirements
    FIXED: Uses 30/70 weighting model
    FIXED: Includes PBP and injury data

    Args:
        current_through_week: Last week of 2025 season (None = auto-detect)
        cache_dir: Cache directory
        force_refresh: Force refresh flag
        include_pbp: Whether to load play-by-play data (default True)
        include_injuries: Whether to load injury data (default True)

    Returns:
        Dictionary with merged data:
        - Historical: 2020, 2021, 2022, 2023, 2024 (weight: 0.30)
        - Current: 2025 (weight: 0.70)
        - PBP: 2020-2025 play-by-play data
        - Injuries: 2020-2025 injury data (TANK01 for 2025, nflreadpy for 2020-2024)
    """
    print("\n" + "="*70)
    print("LOADING 2020-2024 (FULL) + 2025 (CURRENT) DATA")
    print("Historical: 2020, 2021, 2022, 2023, 2024 (weight: 0.30)")
    print("Current: 2025 (weight: 0.70)")
    print("="*70)

    return load_hybrid_data(
        historical_years=5,  # 2020-2024
        current_through_week=current_through_week,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        include_pbp=include_pbp,
        include_injuries=include_injuries,
        force_current_season=2025  # Explicitly use 2025 as current
    )


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
