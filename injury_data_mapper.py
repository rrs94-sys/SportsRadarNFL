"""
Injury Data Mapper - Unified Injury Data Integration
=====================================================
Integrates TANK01 injury data (real-time) with nflreadpy injury data (historical)
Provides field mapping and normalization for consistent injury data access

TANK01 Injury Format (from tank_injury_client.py):
    - player_name: str
    - status: str (out, inactive, doubtful, questionable, limited, probable, returned, cleared, active)
    - injury_impact: float (0.0 to 1.0, where 0.0=out, 1.0=active)
    - injury_description: str
    - source: 'news'
    - fetched_at: datetime

nflreadpy Injury Format (standard):
    - player_id: str
    - player_name: str (or player_display_name)
    - team: str
    - position: str
    - report_status: str (Out, Doubtful, Questionable, Probable, etc.)
    - report_primary_injury: str (body part)
    - practice_status: str
    - date_modified: datetime
    - week: int
    - season: int

This module provides:
    - Field mapping between formats
    - Impact score calculation
    - Unified injury data loading
    - Merging of real-time + historical data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime
from tank_injury_client import TankInjuryClient


class InjuryDataMapper:
    """
    Maps and normalizes injury data between TANK01 and nflreadpy formats
    """

    # Status mapping: TANK01 -> nflreadpy standard
    STATUS_MAP = {
        'out': 'Out',
        'inactive': 'Out',
        'ruled out': 'Out',
        'doubtful': 'Doubtful',
        'questionable': 'Questionable',
        'limited': 'Questionable',  # Limited practice = Questionable
        'probable': 'Probable',
        'returned': 'Probable',     # Returned to practice = Probable
        'cleared': 'Active',
        'active': 'Active'
    }

    # Impact scores for each status (aligned with tank_injury_client.py)
    IMPACT_SCORES = {
        'Out': 0.0,
        'Doubtful': 0.25,
        'Questionable': 0.75,
        'Probable': 0.95,
        'Active': 1.0
    }

    def __init__(self):
        """Initialize injury data mapper"""
        self.tank_client = TankInjuryClient()

    def get_unified_injuries(
        self,
        season: int = 2025,
        week: Optional[int] = None,
        use_tank01: bool = True,
        use_historical: bool = True
    ) -> pd.DataFrame:
        """
        Get unified injury data combining TANK01 (current) and nflreadpy (historical)

        Args:
            season: NFL season year
            week: Specific week (None = current/all)
            use_tank01: Include TANK01 real-time data (for 2025)
            use_historical: Include nflreadpy historical data (for 2020-2024)

        Returns:
            Unified injury DataFrame with standardized fields
        """
        injury_data = []

        # Load TANK01 current injuries (2025 only)
        if use_tank01 and season >= 2025:
            print(f"📊 Loading TANK01 injury data for {season}...")
            tank01_injuries = self._load_tank01_injuries(season, week)
            if not tank01_injuries.empty:
                injury_data.append(tank01_injuries)
                print(f"   ✓ TANK01: {len(tank01_injuries)} player injuries")

        # Load historical injuries from nflreadpy (2020-2024)
        if use_historical and season <= 2024:
            print(f"📊 Loading nflreadpy injury data for {season}...")
            historical_injuries = self._load_nflreadpy_injuries(season)
            if not historical_injuries.empty:
                injury_data.append(historical_injuries)
                print(f"   ✓ nflreadpy: {len(historical_injuries)} player injuries")

        # Merge all injury data
        if injury_data:
            unified = pd.concat(injury_data, ignore_index=True)
            unified = self._deduplicate_injuries(unified)
            print(f"✅ Total unified injuries: {len(unified)}")
            return unified
        else:
            print("⚠️  No injury data available")
            return self._get_empty_injury_dataframe()

    def _load_tank01_injuries(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        """
        Load and normalize TANK01 injury data

        Args:
            season: Season year
            week: Week number (optional)

        Returns:
            Normalized injury DataFrame
        """
        try:
            # Fetch from TANK01 client
            tank_injuries = self.tank_client.get_injury_report(week=week, force_refresh=False)

            if tank_injuries.empty:
                return self._get_empty_injury_dataframe()

            # Map to standard format
            normalized = pd.DataFrame()

            # Required fields
            normalized['player_name'] = tank_injuries['player_name']
            normalized['report_status'] = tank_injuries['status'].map(
                lambda x: self.STATUS_MAP.get(x.lower() if isinstance(x, str) else '', 'Active')
            )
            normalized['injury_impact'] = tank_injuries['injury_impact']

            # Optional fields
            normalized['injury_description'] = tank_injuries.get('injury_description', '')
            normalized['source'] = 'tank01'
            normalized['season'] = season
            normalized['week'] = week if week else 0
            normalized['fetched_at'] = tank_injuries.get('fetched_at', datetime.now().isoformat())

            # Extract body part from description if possible
            normalized['report_primary_injury'] = normalized['injury_description'].apply(
                self._extract_body_part
            )

            # Fill missing values
            normalized['player_id'] = ''
            normalized['team'] = ''
            normalized['position'] = ''

            return normalized

        except Exception as e:
            print(f"   ⚠️  Error loading TANK01 injuries: {e}")
            return self._get_empty_injury_dataframe()

    def _load_nflreadpy_injuries(self, season: int) -> pd.DataFrame:
        """
        Load and normalize nflreadpy injury data

        Args:
            season: Season year (2020-2024)

        Returns:
            Normalized injury DataFrame
        """
        try:
            from nfl_data_utils import import_injuries_safe

            # Load injuries for season
            injuries = import_injuries_safe([season])

            if injuries.empty:
                return self._get_empty_injury_dataframe()

            # Map to standard format
            normalized = pd.DataFrame()

            # Map column names (nflreadpy uses slightly different names)
            col_map = {
                'player_id': 'player_id',
                'player_name': 'player_name',
                'player_display_name': 'player_name',
                'team': 'team',
                'position': 'position',
                'report_status': 'report_status',
                'report_primary_injury': 'report_primary_injury',
                'week': 'week',
                'season': 'season'
            }

            # Copy and rename columns that exist
            for nfl_col, std_col in col_map.items():
                if nfl_col in injuries.columns:
                    normalized[std_col] = injuries[nfl_col]
                elif std_col == 'player_name' and 'player_display_name' in injuries.columns:
                    normalized[std_col] = injuries['player_display_name']

            # Calculate injury impact from status
            if 'report_status' in normalized.columns:
                normalized['injury_impact'] = normalized['report_status'].map(
                    lambda x: self.IMPACT_SCORES.get(x, 0.75)  # Default to Questionable
                )
            else:
                normalized['injury_impact'] = 0.75

            # Add source
            normalized['source'] = 'nflreadpy'

            # Ensure required columns exist
            for col in ['player_id', 'player_name', 'team', 'position', 'report_status',
                       'report_primary_injury', 'injury_impact', 'season', 'week', 'source']:
                if col not in normalized.columns:
                    if col in ['player_id', 'player_name', 'team', 'position',
                              'report_status', 'report_primary_injury', 'source']:
                        normalized[col] = ''
                    elif col == 'injury_impact':
                        normalized[col] = 0.75
                    else:
                        normalized[col] = 0

            return normalized

        except Exception as e:
            print(f"   ⚠️  Error loading nflreadpy injuries: {e}")
            return self._get_empty_injury_dataframe()

    def _extract_body_part(self, description: str) -> str:
        """
        Extract body part from injury description

        Args:
            description: Injury description text

        Returns:
            Body part name or empty string
        """
        if not isinstance(description, str):
            return ''

        description_lower = description.lower()

        body_parts = [
            'ankle', 'knee', 'shoulder', 'back', 'hamstring', 'concussion',
            'hip', 'groin', 'foot', 'hand', 'wrist', 'elbow', 'chest', 'rib',
            'quad', 'calf', 'thigh', 'neck', 'thumb', 'finger', 'toe',
            'achilles', 'bicep', 'tricep', 'forearm', 'shin', 'pectoral'
        ]

        for part in body_parts:
            if part in description_lower:
                return part.capitalize()

        return ''

    def _deduplicate_injuries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate injury entries, keeping most recent/reliable

        Args:
            df: Injury DataFrame

        Returns:
            Deduplicated DataFrame
        """
        if df.empty or 'player_name' not in df.columns:
            return df

        # Sort by source priority (tank01 > nflreadpy for current data)
        source_priority = {'tank01': 0, 'nflreadpy': 1, '': 2}
        df['_source_rank'] = df['source'].map(lambda x: source_priority.get(x, 2))

        # Sort by source rank, then by season/week descending
        df = df.sort_values(
            by=['_source_rank', 'season', 'week'],
            ascending=[True, False, False]
        )

        # Keep first occurrence of each player
        df = df.drop_duplicates(subset=['player_name'], keep='first')

        # Remove temporary ranking column
        df = df.drop(columns=['_source_rank'])

        return df.reset_index(drop=True)

    def _get_empty_injury_dataframe(self) -> pd.DataFrame:
        """
        Return empty injury DataFrame with standard columns

        Returns:
            Empty DataFrame with correct schema
        """
        return pd.DataFrame(columns=[
            'player_id', 'player_name', 'team', 'position',
            'report_status', 'report_primary_injury', 'injury_impact',
            'injury_description', 'season', 'week', 'source', 'fetched_at'
        ])

    def get_player_injury_impact(
        self,
        player_name: str,
        season: int = 2025,
        week: Optional[int] = None
    ) -> float:
        """
        Get injury impact score for a specific player

        Args:
            player_name: Player name to look up
            season: Season year
            week: Week number (optional)

        Returns:
            Impact score (0.0 = out, 1.0 = active)
        """
        injuries = self.get_unified_injuries(season=season, week=week)

        if injuries.empty:
            return 1.0  # Assume active if no injury data

        # Find player (case-insensitive partial match)
        player_injuries = injuries[
            injuries['player_name'].str.contains(player_name, case=False, na=False)
        ]

        if player_injuries.empty:
            return 1.0  # Not injured = active

        # Return injury impact from most recent entry
        return player_injuries.iloc[0]['injury_impact']

    def export_injury_snapshot(
        self,
        season: int = 2025,
        output_path: str = 'nfl_injuries_snapshot.csv'
    ) -> str:
        """
        Export current injury snapshot to CSV

        Args:
            season: Season year
            output_path: Output file path

        Returns:
            Path to exported file
        """
        injuries = self.get_unified_injuries(season=season)

        if injuries.empty:
            print("⚠️  No injury data to export")
            return None

        injuries.to_csv(output_path, index=False)
        print(f"✅ Injury snapshot exported: {output_path} ({len(injuries)} injuries)")

        return output_path


# Convenience functions for backward compatibility
def get_injuries_for_season(season: int, week: Optional[int] = None) -> pd.DataFrame:
    """
    Get unified injury data for a season

    Args:
        season: Season year
        week: Week number (optional)

    Returns:
        Unified injury DataFrame
    """
    mapper = InjuryDataMapper()
    return mapper.get_unified_injuries(season=season, week=week)


def get_player_injury_status(player_name: str, season: int = 2025) -> Dict:
    """
    Get injury status for a specific player

    Args:
        player_name: Player name
        season: Season year

    Returns:
        Dictionary with injury info
    """
    mapper = InjuryDataMapper()
    injuries = mapper.get_unified_injuries(season=season)

    if injuries.empty:
        return {
            'status': 'Active',
            'impact': 1.0,
            'injury': None,
            'source': None
        }

    player_injuries = injuries[
        injuries['player_name'].str.contains(player_name, case=False, na=False)
    ]

    if player_injuries.empty:
        return {
            'status': 'Active',
            'impact': 1.0,
            'injury': None,
            'source': None
        }

    injury = player_injuries.iloc[0]

    return {
        'status': injury.get('report_status', 'Active'),
        'impact': injury.get('injury_impact', 1.0),
        'injury': injury.get('report_primary_injury', None),
        'description': injury.get('injury_description', None),
        'source': injury.get('source', None)
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING INJURY DATA MAPPER")
    print("="*70)

    mapper = InjuryDataMapper()

    # Test 1: Load 2025 injuries (TANK01)
    print("\n[1/3] Testing 2025 injury data (TANK01)...")
    injuries_2025 = mapper.get_unified_injuries(season=2025, use_tank01=True, use_historical=False)
    if not injuries_2025.empty:
        print(f"\nSample 2025 injuries:")
        cols = ['player_name', 'report_status', 'injury_impact', 'report_primary_injury', 'source']
        available_cols = [c for c in cols if c in injuries_2025.columns]
        print(injuries_2025[available_cols].head(10))

    # Test 2: Load 2024 injuries (nflreadpy)
    print("\n[2/3] Testing 2024 injury data (nflreadpy)...")
    injuries_2024 = mapper.get_unified_injuries(season=2024, use_tank01=False, use_historical=True)
    if not injuries_2024.empty:
        print(f"\nSample 2024 injuries:")
        print(injuries_2024[available_cols].head(10))

    # Test 3: Export snapshot
    print("\n[3/3] Testing injury snapshot export...")
    snapshot_file = mapper.export_injury_snapshot(
        season=2025,
        output_path='nfl_injuries_20251026_snapshot.csv'
    )

    print("\n" + "="*70)
    print("INJURY DATA MAPPER TESTING COMPLETE")
    print("="*70)
