"""
SportsGameOdds API Client with Historical Line Tracking
Optimized for limited API calls (500/month limit)
"""

import requests
import pandas as pd
import pickle
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from phase1_config import Config


class SportsGameOddsClient:
    """
    Client for SportsGameOdds API with intelligent caching and call optimization.
    Tracks historical lines for edge calculation and market movement analysis.
    """

    def __init__(self, config: Config = Config):
        self.config = config
        self.api_key = config.SPORTSGAMEODDS_API_KEY
        self.base_url = config.SPORTSGAMEODDS_BASE_URL
        self.lines_cache_dir = config.LINES_CACHE_DIR

    def get_player_props(
        self,
        week: int,
        season: int = None,
        force_refresh: bool = False,
        save_historical: bool = True
    ) -> pd.DataFrame:
        """
        Fetch player prop lines for a specific week.
        Uses cache to minimize API calls.

        Args:
            week: NFL week number
            season: Season year (default: current season from config)
            force_refresh: Force new API call even if cached
            save_historical: Save lines to historical cache for future tracking

        Returns:
            DataFrame with columns: player_name, team, market, line, over_odds,
                                   under_odds, sportsbook, timestamp
        """
        if season is None:
            season = self.config.CURRENT_SEASON

        cache_file = os.path.join(
            self.lines_cache_dir,
            f'props_s{season}_w{week}_{datetime.now().strftime("%Y%m%d")}.pkl'
        )

        # Check cache first (reuse today's call if exists)
        if not force_refresh and os.path.exists(cache_file):
            print(f"📦 Using cached props for {season} Week {week}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        # Check if we can make API call
        if not self.config.can_make_sgo_call():
            print("❌ SGO API call limit reached! Using fallback...")
            return self._get_fallback_lines(week, season)

        print(f"🌐 Fetching props from SGO API (Week {week}, {season})...")

        try:
            url = f"{self.base_url}/props/player"
            params = {
                'sport': 'NFL',
                'season': season,
                'week': week
            }
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }

            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()

            # Track the call
            remaining = self.config.track_sgo_call()
            print(f"✅ SGO API call successful ({remaining} calls remaining)")

            # Parse response
            data = response.json()
            props = data.get('props', [])

            if not props:
                print("⚠️ No props returned from API")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(props)
            df['timestamp'] = datetime.now().isoformat()
            df['week'] = week
            df['season'] = season

            # Standardize market names
            df['market'] = df['market'].str.lower().str.replace(' ', '_')

            # Filter to relevant markets
            relevant_markets = [
                'receiving_yards', 'receptions', 'receiving_tds',
                'rushing_yards', 'rushing_tds',
                'passing_yards', 'completions', 'passing_tds',
                'pass_attempts', 'interceptions'
            ]
            df = df[df['market'].isin(relevant_markets)]

            # Cache it
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)

            # Save to historical lines if requested
            if save_historical:
                self._save_to_historical_lines(df, week, season)

            print(f"📊 Loaded {len(df)} props across {df['player_name'].nunique()} players")

            return df

        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            print("Using fallback lines...")
            return self._get_fallback_lines(week, season)

        except Exception as e:
            print(f"❌ Error processing API response: {e}")
            return pd.DataFrame()

    def get_historical_lines(
        self,
        player_name: str,
        market: str,
        weeks: List[int] = None,
        season: int = None
    ) -> pd.DataFrame:
        """
        Get historical lines for a player/market combination.
        Used for tracking line movement and finding edge.

        Args:
            player_name: Player name to look up
            market: Prop market (e.g., 'receiving_yards')
            weeks: List of weeks to retrieve (default: all available)
            season: Season (default: current)

        Returns:
            DataFrame with historical lines sorted by week
        """
        if season is None:
            season = self.config.CURRENT_SEASON

        historical_file = os.path.join(
            self.lines_cache_dir,
            f'historical_lines_s{season}.pkl'
        )

        if not os.path.exists(historical_file):
            print(f"⚠️ No historical lines found for {season}")
            return pd.DataFrame()

        with open(historical_file, 'rb') as f:
            historical = pickle.load(f)

        # Filter to player and market
        player_lines = historical[
            (historical['player_name'].str.contains(player_name, case=False, na=False)) &
            (historical['market'] == market)
        ]

        if weeks:
            player_lines = player_lines[player_lines['week'].isin(weeks)]

        return player_lines.sort_values('week')

    def get_best_line(self, props_df: pd.DataFrame, player_name: str, market: str) -> Optional[Dict]:
        """
        Get the best available line for a player/market from multiple sportsbooks.

        Args:
            props_df: Props DataFrame from get_player_props()
            player_name: Player to search for
            market: Market type

        Returns:
            Dictionary with best line info, or None if not found
        """
        # Filter to player and market
        matches = props_df[
            (props_df['player_name'].str.contains(player_name, case=False, na=False)) &
            (props_df['market'] == market)
        ]

        if matches.empty:
            return None

        # Prefer books in preferred list
        preferred = matches[matches['sportsbook'].isin(self.config.PREFERRED_SPORTSBOOKS)]

        if not preferred.empty:
            matches = preferred

        # Get best line (most favorable odds)
        # For now, just return first match from preferred book
        # Could enhance to find actual best odds
        best = matches.iloc[0]

        return {
            'player_name': best['player_name'],
            'team': best.get('team', ''),
            'market': best['market'],
            'line': best['line'],
            'over_odds': best.get('over_odds', -110),
            'under_odds': best.get('under_odds', -110),
            'sportsbook': best['sportsbook'],
            'timestamp': best.get('timestamp', '')
        }

    def calculate_line_movement(
        self,
        player_name: str,
        market: str,
        current_line: float,
        lookback_weeks: int = 3
    ) -> Dict:
        """
        Calculate line movement compared to recent history.
        Helps identify sharp action and market signals.

        Args:
            player_name: Player name
            market: Market type
            current_line: Current line value
            lookback_weeks: How many weeks to look back

        Returns:
            Dictionary with movement metrics
        """
        historical = self.get_historical_lines(player_name, market)

        if historical.empty:
            return {
                'has_history': False,
                'movement': 0,
                'avg_historical_line': current_line,
                'weeks_sampled': 0
            }

        # Get recent history
        recent = historical.tail(lookback_weeks)

        avg_line = recent['line'].mean()
        movement = current_line - avg_line

        return {
            'has_history': True,
            'movement': movement,
            'avg_historical_line': avg_line,
            'current_line': current_line,
            'weeks_sampled': len(recent),
            'trend': 'up' if movement > 0.5 else 'down' if movement < -0.5 else 'stable'
        }

    def _save_to_historical_lines(self, df: pd.DataFrame, week: int, season: int):
        """Save current lines to historical tracking file"""
        historical_file = os.path.join(
            self.lines_cache_dir,
            f'historical_lines_s{season}.pkl'
        )

        if os.path.exists(historical_file):
            with open(historical_file, 'rb') as f:
                historical = pickle.load(f)
            # Append new data
            historical = pd.concat([historical, df], ignore_index=True)
            # Remove duplicates (keep latest)
            historical = historical.drop_duplicates(
                subset=['player_name', 'market', 'week', 'sportsbook'],
                keep='last'
            )
        else:
            historical = df

        with open(historical_file, 'wb') as f:
            pickle.dump(historical, f)

        print(f"💾 Saved {len(df)} lines to historical cache")

    def _get_fallback_lines(self, week: int, season: int) -> pd.DataFrame:
        """
        Fallback when API calls exhausted.
        Tries to use previous week's lines or historical averages.
        """
        print(f"⚠️ Using fallback lines for Week {week}")

        # Try to load previous week's cache
        for days_back in range(1, 8):
            date_str = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
            cache_file = os.path.join(
                self.lines_cache_dir,
                f'props_s{season}_w{week}_{date_str}.pkl'
            )
            if os.path.exists(cache_file):
                print(f"📦 Using cached data from {days_back} days ago")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        # Try previous week
        prev_week_cache = os.path.join(
            self.lines_cache_dir,
            f'props_s{season}_w{week-1}_*.pkl'
        )

        import glob
        prev_files = glob.glob(prev_week_cache)
        if prev_files:
            print(f"📦 Using previous week's lines as fallback")
            with open(prev_files[-1], 'rb') as f:
                df = pickle.load(f)
                df['week'] = week  # Update week
                return df

        print("❌ No fallback data available")
        return pd.DataFrame()

    def get_api_usage_stats(self) -> Dict:
        """Get current API usage statistics"""
        return {
            'calls_used': self.config.SPORTSGAMEODDS_CALLS_USED,
            'calls_remaining': self.config.SPORTSGAMEODDS_MONTHLY_LIMIT - self.config.SPORTSGAMEODDS_CALLS_USED,
            'monthly_limit': self.config.SPORTSGAMEODDS_MONTHLY_LIMIT,
            'usage_percent': (self.config.SPORTSGAMEODDS_CALLS_USED / self.config.SPORTSGAMEODDS_MONTHLY_LIMIT) * 100
        }


if __name__ == "__main__":
    # Test the client
    client = SportsGameOddsClient()

    print("\n" + "="*60)
    print("TESTING SPORTSGAMEODDS CLIENT")
    print("="*60)

    # Check usage
    usage = client.get_api_usage_stats()
    print(f"\nAPI Usage: {usage['calls_used']}/{usage['monthly_limit']} ({usage['usage_percent']:.1f}%)")
    print(f"Remaining: {usage['calls_remaining']} calls")

    # Fetch props (will use cache if available)
    props = client.get_player_props(week=8, force_refresh=False)

    if not props.empty:
        print(f"\n✅ Loaded {len(props)} props")
        print(f"\nMarkets available:")
        print(props['market'].value_counts())

        print(f"\nSportsbooks:")
        print(props['sportsbook'].value_counts())

        print(f"\nSample props:")
        print(props.head(10)[['player_name', 'team', 'market', 'line', 'sportsbook']])
    else:
        print("\n⚠️ No props loaded")
