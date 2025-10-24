"""
Tank API Client for NFL Injury Data
Uses /getNFLNews endpoint to parse injury information from news titles
"""

import requests
import pandas as pd
import pickle
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from phase1_config import Config


class TankInjuryClient:
    """
    Client for Tank NFL News API (via RapidAPI).
    Parses news titles for injury-related keywords.
    """

    # Injury keywords to look for in news titles
    INJURY_KEYWORDS = {
        'out': 0.0,          # Player is out
        'inactive': 0.0,
        'ruled out': 0.0,
        'doubtful': 0.25,    # Very unlikely to play
        'questionable': 0.75, # 50/50 chance
        'limited': 0.85,     # Practicing but limited
        'probable': 0.95,    # Likely to play
        'returned': 1.0,     # Returned to practice
        'cleared': 1.0,      # Cleared to play
        'active': 1.0        # Active
    }

    def __init__(self, config: Config = Config):
        self.config = config
        self.api_key = config.TANK_API_KEY
        self.base_url = config.TANK_BASE_URL
        self.cache_dir = config.CACHE_DIR

    def get_injury_report(
        self,
        week: int = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch current injury report by parsing news.

        Args:
            week: Specific week (default: current week)
            force_refresh: Force new API call

        Returns:
            DataFrame with injury information
        """
        cache_file = os.path.join(
            self.config.CACHE_DIR,
            f'injuries_{datetime.now().strftime("%Y%m%d")}.pkl'
        )

        # Use cache if available and recent (same day)
        if not force_refresh and os.path.exists(cache_file):
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if file_age < timedelta(hours=12):
                print("📦 Using cached injury data")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        print("🌐 Fetching injury data from Tank API (via news)...")

        try:
            url = f"{self.base_url}/getNFLNews"
            headers = {
                'X-RapidAPI-Key': self.api_key,
                'X-RapidAPI-Host': 'tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com'
            }

            params = {
                'fantasyNews': 'true',
                'maxItems': '100'  # Get more items for better coverage
            }

            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            news_items = data.get('body', [])

            if not news_items:
                print("⚠️ No news data returned")
                return self._get_cached_injuries()

            # Parse news titles for injury information
            injuries = self._parse_injury_news(news_items)

            if injuries.empty:
                print("⚠️ No injury-related news found")
                return self._get_cached_injuries()

            # Cache it
            with open(cache_file, 'wb') as f:
                pickle.dump(injuries, f)

            print(f"✅ Parsed injury data for {len(injuries)} players from news")

            return injuries

        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            return self._get_cached_injuries()

        except Exception as e:
            print(f"❌ Error processing injury data: {e}")
            import traceback
            traceback.print_exc()
            return self._get_cached_injuries()

    def _parse_injury_news(self, news_items: List[Dict]) -> pd.DataFrame:
        """
        Parse news titles for injury information

        Args:
            news_items: List of news items from API

        Returns:
            DataFrame with parsed injury data
        """
        injuries = []

        for item in news_items:
            title = item.get('title', '')

            # Check if title contains injury keywords
            injury_status, impact = self._extract_injury_status(title)

            if injury_status:
                # Extract player name (usually before the colon)
                player_name = self._extract_player_name(title)

                if player_name:
                    injuries.append({
                        'player_name': player_name,
                        'status': injury_status,
                        'injury_impact': impact,
                        'injury_description': title.split(':', 1)[1].strip() if ':' in title else title,
                        'source': 'news',
                        'fetched_at': datetime.now().isoformat()
                    })

        df = pd.DataFrame(injuries)

        # Remove duplicates (keep first occurrence - most recent)
        if not df.empty and 'player_name' in df.columns:
            df = df.drop_duplicates(subset=['player_name'], keep='first')

        return df

    def _extract_injury_status(self, title: str) -> tuple:
        """
        Extract injury status from news title

        Args:
            title: News title

        Returns:
            Tuple of (status_string, impact_score)
        """
        title_lower = title.lower()

        # Check for each keyword
        for keyword, impact in self.INJURY_KEYWORDS.items():
            if keyword in title_lower:
                return (keyword, impact)

        # Also check for body part mentions that suggest injury
        body_parts = ['ankle', 'knee', 'shoulder', 'back', 'hamstring', 'concussion',
                     'hip', 'groin', 'foot', 'hand', 'wrist', 'elbow', 'chest', 'rib']

        for part in body_parts:
            if part in title_lower:
                # If body part mentioned but no clear status, assume questionable
                return ('questionable', 0.75)

        return (None, None)

    def _extract_player_name(self, title: str) -> Optional[str]:
        """
        Extract player name from news title

        Args:
            title: News title (format usually "Player Name: description")

        Returns:
            Player name or None
        """
        # Most titles are formatted as "Player Name: description"
        if ':' in title:
            name = title.split(':')[0].strip()

            # Validate it's a name (has at least first and last name)
            if len(name.split()) >= 2:
                return name

        return None

    def get_player_injury_status(self, player_name: str) -> Dict:
        """
        Get injury status for a specific player.

        Args:
            player_name: Player to look up

        Returns:
            Dictionary with injury info
        """
        injuries = self.get_injury_report()

        if injuries.empty:
            return {
                'status': 'Active',
                'injury_impact': 1.0,
                'injury_description': None
            }

        # Find player
        player_injury = injuries[
            injuries['player_name'].str.contains(player_name, case=False, na=False)
        ]

        if player_injury.empty:
            return {
                'status': 'Active',
                'injury_impact': 1.0,
                'injury_description': None
            }

        # Get most recent entry if multiple
        injury = player_injury.iloc[0]

        return {
            'status': injury.get('status', 'Active'),
            'injury_impact': injury.get('injury_impact', 1.0),
            'injury_description': injury.get('injury_description', None)
        }

    def _get_cached_injuries(self) -> pd.DataFrame:
        """Get most recent cached injury data"""
        import glob

        cache_pattern = os.path.join(self.cache_dir, 'injuries_*.pkl')
        cache_files = glob.glob(cache_pattern)

        if not cache_files:
            print("⚠️ No cached injury data available")
            return pd.DataFrame()

        # Get most recent
        latest = max(cache_files, key=os.path.getmtime)
        print(f"📦 Using cached injury data from {os.path.basename(latest)}")

        with open(latest, 'rb') as f:
            return pickle.load(f)


if __name__ == "__main__":
    # Test the client
    client = TankInjuryClient()

    print("\n" + "="*60)
    print("TESTING TANK INJURY CLIENT (NEWS PARSER)")
    print("="*60)

    # Fetch injury report
    injuries = client.get_injury_report(force_refresh=True)

    if not injuries.empty:
        print(f"\n✅ Parsed injury data for {len(injuries)} players")

        print(f"\nInjury Status Distribution:")
        if 'status' in injuries.columns:
            print(injuries['status'].value_counts())

        print(f"\nSample injuries:")
        cols = ['player_name', 'status', 'injury_impact', 'injury_description']
        available_cols = [c for c in cols if c in injuries.columns]
        print(injuries[available_cols].head(15))
    else:
        print("\n⚠️ No injury data loaded")
