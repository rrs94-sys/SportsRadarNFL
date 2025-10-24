"""
Open-Meteo Weather API Client
Fetches game-day weather forecasts for NFL stadiums
"""

import requests
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from phase1_config import Config


class WeatherClient:
    """
    Client for Open-Meteo Weather API (no API key required).
    Provides game-day weather forecasts for NFL stadiums.
    """

    def __init__(self, config: Config = Config):
        self.config = config
        self.base_url = config.OPENMETEO_BASE_URL
        self.cache_dir = config.CACHE_DIR
        self.stadium_coords = config.STADIUM_COORDS
        self.dome_stadiums = config.DOME_STADIUMS

    def get_game_weather(
        self,
        team: str,
        game_date: datetime,
        is_home: bool = True
    ) -> Dict:
        """
        Get weather forecast for a specific game.

        Args:
            team: Team abbreviation
            game_date: Date of the game
            is_home: Whether team is home (determines stadium)

        Returns:
            Dictionary with weather data
        """
        # Check if dome stadium
        if team in self.dome_stadiums:
            return self._get_dome_weather()

        # Get stadium coordinates
        coords = self.stadium_coords.get(team)
        if not coords:
            print(f"⚠️ No coordinates found for {team}, using default weather")
            return self._get_default_weather()

        # Check cache
        cache_key = f"{team}_{game_date.strftime('%Y%m%d')}"
        cached = self._get_cached_weather(cache_key)
        if cached:
            return cached

        # Fetch from API
        print(f"🌐 Fetching weather for {team} on {game_date.strftime('%Y-%m-%d')}...")

        try:
            lat, lon = coords
            params = {
                'latitude': lat,
                'longitude': lon,
                'hourly': 'temperature_2m,wind_speed_10m,precipitation_probability,weather_code',
                'temperature_unit': 'fahrenheit',
                'wind_speed_unit': 'mph',
                'timezone': 'America/New_York',
                'start_date': game_date.strftime('%Y-%m-%d'),
                'end_date': game_date.strftime('%Y-%m-%d')
            }

            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Extract hourly data
            hourly = data.get('hourly', {})
            times = hourly.get('time', [])
            temps = hourly.get('temperature_2m', [])
            winds = hourly.get('wind_speed_10m', [])
            precip_prob = hourly.get('precipitation_probability', [])
            weather_codes = hourly.get('weather_code', [])

            if not times:
                return self._get_default_weather()

            # Find game time (assume 1 PM ET for most games, adjust as needed)
            # For simplicity, take midday values (index ~13)
            game_hour_idx = min(13, len(times) - 1)

            weather = {
                'is_dome': False,
                'temp_f': temps[game_hour_idx] if temps else 72,
                'wind_mph': winds[game_hour_idx] if winds else 0,
                'precipitation_prob': precip_prob[game_hour_idx] if precip_prob else 0,
                'weather_code': weather_codes[game_hour_idx] if weather_codes else 0,
                'conditions': self._interpret_weather_code(weather_codes[game_hour_idx] if weather_codes else 0),
                'wind_impact': self._calculate_wind_impact(winds[game_hour_idx] if winds else 0),
                'temp_impact': self._calculate_temp_impact(temps[game_hour_idx] if temps else 72),
                'fetched_at': datetime.now().isoformat()
            }

            # Cache it
            self._cache_weather(cache_key, weather)

            return weather

        except requests.exceptions.RequestException as e:
            print(f"❌ Weather API request failed: {e}")
            return self._get_default_weather()

        except Exception as e:
            print(f"❌ Error processing weather data: {e}")
            return self._get_default_weather()

    def get_weekly_weather(
        self,
        schedules: pd.DataFrame,
        week: int
    ) -> pd.DataFrame:
        """
        Get weather for all games in a specific week.

        Args:
            schedules: Schedules DataFrame
            week: Week number

        Returns:
            DataFrame with weather for each game
        """
        week_games = schedules[schedules['week'] == week].copy()

        weather_data = []

        for idx, game in week_games.iterrows():
            home_team = game['home_team']
            game_date = pd.to_datetime(game['gameday'])

            weather = self.get_game_weather(home_team, game_date, is_home=True)
            weather['game_id'] = game.get('game_id', '')
            weather['home_team'] = home_team
            weather['away_team'] = game['away_team']
            weather['week'] = week

            weather_data.append(weather)

        return pd.DataFrame(weather_data)

    def _get_dome_weather(self) -> Dict:
        """Return ideal conditions for dome stadiums"""
        return {
            'is_dome': True,
            'temp_f': 72,
            'wind_mph': 0,
            'precipitation_prob': 0,
            'weather_code': 0,
            'conditions': 'clear',
            'wind_impact': 0.0,
            'temp_impact': 0.0,
            'fetched_at': datetime.now().isoformat()
        }

    def _get_default_weather(self) -> Dict:
        """Return neutral weather conditions as fallback"""
        return {
            'is_dome': False,
            'temp_f': 65,
            'wind_mph': 5,
            'precipitation_prob': 10,
            'weather_code': 0,
            'conditions': 'partly_cloudy',
            'wind_impact': 0.0,
            'temp_impact': 0.0,
            'fetched_at': datetime.now().isoformat()
        }

    def _interpret_weather_code(self, code: int) -> str:
        """
        Interpret WMO weather code.
        0 = Clear, 1-3 = Partly cloudy, 45/48 = Fog, 51-99 = Precipitation
        """
        if code == 0:
            return 'clear'
        elif code <= 3:
            return 'partly_cloudy'
        elif code in [45, 48]:
            return 'fog'
        elif code <= 69:
            return 'rain'
        elif code <= 79:
            return 'snow'
        elif code <= 99:
            return 'thunderstorm'
        else:
            return 'unknown'

    def _calculate_wind_impact(self, wind_mph: float) -> float:
        """
        Calculate wind impact on passing game.
        Returns negative impact factor (0 = no impact, -1 = severe impact)
        """
        if wind_mph < self.config.WIND_THRESHOLD_MPH:
            return 0.0
        elif wind_mph < 20:
            return -0.1  # Minor impact
        elif wind_mph < 25:
            return -0.2  # Moderate impact
        else:
            return -0.35  # Severe impact

    def _calculate_temp_impact(self, temp_f: float) -> float:
        """
        Calculate temperature impact on performance.
        Returns impact factor (positive or negative)
        """
        if temp_f < self.config.TEMP_COLD_THRESHOLD_F:
            # Cold weather: -0.05 to -0.15 impact
            return max(-0.15, (temp_f - 32) / 320)
        elif temp_f > self.config.TEMP_HOT_THRESHOLD_F:
            # Hot weather: -0.05 to -0.1 impact
            return min(-0.05, (85 - temp_f) / 200)
        else:
            return 0.0  # Neutral temperature

    def _get_cached_weather(self, cache_key: str) -> Optional[Dict]:
        """Get cached weather data if available and recent"""
        cache_file = os.path.join(self.cache_dir, f'weather_{cache_key}.pkl')

        if os.path.exists(cache_file):
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if file_age < timedelta(hours=6):  # Cache for 6 hours
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        return None

    def _cache_weather(self, cache_key: str, weather: Dict):
        """Cache weather data"""
        cache_file = os.path.join(self.cache_dir, f'weather_{cache_key}.pkl')
        with open(cache_file, 'wb') as f:
            pickle.dump(weather, f)


if __name__ == "__main__":
    # Test the client
    client = WeatherClient()

    print("\n" + "="*60)
    print("TESTING WEATHER CLIENT")
    print("="*60)

    # Test dome stadium
    print("\n1. Dome Stadium (DET):")
    dome_weather = client.get_game_weather('DET', datetime.now(), is_home=True)
    print(f"   Dome: {dome_weather['is_dome']}")
    print(f"   Temp: {dome_weather['temp_f']}°F")
    print(f"   Wind: {dome_weather['wind_mph']} mph")

    # Test outdoor stadium
    print("\n2. Outdoor Stadium (GB):")
    outdoor_weather = client.get_game_weather('GB', datetime.now() + timedelta(days=3), is_home=True)
    print(f"   Dome: {outdoor_weather['is_dome']}")
    print(f"   Temp: {outdoor_weather['temp_f']}°F")
    print(f"   Wind: {outdoor_weather['wind_mph']} mph")
    print(f"   Conditions: {outdoor_weather['conditions']}")
    print(f"   Wind Impact: {outdoor_weather['wind_impact']}")
    print(f"   Temp Impact: {outdoor_weather['temp_impact']}")
