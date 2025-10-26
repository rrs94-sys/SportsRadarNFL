"""
Hybrid Data Ingestion: nfl_data_py (historical) + Tank01 (current season)

This approach gives you:
- Deep historical data (2022-2024) for free
- Live current season data (2025) with real-time updates
- Better training: 3 complete seasons + current week
"""

import nfl_data_py as nfl
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

class HybridDataIngestion:
    """Combines nfl_data_py historical + Tank01 current season"""
    
    def __init__(self, config):
        self.config = config
        self.historical_data = {}
        self.current_season_data = {}
        self.combined_data = {}
        
    def load_all_training_data(self, current_season=2025, current_week=7):
        """
        Load training data for predicting current_week + 1
        
        Strategy:
        - Historical: 2022-2024 complete seasons (nfl_data_py)
        - Current: 2025 weeks 1-7 (Tank01)
        - Result: Train on all of this to predict Week 8
        """
        print("="*70)
        print("HYBRID DATA INGESTION")
        print("="*70)
        
        # Step 1: Load historical baseline (FREE)
        print("\n[1/3] Loading historical data (2022-2024) from nfl_data_py...")
        self.historical_data = self._load_historical_baseline()
        
        # Step 2: Load current season (PAID API)
        print("\n[2/3] Loading current season (2025, Weeks 1-7) from Tank01...")
        self.current_season_data = self._load_current_season_tank01(
            season=current_season,
            through_week=current_week
        )
        
        # Step 3: Combine into unified format
        print("\n[3/3] Combining data sources...")
        self.combined_data = self._combine_data_sources()
        
        print(f"\n✅ Training data ready!")
        print(f"   Total player-game records: {len(self.combined_data['weekly'])}")
        print(f"   Historical (2022-2024): {len(self.historical_data['weekly'])}")
        print(f"   Current (2025 W1-7): {len(self.current_season_data['weekly'])}")
        
        return self.combined_data
    
    def _load_historical_baseline(self):
        """Load 2022-2024 from nfl_data_py (free, reliable)"""
        historical_seasons = [2022, 2023, 2024]
        
        data = {}
        
        # Weekly stats
        print("  Loading weekly stats...")
        data['weekly'] = nfl.import_weekly_data(historical_seasons)
        print(f"    ✓ {len(data['weekly'])} records")
        
        # Play-by-play
        print("  Loading play-by-play...")
        data['pbp'] = nfl.import_pbp_data(historical_seasons)
        print(f"    ✓ {len(data['pbp'])} plays")
        
        # Schedules
        print("  Loading schedules...")
        data['schedules'] = nfl.import_schedules(historical_seasons)
        print(f"    ✓ {len(data['schedules'])} games")
        
        # Rosters
        print("  Loading rosters...")
        data['rosters'] = nfl.import_seasonal_rosters(historical_seasons)
        print(f"    ✓ {len(data['rosters'])} player-seasons")
        
        # Snap counts
        print("  Loading snap counts...")
        data['snap_counts'] = nfl.import_snap_counts(historical_seasons)
        print(f"    ✓ {len(data['snap_counts'])} records")
        
        return data
    
    def _load_current_season_tank01(self, season, through_week):
        """
        Load current season data from Tank01 API
        
        This is the KEY improvement - gets you live 2025 data
        """
        data = {
            'weekly': [],
            'pbp': [],
            'schedules': [],
            'snap_counts': []
        }
        
        # Load week by week to avoid rate limits
        for week in range(1, through_week + 1):
            print(f"  Fetching Week {week}...")
            
            try:
                # Get weekly stats
                week_stats = self._fetch_tank01_weekly_stats(season, week)
                if week_stats:
                    data['weekly'].extend(week_stats)
                
                # Get box scores (for play-by-play equivalent)
                week_games = self._fetch_tank01_week_games(season, week)
                if week_games:
                    data['pbp'].extend(week_games)
                
                # Rate limiting (critical!)
                time.sleep(2)  # 2 second delay between weeks
                
            except Exception as e:
                print(f"    ⚠️  Error fetching Week {week}: {e}")
                continue
        
        # Convert to DataFrames
        data['weekly'] = pd.DataFrame(data['weekly'])
        data['pbp'] = pd.DataFrame(data['pbp'])
        
        # Get schedule once (not week-by-week)
        data['schedules'] = self._fetch_tank01_schedule(season)
        
        print(f"  ✓ Current season: {len(data['weekly'])} player-games")
        
        return data
    
    def _fetch_tank01_weekly_stats(self, season, week):
        """
        Fetch weekly player stats from Tank01
        
        Endpoint: getNFLPlayerStats
        """
        url = f"https://{self.config.RAPIDAPI_HOST}/getNFLTeamSchedule"
        
        headers = {
            'x-rapidapi-host': self.config.RAPIDAPI_HOST,
            'x-rapidapi-key': self.config.RAPIDAPI_KEY
        }
        
        params = {
            'season': str(season),
            'week': str(week)
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            games = response.json().get('body', [])
            
            # Extract player stats from each game
            player_stats = []
            for game in games:
                # Parse box score for player stats
                # This requires game-specific endpoint calls
                game_id = game.get('gameID')
                if game_id:
                    box_score = self._fetch_tank01_box_score(game_id)
                    if box_score:
                        player_stats.extend(self._parse_box_score_to_weekly(box_score, week, season))
            
            return player_stats
            
        except Exception as e:
            print(f"      Error: {e}")
            return []
    
    def _fetch_tank01_box_score(self, game_id):
        """Fetch detailed box score with player stats"""
        url = f"https://{self.config.RAPIDAPI_HOST}/getNFLBoxScore"
        
        headers = {
            'x-rapidapi-host': self.config.RAPIDAPI_HOST,
            'x-rapidapi-key': self.config.RAPIDAPI_KEY
        }
        
        params = {
            'gameID': game_id,
            'fantasyPoints': 'true',
            'twoPointConversions': 'true',
            'playByPlay': 'false'  # Don't need PBP for weekly stats
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json().get('body', {})
        except:
            return None
    
    def _parse_box_score_to_weekly(self, box_score, week, season):
        """Convert Tank01 box score to nfl_data_py weekly format"""
        player_stats = []
        
        # Parse both teams
        for team_key in ['home', 'away']:
            team_data = box_score.get(team_key, {})
            team_abbr = team_data.get('team', '')
            
            # Passing stats
            passing = team_data.get('passing', [])
            for p in passing:
                player_stats.append({
                    'player_id': p.get('playerID'),
                    'player_name': p.get('longName'),
                    'position': 'QB',
                    'recent_team': team_abbr,
                    'season': season,
                    'week': week,
                    'attempts': int(p.get('passAttempts', 0)),
                    'completions': int(p.get('passCompletions', 0)),
                    'passing_yards': int(p.get('passYards', 0)),
                    'passing_tds': int(p.get('passTD', 0)),
                    'interceptions': int(p.get('int', 0)),
                    'sacks': int(p.get('sacks', 0)),
                    'sack_yards': int(p.get('sackYards', 0)),
                })
            
            # Receiving stats
            receiving = team_data.get('receiving', [])
            for r in receiving:
                player_stats.append({
                    'player_id': r.get('playerID'),
                    'player_name': r.get('longName'),
                    'position': r.get('pos', 'WR'),
                    'recent_team': team_abbr,
                    'season': season,
                    'week': week,
                    'targets': int(r.get('targets', 0)),
                    'receptions': int(r.get('receptions', 0)),
                    'receiving_yards': int(r.get('recYards', 0)),
                    'receiving_tds': int(r.get('recTD', 0)),
                })
            
            # Rushing stats
            rushing = team_data.get('rushing', [])
            for ru in rushing:
                # Find existing player stat or create new
                existing = next((p for p in player_stats 
                               if p.get('player_id') == ru.get('playerID')), None)
                
                rush_data = {
                    'carries': int(ru.get('rushAttempts', 0)),
                    'rushing_yards': int(ru.get('rushYards', 0)),
                    'rushing_tds': int(ru.get('rushTD', 0)),
                }
                
                if existing:
                    existing.update(rush_data)
                else:
                    player_stats.append({
                        'player_id': ru.get('playerID'),
                        'player_name': ru.get('longName'),
                        'position': ru.get('pos', 'RB'),
                        'recent_team': team_abbr,
                        'season': season,
                        'week': week,
                        **rush_data
                    })
        
        return player_stats
    
    def _fetch_tank01_week_games(self, season, week):
        """Fetch game-level data for play-by-play equivalent"""
        # This would be similar to above but focuses on game-level context
        # For now, return empty as PBP is less critical
        return []
    
    def _fetch_tank01_schedule(self, season):
        """Fetch full season schedule"""
        url = f"https://{self.config.RAPIDAPI_HOST}/getNFLTeamSchedule"
        
        headers = {
            'x-rapidapi-host': self.config.RAPIDAPI_HOST,
            'x-rapidapi-key': self.config.RAPIDAPI_KEY
        }
        
        params = {
            'teamAbv': 'ALL',
            'season': str(season)
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            schedule_data = response.json().get('body', [])
            
            # Convert to nfl_data_py format
            schedule = []
            for game in schedule_data:
                schedule.append({
                    'game_id': game.get('gameID'),
                    'season': season,
                    'week': int(game.get('gameWeek', 0)),
                