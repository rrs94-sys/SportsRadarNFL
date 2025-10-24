"""
Comprehensive Feature Engineering for NFL Player Props
70+ base features + enhanced usage, opponent, context features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from phase1_config import Config
from datetime import datetime, timedelta


class Phase1FeatureEngineer:
    """
    Feature engineering with recency weighting and advanced metrics.
    Implements 70+ features optimized for market prediction.
    """

    def __init__(self, data: Dict, config: Config = Config):
        self.config = config
        self.data = data
        self.player_weeks = data['player_weeks']
        self.schedules = data['schedules']
        self.rosters = data.get('rosters', pd.DataFrame())

        # Optional external data (injected by pipeline)
        self.injuries = None
        self.weather = None
        self.lines = None

    def set_external_data(
        self,
        injuries: pd.DataFrame = None,
        weather: pd.DataFrame = None,
        lines: pd.DataFrame = None
    ):
        """Inject external data sources"""
        self.injuries = injuries
        self.weather = weather
        self.lines = lines

    def create_receiving_features(
        self,
        player_id: str,
        team: str,
        week: int,
        season: int,
        opponent: str
    ) -> Dict:
        """
        Create comprehensive receiving features for WR/TE/RB.

        Returns:
            Dictionary with 70+ features
        """
        features = {}

        # Get player history
        history = self._get_player_history(player_id, week, season)

        if history.empty or len(history) < 1:
            return self._get_default_receiving_features()

        # === CORE RECENCY FEATURES (20 features) ===
        features.update(self._calculate_recency_features(
            history, ['receptions', 'receiving_yards', 'receiving_tds', 'targets']
        ))

        # === VOLATILITY FEATURES (4 features) ===
        features.update(self._calculate_volatility_features(
            history, ['receptions', 'receiving_yards']
        ))

        # === HOME/ROAD SPLITS (4 features) ===
        features.update(self._calculate_home_road_features(
            history, 'receptions', team, week, season
        ))

        # === REST & TRAVEL (3 features) ===
        features.update(self._calculate_rest_travel_features(
            history, team, opponent, week, season
        ))

        # === MATCHUP HISTORY (3 features) ===
        features.update(self._calculate_matchup_features(
            history, opponent, 'receptions'
        ))

        # === USAGE FEATURES (10 features) ===
        features.update(self._calculate_usage_features(
            player_id, team, history, season
        ))

        # === OPPONENT DEFENSE (8 features) ===
        features.update(self._calculate_opponent_defense_features(
            opponent, 'WR', week, season
        ))

        # === WEATHER FEATURES (4 features) ===
        features.update(self._calculate_weather_features(
            team, opponent, week, season
        ))

        # === TREND FEATURES (3 features) ===
        features.update(self._calculate_trend_features(
            history, ['targets', 'receptions']
        ))

        # === GAME CONTEXT FEATURES (8 features) ===
        features.update(self._calculate_game_context_features(
            team, opponent, week, season
        ))

        # === INJURY IMPACT FEATURES (3 features) ===
        features.update(self._calculate_injury_features(
            player_id, team, opponent
        ))

        # === RED ZONE FEATURES (3 features) ===
        features.update(self._calculate_red_zone_features(
            player_id, history
        ))

        # === TWO-MINUTE DRILL / HURRY-UP (3 features) ===
        features.update(self._calculate_situational_usage_features(
            player_id, history
        ))

        # === MARKET FEATURES (2 features) ===
        features.update(self._calculate_market_features(
            player_id, 'receptions', week
        ))

        return features

    def create_qb_features(
        self,
        player_id: str,
        team: str,
        week: int,
        season: int,
        opponent: str
    ) -> Dict:
        """Create QB-specific features"""
        features = {}

        history = self._get_player_history(player_id, week, season)

        if history.empty or len(history) < 1:
            return self._get_default_qb_features()

        # Core QB stats
        qb_stats = ['completions', 'attempts', 'passing_yards', 'passing_tds']

        features.update(self._calculate_recency_features(history, qb_stats))
        features.update(self._calculate_volatility_features(history, ['passing_yards']))
        features.update(self._calculate_home_road_features(history, 'completions', team, week, season))
        features.update(self._calculate_opponent_defense_features(opponent, 'QB', week, season))
        features.update(self._calculate_weather_features(team, opponent, week, season))
        features.update(self._calculate_game_context_features(team, opponent, week, season))
        features.update(self._calculate_injury_features(player_id, team, opponent))

        # QB-specific: pressure rate, sack rate
        if len(history) > 0:
            features['avg_sacks_taken'] = history['sacks'].mean()
            features['sack_rate'] = history['sacks'].sum() / max(history['attempts'].sum(), 1)

        return features

    def create_rushing_features(
        self,
        player_id: str,
        team: str,
        week: int,
        season: int,
        opponent: str
    ) -> Dict:
        """Create rushing-specific features"""
        features = {}

        history = self._get_player_history(player_id, week, season)

        if history.empty or len(history) < 1:
            return self._get_default_rushing_features()

        rushing_stats = ['carries', 'rushing_yards', 'rushing_tds']

        features.update(self._calculate_recency_features(history, rushing_stats))
        features.update(self._calculate_volatility_features(history, ['rushing_yards']))
        features.update(self._calculate_home_road_features(history, 'carries', team, week, season))
        features.update(self._calculate_opponent_defense_features(opponent, 'RB', week, season))
        features.update(self._calculate_game_context_features(team, opponent, week, season))
        features.update(self._calculate_injury_features(player_id, team, opponent))

        # RB-specific: game script dependency
        features.update(self._calculate_game_script_features(team, opponent, week, season))

        return features

    # ========================================================================
    # HELPER METHODS - Feature Calculation
    # ========================================================================

    def _get_player_history(
        self,
        player_id: str,
        current_week: int,
        season: int,
        lookback_weeks: int = 8
    ) -> pd.DataFrame:
        """Get player's historical data"""
        history = self.player_weeks[
            (self.player_weeks['player_id'] == player_id) &
            (self.player_weeks['season'] == season) &
            (self.player_weeks['week'] < current_week)
        ].sort_values('week', ascending=False).head(lookback_weeks)

        return history

    def _calculate_recency_features(
        self,
        history: pd.DataFrame,
        stat_cols: List[str]
    ) -> Dict:
        """
        Calculate recency-weighted features.
        L1: 30%, L2-L3: 40%, L4+: 30%
        """
        features = {}

        if len(history) == 0:
            for stat in stat_cols:
                features[f'avg_{stat}_L1'] = 0
                features[f'avg_{stat}_L2_L3'] = 0
                features[f'avg_{stat}_L4_plus'] = 0
                features[f'avg_{stat}_weighted'] = 0
                features[f'std_{stat}'] = 0
            return features

        weights = self.config.RECENCY_WEIGHTS

        for stat in stat_cols:
            if stat not in history.columns:
                continue

            # Split by recency
            L1 = history.iloc[0][stat] if len(history) > 0 else 0
            L2_L3 = history.iloc[1:3][stat].mean() if len(history) > 1 else L1
            L4_plus = history.iloc[3:][stat].mean() if len(history) > 3 else L2_L3

            features[f'avg_{stat}_L1'] = L1
            features[f'avg_{stat}_L2_L3'] = L2_L3
            features[f'avg_{stat}_L4_plus'] = L4_plus

            # Weighted average
            features[f'avg_{stat}_weighted'] = (
                weights['L1'] * L1 +
                weights['L2_L3'] * L2_L3 +
                weights['L4_PLUS'] * L4_plus
            )

            # Standard deviation (volatility proxy)
            features[f'std_{stat}'] = history[stat].std() if len(history) > 1 else 0

        return features

    def _calculate_volatility_features(
        self,
        history: pd.DataFrame,
        stat_cols: List[str]
    ) -> Dict:
        """
        Calculate volatility metrics (CV = coefficient of variation).
        High CV indicates boom/bust player.
        """
        features = {}

        for stat in stat_cols:
            if stat not in history.columns or len(history) < 2:
                features[f'{stat}_cv'] = 0
                features[f'{stat}_is_volatile'] = 0
                continue

            mean_val = history[stat].mean()
            std_val = history[stat].std()

            # Coefficient of variation
            cv = (std_val / mean_val) if mean_val > 0 else 0
            features[f'{stat}_cv'] = cv

            # Binary flag for high volatility
            features[f'{stat}_is_volatile'] = 1.0 if cv > self.config.HIGH_VOLATILITY_THRESHOLD else 0.0

        return features

    def _calculate_home_road_features(
        self,
        history: pd.DataFrame,
        stat: str,
        team: str,
        week: int,
        season: int
    ) -> Dict:
        """Calculate home/away splits"""
        features = {}

        if len(history) == 0 or 'home_away' not in history.columns:
            features[f'avg_{stat}_home'] = 0
            features[f'avg_{stat}_road'] = 0
            features[f'{stat}_home_road_diff'] = 0
            features['is_home_game'] = 0
            return features

        home_games = history[history['home_away'] == 1.0]
        road_games = history[history['home_away'] == 0.0]

        features[f'avg_{stat}_home'] = home_games[stat].mean() if len(home_games) > 0 else history[stat].mean()
        features[f'avg_{stat}_road'] = road_games[stat].mean() if len(road_games) > 0 else history[stat].mean()
        features[f'{stat}_home_road_diff'] = features[f'avg_{stat}_home'] - features[f'avg_{stat}_road']

        # Check if current game is home
        game = self.schedules[
            (self.schedules['season'] == season) &
            (self.schedules['week'] == week) &
            ((self.schedules['home_team'] == team) | (self.schedules['away_team'] == team))
        ]
        features['is_home_game'] = 1.0 if len(game) > 0 and game.iloc[0]['home_team'] == team else 0.0

        return features

    def _calculate_rest_travel_features(
        self,
        history: pd.DataFrame,
        team: str,
        opponent: str,
        week: int,
        season: int
    ) -> Dict:
        """Calculate rest days and travel impact"""
        features = {
            'rest_days': 7,  # Default
            'timezone_change': 0,
            'is_short_week': 0
        }

        if len(history) == 0:
            return features

        # Get last game date
        last_game = history.iloc[0]
        if 'game_date' in last_game:
            last_date = pd.to_datetime(last_game['game_date'])
            current_game = self.schedules[
                (self.schedules['season'] == season) &
                (self.schedules['week'] == week) &
                ((self.schedules['home_team'] == team) | (self.schedules['away_team'] == team))
            ]

            if len(current_game) > 0:
                current_date = pd.to_datetime(current_game.iloc[0]['gameday'])
                rest_days = (current_date - last_date).days
                features['rest_days'] = rest_days
                features['is_short_week'] = 1.0 if rest_days < 6 else 0.0

        # Timezone change
        team_tz = self.config.TEAM_TIMEZONES.get(team, -5)
        opp_tz = self.config.TEAM_TIMEZONES.get(opponent, -5)
        features['timezone_change'] = abs(team_tz - opp_tz)

        return features

    def _calculate_matchup_features(
        self,
        history: pd.DataFrame,
        opponent: str,
        stat: str
    ) -> Dict:
        """Historical performance vs this opponent"""
        features = {}

        vs_opponent = history[history['opponent_team'] == opponent]

        features[f'avg_{stat}_vs_opp'] = vs_opponent[stat].mean() if len(vs_opponent) > 0 else 0
        features['games_vs_opp'] = len(vs_opponent)
        features['familiarity_factor'] = min(len(vs_opponent) / 4.0, 1.0)  # Cap at 4 games

        return features

    def _calculate_usage_features(
        self,
        player_id: str,
        team: str,
        history: pd.DataFrame,
        season: int
    ) -> Dict:
        """
        Advanced usage features:
        - Target share, route participation, snap%
        - Targets per route run
        - Air yards share
        - Pass-block rate
        """
        features = {
            'target_share': 0,
            'route_participation_pct': 0,
            'snap_pct': 0,
            'targets_per_route': 0,
            'air_yards_share': 0,
            'red_zone_target_share': 0,
            'yards_after_catch_pct': 0,
            'avg_target_depth': 0,
            'usage_trend': 0,
            'role_consistency': 0
        }

        if len(history) == 0:
            return features

        # Get team totals
        team_history = self.player_weeks[
            (self.player_weeks['team'] == team) &
            (self.player_weeks['season'] == season) &
            (self.player_weeks['week'].isin(history['week']))
        ]

        if len(team_history) == 0:
            return features

        # Target share
        player_targets = history['targets'].sum()
        team_targets = team_history.groupby('week')['targets'].sum().sum()
        features['target_share'] = player_targets / max(team_targets, 1)

        # Air yards share
        if 'receiving_air_yards' in history.columns:
            player_air_yards = history['receiving_air_yards'].sum()
            team_air_yards = team_history['receiving_air_yards'].sum()
            features['air_yards_share'] = player_air_yards / max(team_air_yards, 1)

            # Average depth of target
            features['avg_target_depth'] = player_air_yards / max(player_targets, 1)

        # YAC percentage
        if 'receiving_yac' in history.columns and 'receiving_yards' in history.columns:
            total_yards = history['receiving_yards'].sum()
            total_yac = history['receiving_yac'].sum()
            features['yards_after_catch_pct'] = total_yac / max(total_yards, 1)

        # Usage trend (are targets increasing?)
        if len(history) >= 3:
            recent_targets = history.head(2)['targets'].mean()
            older_targets = history.tail(2)['targets'].mean()
            features['usage_trend'] = (recent_targets - older_targets) / max(older_targets, 1)

        # Role consistency (low std in target share = consistent role)
        weekly_target_share = []
        for week in history['week'].unique():
            week_history = history[history['week'] == week]
            week_team = team_history[team_history['week'] == week]
            player_wk_targets = week_history['targets'].sum()
            team_wk_targets = week_team['targets'].sum()
            weekly_target_share.append(player_wk_targets / max(team_wk_targets, 1))

        features['role_consistency'] = 1 - np.std(weekly_target_share) if len(weekly_target_share) > 1 else 1

        return features

    def _calculate_opponent_defense_features(
        self,
        opponent: str,
        position: str,
        week: int,
        season: int
    ) -> Dict:
        """
        Opponent defensive metrics:
        - Stats allowed to position
        - Coverage tendencies
        - Missed tackle rate
        - Pressure rate (for QBs)
        """
        features = {
            'opp_avg_allowed': 0,
            'opp_yards_allowed': 0,
            'opp_tds_allowed': 0,
            'opp_defensive_rank': 16,  # Neutral
            'opp_coverage_tendency': 0,  # -1 = man, 0 = mixed, 1 = zone
            'opp_pass_rush_rate': 0,
            'opp_defensive_pace': 0,
            'opp_epa_allowed': 0
        }

        # Get opponent's games
        opp_games = self.player_weeks[
            (self.player_weeks['opponent_team'] == opponent) &
            (self.player_weeks['season'] == season) &
            (self.player_weeks['week'] < week)
        ]

        if len(opp_games) == 0:
            return features

        # Filter by position
        position_map = {
            'QB': ['QB'],
            'RB': ['RB'],
            'WR': ['WR', 'TE'],
            'TE': ['TE']
        }
        relevant_positions = position_map.get(position, [position])
        opp_pos_games = opp_games[opp_games['position'].isin(relevant_positions)]

        if len(opp_pos_games) > 0:
            if 'receptions' in opp_pos_games.columns:
                features['opp_avg_allowed'] = opp_pos_games['receptions'].mean()
            if 'receiving_yards' in opp_pos_games.columns:
                features['opp_yards_allowed'] = opp_pos_games['receiving_yards'].mean()
            if 'receiving_tds' in opp_pos_games.columns:
                features['opp_tds_allowed'] = opp_pos_games['receiving_tds'].mean()
            if 'receiving_epa' in opp_pos_games.columns:
                features['opp_epa_allowed'] = opp_pos_games['receiving_epa'].mean()

        # Defensive rank (lower is better defense)
        all_defenses = self.player_weeks[
            (self.player_weeks['season'] == season) &
            (self.player_weeks['week'] < week) &
            (self.player_weeks['position'].isin(relevant_positions))
        ].groupby('opponent_team')['receiving_yards'].mean().rank()

        features['opp_defensive_rank'] = all_defenses.get(opponent, 16)

        return features

    def _calculate_weather_features(
        self,
        team: str,
        opponent: str,
        week: int,
        season: int
    ) -> Dict:
        """Weather impact features"""
        features = {
            'is_dome': 0,
            'wind_mph': 0,
            'temp_f': 72,
            'wind_impact': 0
        }

        # Check if dome
        game = self.schedules[
            (self.schedules['season'] == season) &
            (self.schedules['week'] == week) &
            ((self.schedules['home_team'] == team) | (self.schedules['away_team'] == team))
        ]

        if len(game) == 0:
            return features

        home_team = game.iloc[0]['home_team']

        if home_team in self.config.DOME_STADIUMS:
            features['is_dome'] = 1
            return features

        # If weather data provided
        if self.weather is not None:
            game_weather = self.weather[
                (self.weather['week'] == week) &
                (self.weather['home_team'] == home_team)
            ]

            if len(game_weather) > 0:
                w = game_weather.iloc[0]
                features['is_dome'] = 1 if w.get('is_dome', False) else 0
                features['wind_mph'] = w.get('wind_mph', 0)
                features['temp_f'] = w.get('temp_f', 72)
                features['wind_impact'] = w.get('wind_impact', 0)

        return features

    def _calculate_trend_features(
        self,
        history: pd.DataFrame,
        stat_cols: List[str]
    ) -> Dict:
        """Calculate trending features (increasing/decreasing)"""
        features = {}

        for stat in stat_cols:
            if stat not in history.columns or len(history) < 4:
                features[f'{stat}_trend'] = 0
                features[f'{stat}_momentum'] = 0
                continue

            # Simple trend: recent vs older
            recent = history.head(2)[stat].mean()
            older = history.tail(2)[stat].mean()

            features[f'{stat}_trend'] = (recent - older) / max(older, 1)

            # Momentum: last game vs L2-L3 avg
            last = history.iloc[0][stat]
            prev_avg = history.iloc[1:3][stat].mean()
            features[f'{stat}_momentum'] = (last - prev_avg) / max(prev_avg, 1)

        return features

    def _calculate_game_context_features(
        self,
        team: str,
        opponent: str,
        week: int,
        season: int
    ) -> Dict:
        """
        Game context from betting markets:
        - Spread (implied game script)
        - Total (implied pace)
        - Team implied total
        """
        features = {
            'spread': 0,
            'team_total': 24,  # Neutral
            'game_total': 48,  # Neutral
            'is_favorite': 0,
            'is_underdog': 0,
            'implied_game_script': 0,  # -1 = trailing, 0 = close, 1 = leading
            'pace_expectation': 0,
            'pass_rate_over_expected': 0
        }

        # Get game from schedule
        game = self.schedules[
            (self.schedules['season'] == season) &
            (self.schedules['week'] == week) &
            ((self.schedules['home_team'] == team) | (self.schedules['away_team'] == team))
        ]

        if len(game) == 0:
            return features

        g = game.iloc[0]

        # Spread
        if 'spread_line' in g:
            spread = g['spread_line']
            is_home = (g['home_team'] == team)

            # Adjust spread for team perspective
            team_spread = spread if is_home else -spread

            features['spread'] = team_spread
            features['is_favorite'] = 1.0 if team_spread < -2.5 else 0.0
            features['is_underdog'] = 1.0 if team_spread > 2.5 else 0.0

            # Implied game script
            if team_spread < -7:
                features['implied_game_script'] = 1.0  # Expected to lead
            elif team_spread > 7:
                features['implied_game_script'] = -1.0  # Expected to trail
            else:
                features['implied_game_script'] = 0.0  # Close game

        # Total
        if 'total_line' in g:
            total = g['total_line']
            features['game_total'] = total
            features['team_total'] = total / 2  # Simplified

            # Adjust for spread
            if 'spread' in features:
                features['team_total'] = (total / 2) - (features['spread'] / 2)

            # Pace expectation (higher total = higher pace)
            features['pace_expectation'] = (total - 45) / 10  # Normalized

        return features

    def _calculate_injury_features(
        self,
        player_id: str,
        team: str,
        opponent: str
    ) -> Dict:
        """Injury impact features"""
        features = {
            'player_injury_status': 1.0,  # 1.0 = healthy
            'team_offensive_injuries': 0,
            'opponent_defensive_injuries': 0
        }

        if self.injuries is None:
            return features

        # Player injury status
        player_injuries = self.injuries[
            self.injuries['player_id'] == player_id
        ] if 'player_id' in self.injuries.columns else pd.DataFrame()

        if not player_injuries.empty:
            features['player_injury_status'] = player_injuries.iloc[0].get('injury_impact', 1.0)

        # Team offensive injuries
        team_injuries = self.injuries[
            (self.injuries['team'] == team) &
            (self.injuries['position'].isin(['QB', 'RB', 'WR', 'TE']))
        ]
        features['team_offensive_injuries'] = len(team_injuries)

        # Opponent defensive injuries
        opp_injuries = self.injuries[
            (self.injuries['team'] == opponent) &
            (self.injuries['position'].isin(['CB', 'S', 'LB', 'DL', 'DB']))
        ]
        features['opponent_defensive_injuries'] = len(opp_injuries)

        return features

    def _calculate_red_zone_features(
        self,
        player_id: str,
        history: pd.DataFrame
    ) -> Dict:
        """
        Red zone usage features.
        Note: Requires play-by-play with field position data
        """
        features = {
            'red_zone_targets': 0,
            'red_zone_target_share': 0,
            'red_zone_td_rate': 0
        }

        # This would require play-by-play analysis
        # For now, use TDs as proxy
        if len(history) > 0 and 'receiving_tds' in history.columns:
            features['red_zone_td_rate'] = history['receiving_tds'].sum() / max(len(history), 1)

        return features

    def _calculate_situational_usage_features(
        self,
        player_id: str,
        history: pd.DataFrame
    ) -> Dict:
        """
        2-minute drill and hurry-up usage.
        Requires play-by-play with game situation data
        """
        features = {
            'two_min_usage': 0,
            'hurry_up_usage': 0,
            'clutch_target_share': 0
        }

        # Would need play-by-play with time remaining
        # Placeholder for now

        return features

    def _calculate_market_features(
        self,
        player_id: str,
        market: str,
        week: int
    ) -> Dict:
        """Features from betting market (line movement, etc.)"""
        features = {
            'historical_line_avg': 0,
            'line_movement': 0
        }

        if self.lines is None:
            return features

        # Get historical lines
        player_lines = self.lines[
            (self.lines['player_id'] == player_id) &
            (self.lines['market'] == market) &
            (self.lines['week'] < week)
        ]

        if len(player_lines) > 0:
            features['historical_line_avg'] = player_lines['line'].mean()

            # Line movement (current vs historical)
            if len(player_lines) > 1:
                recent_line = player_lines.iloc[-1]['line']
                older_line = player_lines.iloc[0]['line']
                features['line_movement'] = recent_line - older_line

        return features

    def _calculate_game_script_features(
        self,
        team: str,
        opponent: str,
        week: int,
        season: int
    ) -> Dict:
        """
        Game script dependency (crucial for RB rushing yards).
        Positive game script = leading = more rushes
        """
        features = {
            'expected_game_script': 0,
            'rush_rate_when_leading': 0,
            'rush_rate_when_trailing': 0
        }

        # Get spread
        game = self.schedules[
            (self.schedules['season'] == season) &
            (self.schedules['week'] == week) &
            ((self.schedules['home_team'] == team) | (self.schedules['away_team'] == team))
        ]

        if len(game) > 0 and 'spread_line' in game.iloc[0]:
            spread = game.iloc[0]['spread_line']
            is_home = (game.iloc[0]['home_team'] == team)
            team_spread = spread if is_home else -spread

            # Expected game script based on spread
            features['expected_game_script'] = -team_spread / 14  # Normalized

        return features

    # ========================================================================
    # DEFAULT FEATURES (when insufficient data)
    # ========================================================================

    def _get_default_receiving_features(self) -> Dict:
        """Return default features when no history available"""
        return {f: 0 for f in [
            'avg_receptions_L1', 'avg_receptions_L2_L3', 'avg_receptions_L4_plus',
            'avg_receptions_weighted', 'std_receptions',
            'avg_receiving_yards_L1', 'avg_receiving_yards_L2_L3', 'avg_receiving_yards_L4_plus',
            'avg_receiving_yards_weighted', 'std_receiving_yards',
            'avg_targets_L1', 'avg_targets_L2_L3', 'avg_targets_L4_plus',
            'avg_targets_weighted', 'std_targets',
            'receiving_yards_cv', 'receiving_yards_is_volatile',
            'receptions_cv', 'receptions_is_volatile',
            'avg_receptions_home', 'avg_receptions_road', 'receptions_home_road_diff', 'is_home_game',
            'rest_days', 'timezone_change', 'is_short_week',
            'avg_receptions_vs_opp', 'games_vs_opp', 'familiarity_factor',
            'target_share', 'route_participation_pct', 'snap_pct',
            'targets_per_route', 'air_yards_share', 'red_zone_target_share',
            'yards_after_catch_pct', 'avg_target_depth', 'usage_trend', 'role_consistency',
            'opp_avg_allowed', 'opp_yards_allowed', 'opp_tds_allowed',
            'opp_defensive_rank', 'opp_coverage_tendency', 'opp_pass_rush_rate',
            'opp_defensive_pace', 'opp_epa_allowed',
            'is_dome', 'wind_mph', 'temp_f', 'wind_impact',
            'targets_trend', 'targets_momentum', 'receptions_trend', 'receptions_momentum',
            'spread', 'team_total', 'game_total', 'is_favorite', 'is_underdog',
            'implied_game_script', 'pace_expectation', 'pass_rate_over_expected',
            'player_injury_status', 'team_offensive_injuries', 'opponent_defensive_injuries',
            'red_zone_targets', 'red_zone_target_share', 'red_zone_td_rate',
            'two_min_usage', 'hurry_up_usage', 'clutch_target_share',
            'historical_line_avg', 'line_movement'
        ]}

    def _get_default_qb_features(self) -> Dict:
        """Default QB features"""
        return {f: 0 for f in [
            'avg_completions_L1', 'avg_completions_weighted', 'std_completions',
            'avg_passing_yards_L1', 'avg_passing_yards_weighted', 'std_passing_yards',
            'passing_yards_cv', 'avg_sacks_taken', 'sack_rate',
            'is_dome', 'wind_mph', 'spread', 'team_total'
        ]}

    def _get_default_rushing_features(self) -> Dict:
        """Default rushing features"""
        return {f: 0 for f in [
            'avg_carries_L1', 'avg_carries_weighted', 'std_carries',
            'avg_rushing_yards_L1', 'avg_rushing_yards_weighted', 'std_rushing_yards',
            'rushing_yards_cv', 'expected_game_script',
            'is_favorite', 'is_underdog'
        ]}


if __name__ == "__main__":
    # Test feature engineering
    # FIXED: Import correct classes
    from load_2025_data import CurrentSeasonLoader

    print("\n" + "="*60)
    print("TESTING FEATURE ENGINEERING")
    print("="*60)

    # FIXED: Use CurrentSeasonLoader
    loader = CurrentSeasonLoader()
    data = loader.load_2025_data(through_week=None)  # Auto-detect

    engineer = Phase1FeatureEngineer(data)

    # Get a sample player
    sample_player = data['player_weeks'][
        (data['player_weeks']['position'] == 'WR') &
        (data['player_weeks']['season'] == Config.CURRENT_SEASON)
    ].iloc[0]

    player_id = sample_player['player_id']
    team = sample_player['team']

    print(f"\nGenerating features for: {sample_player['player_name']}")
    print(f"Position: {sample_player['position']}, Team: {team}")

    features = engineer.create_receiving_features(
        player_id=player_id,
        team=team,
        week=8,
        season=Config.CURRENT_SEASON,
        opponent='BUF'
    )

    print(f"\n✅ Generated {len(features)} features")
    print("\nSample features:")
    for i, (key, val) in enumerate(features.items()):
        if i < 20:
            print(f"  {key}: {val:.3f}" if isinstance(val, float) else f"  {key}: {val}")
