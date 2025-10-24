"""
Configuration for NFL Player Props Prediction System
Version 2.0 - Market-Optimized with Enhancements
"""

import os
from typing import Dict, List


class Config:
    # ========================================================================
    # API CONFIGURATION
    # ========================================================================

    # SportsGameOdds API (500 calls/month limit, 100 used)
    SPORTSGAMEODDS_API_KEY = "0b4d590cc09e7cfe05da2247b338698e"
    SPORTSGAMEODDS_BASE_URL = "https://api.sportsgameodds.com/v2"
    SPORTSGAMEODDS_MONTHLY_LIMIT = 500
    SPORTSGAMEODDS_CALLS_USED = 100  # Track usage

    # Tank (Injury API via RapidAPI)
    TANK_API_KEY = "9c58e7cda2msh95af7473755205ep12f820jsn13cda64975cf"
    TANK_BASE_URL = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"

    # Open-Meteo Weather API (no key needed)
    OPENMETEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"

    # ========================================================================
    # BANKROLL & BETTING
    # ========================================================================
    INITIAL_BANKROLL = 10000
    KELLY_FRACTION = 0.25  # Quarter Kelly for safety
    MIN_EDGE = 0.05  # 5% minimum edge to bet
    MIN_BET = 10
    MAX_BET = 500
    MAX_CORRELATED_EXPOSURE = 0.15  # Max 15% of bankroll on correlated bets

    # Position-specific edge thresholds (based on historical ROI)
    EDGE_THRESHOLDS = {
        'receiving_yards': 1.0,
        'receptions': 1.0,
        'receiving_tds': 1.5,
        'rushing_yards': 2.0,  # Higher threshold due to lower R²
        'rushing_tds': 1.5,
        'passing_yards': 1.2,
        'completions': 1.0,
        'passing_tds': 1.5
    }

    # ========================================================================
    # DATA SETTINGS
    # ========================================================================
    HISTORICAL_SEASONS = [2022, 2023, 2024]  # Last 3 full seasons
    CURRENT_SEASON = 2025  # Current 2025 season (updates weekly)
    CACHE_DIR = "data_cache"
    MODEL_DIR = "saved_models_v2"
    LINES_CACHE_DIR = "lines_cache"  # Historical lines for edge tracking

    # Data weighting
    HISTORICAL_WEIGHT = 0.30  # 30% historical (2022-2024)
    RECENT_WEIGHT = 0.70  # 70% recent (2025)

    # Recency weights for feature engineering
    RECENCY_WEIGHTS = {
        'L1': 0.30,       # Last 1 game
        'L2_L3': 0.40,    # Last 2-3 games
        'L4_PLUS': 0.30   # Last 4+ games
    }

    MIN_GAMES_PLAYED = 3
    MIN_SAMPLES_FOR_TRAINING = 50
    MIN_SNAPS_PER_GAME = 10

    # ========================================================================
    # MODEL ARCHITECTURE
    # ========================================================================

    # Position-specific quantile targets (optimized per market)
    QUANTILE_TARGETS = {
        'receiving_yards': 0.60,
        'receptions': 0.58,  # Tighter distribution
        'receiving_tds': 0.70,  # Higher for rare events
        'rushing_yards': 0.65,  # Higher due to volatility
        'rushing_tds': 0.70,
        'passing_yards': 0.58,
        'completions': 0.57,
        'passing_tds': 0.68
    }

    # XGBoost parameters
    XGB_PARAMS = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'random_state': 42
    }

    # LightGBM parameters
    LGBM_PARAMS = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'verbose': -1
    }

    # Ensemble weights
    ENSEMBLE_WEIGHTS = {
        'xgb_quantile': 0.60,
        'lgb_quantile': 0.40
    }

    # ========================================================================
    # CROSS-VALIDATION & BACKTESTING
    # ========================================================================
    N_SPLITS = 5  # GroupKFold splits
    ISOTONIC_CALIBRATION_ENABLED = True

    # Time-series backtest windows
    BACKTEST_WINDOWS = [
        {'train_weeks': (1, 4), 'val_weeks': (5, 6)},
        {'train_weeks': (1, 5), 'val_weeks': (6, 7)},
        {'train_weeks': (1, 6), 'val_weeks': (7, 8)}
    ]

    # ========================================================================
    # MONTE CARLO SIMULATION
    # ========================================================================
    MONTE_CARLO_ITERATIONS = 10000
    MC_CONFIDENCE_LEVEL = 0.95
    MC_MIN_HIT_RATE = 0.54  # Minimum hit rate to recommend bet (beats -110 juice)

    # ========================================================================
    # FEATURE ENGINEERING
    # ========================================================================

    # Volatility settings
    VOLATILITY_WINDOW = 4  # Games to calculate CV
    HIGH_VOLATILITY_THRESHOLD = 0.5  # CV threshold for "high volatility"
    VOLATILITY_EDGE_MULTIPLIER = 1.5  # Require 1.5x edge for volatile players

    # Usage features
    USAGE_FEATURES = [
        'route_participation_pct',
        'pass_snap_pct',
        'rush_share',
        'targets_per_route',
        'air_yards_share',
        'red_zone_share',
        'two_min_usage',
        'hurry_up_usage',
        'pass_block_rate'
    ]

    # ========================================================================
    # STADIUM & ENVIRONMENTAL DATA
    # ========================================================================

    DOME_STADIUMS = ['ARI', 'ATL', 'DET', 'HOU', 'IND', 'LV', 'MIN', 'NO']

    STADIUM_COORDS = {
        'ARI': (33.5276, -112.2626),
        'ATL': (33.7554, -84.4009),
        'BAL': (39.2780, -76.6227),
        'BUF': (42.7738, -78.7870),
        'CAR': (35.2258, -80.8529),
        'CHI': (41.8623, -87.6167),
        'CIN': (39.0954, -84.5160),
        'CLE': (41.5061, -81.6995),
        'DAL': (32.7473, -97.0945),
        'DEN': (39.7439, -105.0201),
        'DET': (42.3400, -83.0456),
        'GB': (44.5013, -88.0622),
        'HOU': (29.6847, -95.4107),
        'IND': (39.7601, -86.1639),
        'JAX': (30.3240, -81.6373),
        'KC': (39.0489, -94.4839),
        'LV': (36.0909, -115.1833),
        'LAC': (33.9536, -118.3390),
        'LAR': (33.9536, -118.3390),
        'MIA': (25.9580, -80.2389),
        'MIN': (44.9738, -93.2577),
        'NE': (42.0909, -71.2643),
        'NO': (29.9511, -90.0812),
        'NYG': (40.8128, -74.0742),
        'NYJ': (40.8128, -74.0742),
        'PHI': (39.9008, -75.1675),
        'PIT': (40.4468, -80.0158),
        'SF': (37.4032, -121.9698),
        'SEA': (47.5952, -122.3316),
        'TB': (27.9759, -82.5033),
        'TEN': (36.1665, -86.7713),
        'WAS': (38.9076, -76.8645)
    }

    TEAM_TIMEZONES = {
        'LAC': -8, 'LAR': -8, 'SF': -8, 'SEA': -8,  # Pacific
        'ARI': -7, 'DEN': -7,  # Mountain
        'DAL': -6, 'HOU': -6, 'KC': -6, 'MIN': -6, 'NO': -6, 'CHI': -6,
        'GB': -6, 'TEN': -6, 'IND': -6, 'DET': -6,  # Central
        'ATL': -5, 'BAL': -5, 'BUF': -5, 'CAR': -5, 'CIN': -5, 'CLE': -5,
        'JAX': -5, 'MIA': -5, 'NE': -5, 'NYG': -5, 'NYJ': -5, 'PHI': -5,
        'PIT': -5, 'TB': -5, 'WAS': -5  # Eastern
    }

    # Weather impact thresholds
    WIND_THRESHOLD_MPH = 15  # Strong wind affects passing
    TEMP_COLD_THRESHOLD_F = 32  # Freezing affects ball handling
    TEMP_HOT_THRESHOLD_F = 85  # Heat affects stamina

    # ========================================================================
    # LOSS FUNCTIONS
    # ========================================================================
    NEGBINOM_ALPHA_THRESHOLD = 0.2  # Use NegBin if dispersion > 0.2
    LOGNORMAL_DEFAULT_CV = 0.5
    HUBER_DELTA = 1.0  # Huber loss delta parameter

    # TD model settings
    TD_PROBABILITY_THRESHOLD = 0.5
    TD_CALIBRATION_METHOD = 'isotonic'  # 'isotonic' or 'platt'

    # ========================================================================
    # EVALUATION METRICS
    # ========================================================================

    # Target metrics for validation
    TARGET_METRICS = {
        'receptions': {'mae': 1.3, 'r2': 0.10},
        'receiving_yards': {'mae': 18.0, 'r2': 0.10},
        'completions': {'mae': 2.5, 'r2': 0.10},
        'passing_yards': {'mae': 30.0, 'r2': 0.10},
        'rushing_yards': {'mae': 15.0, 'r2': 0.04},  # Lower expectation
        'receiving_tds': {'mae': 0.35, 'r2': 0.05},
        'passing_tds': {'mae': 0.40, 'r2': 0.05},
        'rushing_tds': {'mae': 0.35, 'r2': 0.05},
    }

    # PnL-based evaluation
    JUICE = 1.10  # Standard -110 odds
    MIN_RELATIVE_IMPROVEMENT = 0.10  # 10% over baseline

    # ========================================================================
    # PREFERRED SPORTSBOOKS
    # ========================================================================
    PREFERRED_SPORTSBOOKS = ['hardrock', 'fanduel', 'draftkings', 'betmgm', 'caesars']

    # ========================================================================
    # UPDATE SCHEDULE
    # ========================================================================
    WEEKLY_MICRO_UPDATE = True  # Re-fit isotonic calibration only
    MACRO_RETRAIN_FREQUENCY = 4  # Full retrain every 4 weeks

    # ========================================================================
    # CORRELATION MATRIX (for exposure management)
    # ========================================================================
    # Positions that are highly correlated
    CORRELATED_POSITIONS = {
        'QB_WR_same_team': 0.7,
        'QB_TE_same_team': 0.6,
        'WR_WR_same_team': 0.3,
        'RB_DEF_opp_team': -0.5  # Negative correlation
    }

    # ========================================================================
    # LOGGING
    # ========================================================================
    LOG_LEVEL = "INFO"
    LOG_FILE = "nfl_model.log"

    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        for directory in [cls.CACHE_DIR, cls.MODEL_DIR, cls.LINES_CACHE_DIR]:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def get_quantile_target(cls, prop_type: str) -> float:
        """Get position-specific quantile target"""
        return cls.QUANTILE_TARGETS.get(prop_type, 0.60)

    @classmethod
    def get_edge_threshold(cls, prop_type: str) -> float:
        """Get position-specific edge threshold"""
        return cls.EDGE_THRESHOLDS.get(prop_type, 1.0)

    @classmethod
    def track_sgo_call(cls):
        """Track SGO API usage"""
        cls.SPORTSGAMEODDS_CALLS_USED += 1
        remaining = cls.SPORTSGAMEODDS_MONTHLY_LIMIT - cls.SPORTSGAMEODDS_CALLS_USED
        if remaining < 50:
            print(f"⚠️ WARNING: Only {remaining} SGO API calls remaining this month!")
        return remaining

    @classmethod
    def can_make_sgo_call(cls) -> bool:
        """Check if we can make an SGO API call"""
        return cls.SPORTSGAMEODDS_CALLS_USED < cls.SPORTSGAMEODDS_MONTHLY_LIMIT


# Initialize directories on import
Config.ensure_directories()
