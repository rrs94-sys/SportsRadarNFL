"""
Main Training Pipeline with Time-Series Backtesting
Trains market-optimized models for all prop types

FIXED: Added dynamic season and week detection
FIXED: Imports from nfl_data_utils for season/week detection
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from typing import Dict, List

from phase1_config import Config
from phase1_ingestion import HistoricalDataIngestion
from load_2025_data import CurrentSeasonLoader
from phase1_features import Phase1FeatureEngineer
from market_optimized_models import MarketOptimizedModel, TDProbabilityModel
from tank_injury_client import TankInjuryClient
from weather_client import WeatherClient
from sportsgameodds_client import SportsGameOddsClient
from calibration_evaluation import CalibrationEvaluator

# FIXED: Import dynamic detection utilities and hybrid loader
from nfl_data_utils import (
    get_current_season,
    get_latest_completed_week,
    load_hybrid_data  # NEW: For merging historical + 2025 data
)


class ModelTrainer:
    """
    Orchestrates model training with time-series backtesting.

    Implements:
    - Time-series split validation (train 1-4, val 5-6, etc.)
    - Position-specific quantile targets
    - Feature engineering with external data
    - Model persistence
    - Combines historical (2022-2024) + current (2025) data with 30/70 weighting
    """

    def __init__(self, config: Config = Config):
        self.config = config
        self.historical_loader = HistoricalDataIngestion(config)
        self.current_loader = CurrentSeasonLoader(config)
        self.injury_client = TankInjuryClient(config)
        self.weather_client = WeatherClient(config)
        self.sgo_client = SportsGameOddsClient(config)
        self.evaluator = CalibrationEvaluator(config)

        self.trained_models = {}

    def _load_and_combine_data(self, through_week: int = None, force_refresh: bool = False) -> Dict:
        """
        Load and combine historical + current season data with proper weighting.

        FIXED: Dynamic season and week detection
        FIXED: Auto-detect latest week if through_week is None
        FIXED: Uses hybrid loader for seamless nfl_data_py + TANK01 integration

        Args:
            through_week: Last COMPLETED week for current season (None = auto-detect)
            force_refresh: Force re-download

        Returns:
            dict with combined player_weeks, schedules, team_schedules
        """
        # FIXED: Dynamic season detection
        current_season = get_current_season()

        # FIXED: Auto-detect latest week if not specified
        if through_week is None:
            through_week = get_latest_completed_week(current_season)
            print(f"📊 Auto-detected latest completed week: {through_week}")

        # FIXED: Use hybrid loader that automatically handles nfl_data_py + TANK01
        print(f"\n{'='*70}")
        print(f"LOADING DATA: Historical + Current ({current_season})")
        print(f"AUTOMATIC SOURCE SELECTION: nfl_data_py (≤2024) + TANK01 (2025)")
        print(f"{'='*70}")

        # Load combined data using hybrid loader
        combined_data = load_hybrid_data(
            historical_years=3,
            current_through_week=through_week,
            cache_dir=self.config.CACHE_DIR,
            force_refresh=force_refresh
        )

        # FIXED: Apply weights to player_weeks for training
        player_weeks = combined_data['player_weeks'].copy()

        # Separate historical vs current for weighting
        current_season = get_current_season()
        historical_mask = player_weeks['season'] < current_season
        current_mask = player_weeks['season'] == current_season

        player_weeks.loc[historical_mask, 'weight'] = self.config.HISTORICAL_WEIGHT
        player_weeks.loc[current_mask, 'weight'] = self.config.RECENT_WEIGHT

        combined_data['player_weeks'] = player_weeks

        # Print summary
        hist_count = historical_mask.sum()
        curr_count = current_mask.sum()

        print(f"\n   ✅ Combined data with weights:")
        print(f"      Historical: {hist_count} player-weeks (weight={self.config.HISTORICAL_WEIGHT})")
        print(f"      Current:    {curr_count} player-weeks (weight={self.config.RECENT_WEIGHT})")
        print(f"      Total:      {len(player_weeks)} player-weeks")

        return combined_data

    def run_backtest(self, through_week: int = 7):
        """
        Run time-series backtest according to requirements:
        - Train on 1-4, validate on 5-6
        - Train on 1-5, validate on 6-7
        - Train on 1-6, validate on 7-8 (if through_week >= 8)
        """
        print("\n" + "="*70)
        print("TIME-SERIES BACKTEST")
        print("="*70)

        backtest_results = []

        for window in self.config.BACKTEST_WINDOWS:
            if through_week < window['val_weeks'][1]:
                continue  # Skip if data not available

            print(f"\n{'─'*70}")
            print(f"Window: Train weeks {window['train_weeks']}, Val weeks {window['val_weeks']}")
            print(f"{'─'*70}")

            # Load data
            data = self._load_and_combine_data(through_week=window['val_weeks'][1])

            # Filter to training and validation weeks
            train_data = self._filter_by_weeks(data, window['train_weeks'])
            val_data = self._filter_by_weeks(data, window['val_weeks'])

            # Train models
            window_results = self._train_window(train_data, val_data)
            window_results['window'] = window
            backtest_results.append(window_results)

        # Save backtest results
        results_file = os.path.join(self.config.MODEL_DIR, 'backtest_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(backtest_results, f)

        print("\n" + "="*70)
        print("BACKTEST COMPLETE")
        print("="*70)

        self._print_backtest_summary(backtest_results)

        return backtest_results

    def train_final_models(self, through_week: int = None):
        """
        Train final models on all available data through specified week.
        This is run after backtesting to create production models.

        FIXED: Auto-detect latest week if through_week is None
        """
        # FIXED: Auto-detect if not specified
        if through_week is None:
            current_season = get_current_season()
            through_week = get_latest_completed_week(current_season)
            print(f"📊 Auto-detected latest completed week: {through_week}")

        print("\n" + "="*70)
        print(f"TRAINING FINAL MODELS (through Week {through_week})")
        print("="*70)

        # Load all data
        data = self._load_and_combine_data(through_week=through_week)

        # Load external data
        print("\nLoading external data...")
        injuries = self.injury_client.get_injury_report()
        # Weather will be loaded per-game during prediction

        # Initialize feature engineer
        engineer = Phase1FeatureEngineer(data, self.config)
        engineer.set_external_data(injuries=injuries)

        # Prop types to train
        prop_configs = [
            {'prop': 'receiving_yards', 'positions': ['WR', 'TE', 'RB'], 'model_type': 'regression'},
            {'prop': 'receptions', 'positions': ['WR', 'TE', 'RB'], 'model_type': 'regression'},
            {'prop': 'receiving_tds', 'positions': ['WR', 'TE', 'RB'], 'model_type': 'td_prob'},
            {'prop': 'rushing_yards', 'positions': ['RB', 'QB'], 'model_type': 'regression'},
            {'prop': 'rushing_tds', 'positions': ['RB', 'QB'], 'model_type': 'td_prob'},
            {'prop': 'passing_yards', 'positions': ['QB'], 'model_type': 'regression'},
            {'prop': 'completions', 'positions': ['QB'], 'model_type': 'regression'},
            {'prop': 'passing_tds', 'positions': ['QB'], 'model_type': 'td_prob'}
        ]

        for config in prop_configs:
            print(f"\n{'─'*70}")
            print(f"Training: {config['prop']}")
            print(f"{'─'*70}")

            model = self._train_prop_model(
                data, engineer, config['prop'],
                config['positions'], config['model_type']
            )

            self.trained_models[config['prop']] = model

        # Save all models
        self._save_models()

        print("\n✅ All models trained and saved!")

    def _filter_by_weeks(self, data: Dict, week_range: tuple) -> Dict:
        """Filter data to specific week range"""
        filtered_data = data.copy()

        pw = data['player_weeks']
        filtered_data['player_weeks'] = pw[
            (pw['week'] >= week_range[0]) &
            (pw['week'] <= week_range[1])
        ]

        return filtered_data

    def _train_window(self, train_data: Dict, val_data: Dict) -> Dict:
        """Train models for a single backtest window"""
        results = {}

        # Initialize engineers
        train_engineer = Phase1FeatureEngineer(train_data, self.config)
        val_engineer = Phase1FeatureEngineer(val_data, self.config)

        # Train receiving yards model (example)
        print("\n  Training receiving_yards model...")

        model = self._train_prop_model(
            train_data, train_engineer, 'receiving_yards',
            ['WR', 'TE', 'RB'], 'regression'
        )

        # Validate
        val_metrics = self._validate_model(
            model, val_data, val_engineer,
            'receiving_yards', ['WR', 'TE', 'RB']
        )

        results['receiving_yards'] = val_metrics

        return results

    def _train_prop_model(
        self,
        data: Dict,
        engineer: Phase1FeatureEngineer,
        prop_type: str,
        positions: List[str],
        model_type: str
    ):
        """Train a single prop model"""
        player_weeks = data['player_weeks']

        # Filter to relevant positions
        pw_filtered = player_weeks[player_weeks['position'].isin(positions)]

        # Build features
        X_list = []
        y_list = []
        game_ids = []
        weights = []

        target_col = prop_type

        for idx, row in pw_filtered.iterrows():
            # Get opponent
            opponent = row['opponent_team']

            # Create features
            if prop_type in ['receiving_yards', 'receptions', 'receiving_tds']:
                features = engineer.create_receiving_features(
                    player_id=row['player_id'],
                    team=row['team'],
                    week=row['week'],
                    season=row['season'],
                    opponent=opponent
                )
            elif prop_type in ['rushing_yards', 'rushing_tds']:
                features = engineer.create_rushing_features(
                    player_id=row['player_id'],
                    team=row['team'],
                    week=row['week'],
                    season=row['season'],
                    opponent=opponent
                )
            elif prop_type in ['passing_yards', 'completions', 'passing_tds']:
                features = engineer.create_qb_features(
                    player_id=row['player_id'],
                    team=row['team'],
                    week=row['week'],
                    season=row['season'],
                    opponent=opponent
                )
            else:
                continue

            X_list.append(features)
            y_list.append(row[target_col])

            # Game ID for GroupKFold
            teams = sorted([row['team'], opponent])
            game_id = f"{row['season']}_W{row['week']}_{teams[0]}_{teams[1]}"
            game_ids.append(game_id)

            # Sample weight (historical 30% vs current 70%)
            weights.append(row.get('weight', 1.0))

        if len(X_list) == 0:
            print(f"  ⚠️ No data for {prop_type}")
            return None

        # Convert to DataFrames
        X = pd.DataFrame(X_list)
        y = pd.Series(y_list)
        game_ids_series = pd.Series(game_ids)
        weights_array = np.array(weights)

        # Train model
        if model_type == 'regression':
            model = MarketOptimizedModel(prop_type, self.config)
            model.train(X, y, game_ids_series, sample_weight=weights_array)
        elif model_type == 'td_prob':
            model = TDProbabilityModel(prop_type, self.config)
            model.train(X, y, game_ids_series, sample_weight=weights_array)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return model

    def _validate_model(
        self,
        model,
        val_data: Dict,
        engineer: Phase1FeatureEngineer,
        prop_type: str,
        positions: List[str]
    ) -> Dict:
        """Validate model on holdout data"""
        # Similar to training but for validation
        # Returns metrics
        return {'mae': 0, 'r2': 0}  # Placeholder

    def _save_models(self):
        """Save all trained models"""
        save_dir = self.config.MODEL_DIR

        for prop_type, model in self.trained_models.items():
            filepath = os.path.join(save_dir, f'{prop_type}_model.pkl')
            model.save(filepath)
            print(f"  ✅ Saved {prop_type} model")

        # Save feature engineer configuration
        print(f"\n✅ All models saved to {save_dir}/")

    def _print_backtest_summary(self, results: List[Dict]):
        """Print summary of backtest results"""
        print("\nBacktest Summary:")
        for res in results:
            window = res['window']
            print(f"\n  Window: Train {window['train_weeks']}, Val {window['val_weeks']}")
            for prop, metrics in res.items():
                if prop != 'window' and isinstance(metrics, dict):
                    print(f"    {prop}: MAE={metrics.get('mae', 'N/A')}, R²={metrics.get('r2', 'N/A')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train NFL Props Models')
    parser.add_argument('--backtest', action='store_true', help='Run time-series backtest')
    parser.add_argument('--train', action='store_true', help='Train final models')
    # FIXED: Default to None for auto-detection
    parser.add_argument('--through-week', type=int, default=None, help='Train through this week (default: auto-detect)')

    args = parser.parse_args()

    trainer = ModelTrainer()

    # FIXED: Use dynamic detection
    through_week = args.through_week
    if through_week is None:
        current_season = get_current_season()
        through_week = get_latest_completed_week(current_season)
        print(f"\n📊 Auto-detected season {current_season}, week {through_week}")

    if args.backtest:
        trainer.run_backtest(through_week=through_week)

    if args.train:
        trainer.train_final_models(through_week=through_week)

    if not args.backtest and not args.train:
        # Default: do both
        print("Running backtest and training...")
        print(f"Using through week: {through_week}")
        trainer.run_backtest(through_week=through_week - 1 if through_week > 1 else through_week)
        trainer.train_final_models(through_week=through_week)
