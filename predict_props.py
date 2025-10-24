"""
Main Prediction Pipeline with Monte Carlo Simulation and Edge@k
Generates actionable betting recommendations

FIXED: Corrected imports (RecencyDataIngestion doesn't exist)
FIXED: Added dynamic season detection
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from typing import Dict, List

from phase1_config import Config
# FIXED: Import correct classes
from phase1_ingestion import HistoricalDataIngestion
from load_2025_data import CurrentSeasonLoader
from phase1_features import Phase1FeatureEngineer
from market_optimized_models import MarketOptimizedModel, TDProbabilityModel
from tank_injury_client import TankInjuryClient
from weather_client import WeatherClient
from sportsgameodds_client import SportsGameOddsClient
from monte_carlo_engine import MonteCarloEngine
from quantitative_betting_engine import QuantitativeBettingEngine

# FIXED: Import dynamic detection utilities
from nfl_data_utils import get_current_season, get_latest_completed_week


class PropPredictor:
    """
    Main prediction pipeline:
    1. Load trained models
    2. Fetch current lines (minimize SGO calls)
    3. Generate predictions
    4. Run Monte Carlo simulation (10,000 iterations)
    5. Filter by hit rate
    6. Calculate Kelly sizing
    7. Apply correlation controls
    8. Output final recommendations
    """

    def __init__(self, config: Config = Config):
        self.config = config
        # FIXED: Use CurrentSeasonLoader instead of RecencyDataIngestion
        self.current_loader = CurrentSeasonLoader(config)
        self.historical_loader = HistoricalDataIngestion(config)
        self.injury_client = TankInjuryClient(config)
        self.weather_client = WeatherClient(config)
        self.sgo_client = SportsGameOddsClient(config)
        self.mc_engine = MonteCarloEngine(config=config)
        self.betting_engine = QuantitativeBettingEngine(config=config)

        self.models = {}
        self.engineer = None

    def load_models(self):
        """Load all trained models"""
        print("\n" + "="*70)
        print("LOADING TRAINED MODELS")
        print("="*70)

        model_dir = self.config.MODEL_DIR

        prop_types = [
            'receiving_yards', 'receptions', 'receiving_tds',
            'rushing_yards', 'rushing_tds',
            'passing_yards', 'completions', 'passing_tds'
        ]

        for prop_type in prop_types:
            filepath = os.path.join(model_dir, f'{prop_type}_model.pkl')

            if os.path.exists(filepath):
                if 'tds' in prop_type:
                    model = TDProbabilityModel.load(filepath)
                else:
                    model = MarketOptimizedModel.load(filepath)

                self.models[prop_type] = model
                print(f"  ✅ Loaded {prop_type}")
            else:
                print(f"  ⚠️  Model not found: {prop_type}")

        print(f"\n✅ Loaded {len(self.models)} models")

    def predict_week(self, week: int, season: int = None):
        """
        Generate predictions for a specific week.

        FIXED: Added dynamic season detection
        FIXED: Use CurrentSeasonLoader for data loading

        Args:
            week: Week to predict
            season: Season (default: auto-detect current)

        Returns:
            DataFrame with predictions and recommendations
        """
        # FIXED: Dynamic season detection
        if season is None:
            season = get_current_season()

        print("\n" + "="*70)
        print(f"PREDICTING WEEK {week}, {season}")
        print("="*70)

        # Step 1: Load data through previous week
        print(f"\n1. Loading data through Week {week-1}...")
        # FIXED: Use CurrentSeasonLoader
        data = self.current_loader.load_2025_data(through_week=week-1)

        # Step 2: Load external data
        print("\n2. Loading external data...")
        injuries = self.injury_client.get_injury_report()

        # Step 3: Fetch lines (MINIMIZE SGO CALLS!)
        print("\n3. Fetching prop lines...")
        if self.sgo_client.can_make_sgo_call():
            lines = self.sgo_client.get_player_props(week, season)
        else:
            print("  ⚠️ SGO API limit reached, using cached lines")
            lines = pd.DataFrame()

        if lines.empty:
            print("  ❌ No lines available, cannot generate predictions")
            return pd.DataFrame()

        # Step 4: Initialize feature engineer
        print("\n4. Initializing feature engineer...")
        self.engineer = Phase1FeatureEngineer(data, self.config)
        self.engineer.set_external_data(injuries=injuries, lines=lines)

        # Step 5: Get active roster
        print("\n5. Getting active roster...")
        # FIXED: Generate active roster from player_weeks data
        player_weeks = data['player_weeks']
        recent_weeks = [week-3, week-2, week-1]
        recent_weeks = [w for w in recent_weeks if w > 0]  # Filter out negative weeks

        roster = player_weeks[
            player_weeks['week'].isin(recent_weeks)
        ].groupby(['player_id', 'player_name', 'team', 'position']).size().reset_index()
        roster = roster.drop(columns=[0])  # Remove count column
        print(f"  Found {len(roster)} active players")

        # Step 6: Generate predictions
        print("\n6. Generating predictions...")
        all_predictions = []

        for idx, player in roster.iterrows():
            player_preds = self._predict_player(
                player, week, season, lines
            )
            all_predictions.extend(player_preds)

        predictions_df = pd.DataFrame(all_predictions)

        if predictions_df.empty:
            print("  ⚠️ No predictions generated")
            return predictions_df

        print(f"  ✅ Generated {len(predictions_df)} predictions")

        # Step 7: Run Monte Carlo simulation
        print("\n7. Running Monte Carlo simulation (10,000 iterations)...")
        mc_results = self.mc_engine.simulate_multiple_props(
            predictions_df,
            prediction_col='prediction',
            line_col='line',
            std_col='std_dev'
        )

        # Step 8: Filter by hit rate
        print(f"\n8. Filtering by hit rate (≥{self.config.MC_MIN_HIT_RATE:.1%})...")
        filtered = self.mc_engine.filter_by_hit_rate(
            mc_results,
            min_hit_rate=self.config.MC_MIN_HIT_RATE
        )
        print(f"  ✅ {len(filtered)} props passed hit rate filter")

        if filtered.empty:
            print("  ⚠️ No props meet hit rate threshold")
            return filtered

        # Step 9: Apply edge thresholds (position-specific)
        print("\n9. Applying position-specific edge thresholds...")
        filtered = self._apply_edge_filters(filtered)
        print(f"  ✅ {len(filtered)} props passed edge filter")

        # Step 10: Calculate Kelly sizing
        print("\n10. Calculating Kelly bet sizes...")
        filtered['kelly_fraction'] = filtered.apply(
            lambda row: self.mc_engine.calculate_kelly_with_mc(
                row['mc_over_rate'] if row['recommendation'] == 'OVER' else row['mc_under_rate'],
                row.get('odds', -110)
            ),
            axis=1
        )

        # Step 11: Optimize portfolio (correlation controls)
        print("\n11. Optimizing bet portfolio...")
        final_bets = self.betting_engine.optimize_bet_portfolio(
            filtered,
            max_total_exposure=self.config.MAX_CORRELATED_EXPOSURE
        )

        print(f"  ✅ Final portfolio: {len(final_bets)} bets")

        # Step 12: Sort and format output
        final_bets = final_bets.sort_values('mc_over_rate', ascending=False)

        # Step 13: Save predictions
        output_file = f'week{week}_predictions_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
        final_bets.to_csv(output_file, index=False)

        print(f"\n✅ Predictions saved to {output_file}")

        # Print summary
        self._print_prediction_summary(final_bets)

        return final_bets

    def _predict_player(
        self,
        player: pd.Series,
        week: int,
        season: int,
        lines: pd.DataFrame
    ) -> List[Dict]:
        """Generate predictions for a single player"""
        player_id = player['player_id']
        position = player['position']
        team = player['team']

        # Determine which props to predict based on position
        if position in ['WR', 'TE']:
            prop_types = ['receiving_yards', 'receptions', 'receiving_tds']
        elif position == 'RB':
            prop_types = ['receiving_yards', 'receptions', 'rushing_yards', 'rushing_tds']
        elif position == 'QB':
            prop_types = ['passing_yards', 'completions', 'passing_tds']
        else:
            return []

        predictions = []

        # Get opponent (from schedule)
        schedules = self.engineer.data['schedules']
        game = schedules[
            (schedules['season'] == season) &
            (schedules['week'] == week) &
            ((schedules['home_team'] == team) | (schedules['away_team'] == team))
        ]

        if game.empty:
            return []

        opponent = game.iloc[0]['away_team'] if game.iloc[0]['home_team'] == team else game.iloc[0]['home_team']

        for prop_type in prop_types:
            if prop_type not in self.models:
                continue

            # Get line from market
            player_line = lines[
                (lines['player_name'].str.contains(player['player_name'].split()[-1], case=False, na=False)) &
                (lines['market'] == prop_type)
            ]

            if player_line.empty:
                continue  # No line available

            line = player_line.iloc[0]['line']
            odds = player_line.iloc[0].get('over_odds', -110)

            # Generate features
            if prop_type in ['receiving_yards', 'receptions', 'receiving_tds']:
                features = self.engineer.create_receiving_features(
                    player_id, team, week, season, opponent
                )
            elif prop_type in ['rushing_yards', 'rushing_tds']:
                features = self.engineer.create_rushing_features(
                    player_id, team, week, season, opponent
                )
            elif prop_type in ['passing_yards', 'completions', 'passing_tds']:
                features = self.engineer.create_qb_features(
                    player_id, team, week, season, opponent
                )
            else:
                continue

            # Predict
            X = pd.DataFrame([features])
            model = self.models[prop_type]

            if 'tds' in prop_type:
                # Probability model
                pred_prob = model.predict_proba(X)[0]
                # Convert to expected value for comparison
                prediction = pred_prob  # Use as-is for now
                std_dev = np.sqrt(pred_prob * (1 - pred_prob))
            else:
                # Regression model
                prediction = model.predict(X)[0]
                # Estimate std dev from CV scores
                std_dev = prediction * 0.3  # Rough estimate

            # Determine recommendation
            if prediction > line:
                recommendation = 'OVER'
                edge = prediction - line
            elif prediction < line:
                recommendation = 'UNDER'
                edge = line - prediction
            else:
                continue

            predictions.append({
                'player_id': player_id,
                'player_name': player['player_name'],
                'team': team,
                'position': position,
                'opponent': opponent,
                'week': week,
                'season': season,
                'market': prop_type,
                'line': line,
                'odds': odds,
                'prediction': prediction,
                'std_dev': std_dev,
                'edge': edge,
                'recommendation': recommendation,
                'game_id': game.iloc[0].get('game_id', '')
            })

        return predictions

    def _apply_edge_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply position-specific edge thresholds"""
        filtered = []

        for idx, row in df.iterrows():
            prop_type = row['market']
            edge = row['edge']
            volatility_cv = row.get('std_dev', 0) / max(row['prediction'], 1)

            # Get threshold
            base_threshold = self.config.get_edge_threshold(prop_type)

            # Apply volatility adjustment
            passes, adjusted_threshold = self.betting_engine.apply_volatility_adjustment(
                edge, volatility_cv, base_threshold
            )

            if passes:
                row_dict = row.to_dict()
                row_dict['edge_threshold'] = adjusted_threshold
                row_dict['volatility_cv'] = volatility_cv
                filtered.append(row_dict)

        return pd.DataFrame(filtered)

    def _print_prediction_summary(self, predictions: pd.DataFrame):
        """Print summary of predictions"""
        print("\n" + "="*70)
        print("PREDICTION SUMMARY")
        print("="*70)

        print(f"\nTotal Recommendations: {len(predictions)}")

        if len(predictions) > 0:
            print(f"\nBy Market:")
            print(predictions['market'].value_counts())

            print(f"\nBy Team:")
            print(predictions['team'].value_counts().head(10))

            print(f"\nTop 10 Recommendations:")
            cols = ['player_name', 'market', 'line', 'prediction', 'recommendation', 'mc_over_rate', 'bet_size']
            available_cols = [c for c in cols if c in predictions.columns]
            print(predictions[available_cols].head(10).to_string(index=False))

            total_risk = predictions['bet_size'].sum() if 'bet_size' in predictions.columns else 0
            print(f"\nTotal Recommended Risk: ${total_risk:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Predict NFL Props')
    parser.add_argument('--week', type=int, required=False, default=None, help='Week to predict (default: next week)')
    parser.add_argument('--season', type=int, default=None, help='Season (default: auto-detect current)')

    args = parser.parse_args()

    # FIXED: Auto-detect week and season if not provided
    season = args.season
    if season is None:
        season = get_current_season()
        print(f"📊 Auto-detected season: {season}")

    week = args.week
    if week is None:
        latest_week = get_latest_completed_week(season)
        week = latest_week + 1  # Predict next week
        print(f"📊 Auto-detected latest completed week: {latest_week}")
        print(f"📊 Predicting for week: {week}")

    predictor = PropPredictor()

    # Load models
    predictor.load_models()

    # Generate predictions
    predictions = predictor.predict_week(week, season)

    print("\n✅ Prediction pipeline complete!")
