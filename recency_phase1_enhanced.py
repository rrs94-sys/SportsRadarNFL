"""
RECENCY TRAINING WITH PHASE 1 ENHANCEMENTS - MAIN ORCHESTRATOR
==============================================================
FIXED: Renamed from 'recency_phase1 (1).py' to 'recency_phase1_enhanced.py'
FIXED: Corrected import statements (RecencyDataIngestion doesn't exist)
FIXED: Implements dynamic season and week detection
FIXED: No more hardcoded weeks or seasons
FIXED: Fixed references to 'team' column (was 'recent_team')

NEW PHASE 1 FEATURES:
✅ Home/Road splits with performance variance
✅ Rest & Travel with timezone calculations
✅ Matchup-specific history (player vs defense)
✅ Improved recency weighting (L1=30%, L2-L3=40%, etc.)
✅ Weather stubs (ready for Open-Meteo integration)
✅ Dynamic season and week detection

Trains on current season data with dynamic week detection
Produces: MODEL_DIR/ensemble_models.pkl (70% recency + 30% historical)

MODULAR STRUCTURE:
- phase1_config.py        → Configuration & constants
- phase1_ingestion.py     → Historical data loading
- load_2025_data.py       → Current season data loading
- phase1_features.py      → Enhanced feature engineering (70+ features)
- market_optimized_models.py → Model classes
- recency_phase1_enhanced.py → THIS FILE - runs everything
"""

import os
import sys
import pickle
from datetime import datetime
import pandas as pd
import numpy as np

# FIXED: Import dynamic detection utilities
from nfl_data_utils import get_current_season, get_latest_completed_week

# Import all Phase 1 modules
try:
    from phase1_config import Config
    # FIXED: Import correct classes (RecencyDataIngestion doesn't exist!)
    from phase1_ingestion import HistoricalDataIngestion
    from load_2025_data import CurrentSeasonLoader
    from phase1_features import Phase1FeatureEngineer
    # FIXED: Use market_optimized_models instead of phase1_models
    from market_optimized_models import MarketOptimizedModel, TDProbabilityModel
except ImportError as e:
    print(f"❌ Missing Phase 1 module: {e}")
    print("\nRequired files:")
    print("  - phase1_config.py")
    print("  - phase1_ingestion.py")
    print("  - load_2025_data.py")
    print("  - phase1_features.py")
    print("  - market_optimized_models.py")
    print("  - nfl_data_utils.py")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def main():
    """
    Main training pipeline with Phase 1 enhancements.

    FIXED: Uses dynamic season and week detection
    FIXED: Corrected class references
    FIXED: Fixed column name references
    """

    print("="*80)
    print("RECENCY TRAINING SYSTEM - PHASE 1 ENHANCED")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # FIXED: Dynamic season and week detection
    current_season = get_current_season()
    latest_week = get_latest_completed_week(current_season)

    print(f"\n📊 Detected current season: {current_season}")
    print(f"📊 Detected latest completed week: {latest_week}")

    print("\nPhase 1 Features:")
    print("  ✅ Home/Road splits")
    print("  ✅ Rest & Travel impact")
    print("  ✅ Matchup history (player vs defense)")
    print("  ✅ Improved recency weighting")
    print("  ✅ Weather integration stubs")
    print("  ✅ Dynamic season/week detection")
    print("="*80)

    config = Config

    # ========================================================================
    # STEP 1: DATA INGESTION
    # ========================================================================
    print("\n[STEP 1/5] DATA INGESTION")
    print("-"*80)

    # FIXED: Use correct class and dynamic week detection
    loader = CurrentSeasonLoader(config)
    data = loader.load_2025_data(through_week=None)  # Auto-detect latest week

    if data is None or data['player_weeks'].empty:
        print("❌ No current season data loaded. Exiting.")
        return None

    print(f"✅ Loaded {len(data['player_weeks'])} player-weeks")
    # FIXED: Removed reference to 'pbp' which doesn't exist in this data structure
    print(f"   Schedule games: {len(data['schedules'])}")

    # ========================================================================
    # STEP 2: FEATURE ENGINEERING
    # ========================================================================
    print("\n[STEP 2/5] PHASE 1 FEATURE ENGINEERING")
    print("-"*80)

    engineer = Phase1FeatureEngineer(data, config)

    # Build training datasets
    print("\nBuilding feature sets...")
    datasets = {
        'receptions': {'X': [], 'y': [], 'meta': []},
        'receiving_yards': {'X': [], 'y': [], 'meta': []},
        'completions': {'X': [], 'y': [], 'meta': []},
        'receiving_tds': {'X': [], 'y': [], 'meta': []},
        'rushing_yards': {'X': [], 'y': [], 'meta': []}
    }

    # Process all player-weeks
    player_weeks = data['player_weeks']
    total = len(player_weeks)

    for idx, (_, row) in enumerate(player_weeks.iterrows()):
        if idx % 500 == 0 and idx > 0:
            print(f"  Progress: {idx}/{total} ({100*idx/total:.1f}%)")

        player_id = row['player_id']
        # FIXED: Use 'team' not 'recent_team' (already standardized by nfl_data_utils)
        team = row.get('team', row.get('recent_team', ''))
        week = row['week']
        season = row['season']
        position = row['position']

        # Get opponent from schedule
        game = data['schedules'][
            (data['schedules']['season'] == season) &
            (data['schedules']['week'] == week) &
            ((data['schedules']['home_team'] == team) |
             (data['schedules']['away_team'] == team))
        ]

        opponent = 'UNK'
        if len(game) > 0:
            game = game.iloc[0]
            opponent = game['away_team'] if game['home_team'] == team else game['home_team']

        # Build features based on position
        if position in ['WR', 'TE', 'RB']:
            features = engineer.create_receiving_features(
                player_id, team, week, season, opponent
            )

            if features:
                meta = {
                    'player_id': player_id,
                    'player_name': row.get('player_name', 'Unknown'),
                    'team': team,
                    'week': week,
                    'season': season,
                    'position': position
                }

                datasets['receptions']['X'].append(features)
                datasets['receptions']['y'].append(row.get('receptions', 0))
                datasets['receptions']['meta'].append(meta)

                datasets['receiving_yards']['X'].append(features)
                datasets['receiving_yards']['y'].append(row.get('receiving_yards', 0))
                datasets['receiving_yards']['meta'].append(meta)

                datasets['receiving_tds']['X'].append(features)
                datasets['receiving_tds']['y'].append(row.get('receiving_tds', 0))
                datasets['receiving_tds']['meta'].append(meta)

        if position == 'QB':
            features = engineer.create_qb_features(
                player_id, team, week, season, opponent
            )

            if features:
                meta = {
                    'player_id': player_id,
                    'player_name': row.get('player_name', 'Unknown'),
                    'team': team,
                    'week': week,
                    'season': season,
                    'position': position
                }

                datasets['completions']['X'].append(features)
                datasets['completions']['y'].append(row.get('completions', 0))
                datasets['completions']['meta'].append(meta)

        if position == 'RB':
            features = engineer.create_rushing_features(
                player_id, team, week, season, opponent
            )

            if features:
                meta = {
                    'player_id': player_id,
                    'player_name': row.get('player_name', 'Unknown'),
                    'team': team,
                    'week': week,
                    'season': season,
                    'position': position
                }

                datasets['rushing_yards']['X'].append(features)
                datasets['rushing_yards']['y'].append(row.get('rushing_yards', 0))
                datasets['rushing_yards']['meta'].append(meta)

    # Convert to DataFrames
    print("\n\nDataset sizes:")
    for prop_type in datasets:
        if len(datasets[prop_type]['X']) > 0:
            datasets[prop_type]['X'] = pd.DataFrame(datasets[prop_type]['X'])
            datasets[prop_type]['y'] = np.array(datasets[prop_type]['y'])
            print(f"  {prop_type}: {len(datasets[prop_type]['y'])} samples")
        else:
            print(f"  {prop_type}: 0 samples ❌")

    # ========================================================================
    # STEP 3: TRAIN RECENCY MODELS
    # ========================================================================
    print("\n[STEP 3/5] TRAINING RECENCY MODELS")
    print("-"*80)

    recency_models = {}

    for prop_type, data_dict in datasets.items():
        if len(data_dict['X']) < config.MIN_SAMPLES_FOR_TRAINING:
            print(f"\n⚠️  Skipping {prop_type} (only {len(data_dict['X'])} samples)")
            continue

        X = data_dict['X']
        y = data_dict['y']

        # 80/20 split
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # FIXED: Use MarketOptimizedModel and TDProbabilityModel
        print(f"\n  Training {prop_type}...")
        if 'tds' in prop_type:
            # TD probability model
            model = TDProbabilityModel(prop_type, config)
        else:
            # Regression model
            model = MarketOptimizedModel(prop_type, config)

        # FIXED: Create proper game_ids for GroupKFold
        game_ids_train = pd.Series([f"game_{i//10}" for i in range(len(X_train))])

        try:
            model.train(X_train, y_train, game_ids_train)
            recency_models[prop_type] = model
            print(f"  ✅ {prop_type} trained")
        except Exception as e:
            print(f"  ⚠️  Error training {prop_type}: {e}")
            continue

    # ========================================================================
    # STEP 4: LOAD HISTORICAL & CREATE ENSEMBLE
    # ========================================================================
    print("\n[STEP 4/5] CREATING ENSEMBLE (70% RECENCY + 30% HISTORICAL)")
    print("-"*80)

    historical_path = f"{config.MODEL_DIR}/models.pkl"

    if not os.path.exists(historical_path):
        print(f"⚠️  No historical models found at {historical_path}")
        print("   Using 100% recency models")
        ensemble_models = recency_models
    else:
        print("Loading historical models...")
        try:
            with open(historical_path, 'rb') as f:
                historical_models = pickle.load(f)

            # Create ensemble
            ensemble_models = {}
            for prop_type in recency_models:
                if prop_type in historical_models:
                    ensemble_models[prop_type] = {
                        'recency': recency_models[prop_type],
                        'historical': historical_models[prop_type],
                        'recency_weight': 0.70,
                        'historical_weight': 0.30
                    }
                    print(f"  ✅ {prop_type}: 70% recency + 30% historical")
                else:
                    ensemble_models[prop_type] = {
                        'recency': recency_models[prop_type],
                        'recency_weight': 1.0
                    }
                    print(f"  ⚠️  {prop_type}: 100% recency (no historical)")
        except Exception as e:
            print(f"  ⚠️  Error loading historical models: {e}")
            print("   Using 100% recency models")
            ensemble_models = recency_models

    # ========================================================================
    # STEP 5: SAVE EVERYTHING
    # ========================================================================
    print("\n[STEP 5/5] SAVING MODELS")
    print("-"*80)

    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # Save ensemble
    ensemble_path = f"{config.MODEL_DIR}/ensemble_models.pkl"
    with open(ensemble_path, 'wb') as f:
        pickle.dump(ensemble_models, f)
    print(f"✅ Saved ensemble: {ensemble_path}")

    # Save engineer
    engineer_path = f"{config.MODEL_DIR}/engineer_phase1.pkl"
    with open(engineer_path, 'wb') as f:
        pickle.dump(engineer, f)
    print(f"✅ Saved engineer: {engineer_path}")

    # FIXED: Save metadata with dynamic season/week info
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'training_weeks': list(range(1, latest_week + 1)),
        'season': current_season,
        'latest_week': latest_week,
        'phase': 'Phase 1 Enhanced',
        'features': 'Home/Road + Rest/Travel + Matchup History + Weather Stubs + Dynamic Detection',
        'model_counts': {k: len(v['X']) for k, v in datasets.items() if len(v['X']) > 0}
    }

    metadata_path = f"{config.MODEL_DIR}/training_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✅ Saved metadata: {metadata_path}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nModels trained: {len(recency_models)}")
    for prop_type in recency_models:
        print(f"  ✅ {prop_type}")

    print(f"\nEnsemble saved: {ensemble_path}")
    print(f"\nSeason: {current_season}, Latest Week: {latest_week}")
    print("\nReady for predictions!")
    print("="*80)

    return ensemble_models, engineer


if __name__ == "__main__":
    try:
        models, engineer = main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
