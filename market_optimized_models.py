"""
Market-Optimized Models for NFL Player Props
Position-specific quantile regression with isotonic calibration
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

from phase1_config import Config
from custom_loss_functions import (
    poisson_probability_over_line,
    lognormal_probability_over_line,
    brier_score,
    log_loss
)


class MarketOptimizedModel:
    """
    Market-optimized model ensemble for player props.

    Uses:
    - XGBoost Quantile (τ=position-specific)
    - LightGBM Quantile (τ=position-specific)
    - XGBoost Mean (fallback)
    - Isotonic calibration on out-of-fold predictions
    """

    def __init__(
        self,
        market_type: str,
        config: Config = Config,
        target_quantile: float = None
    ):
        self.market_type = market_type
        self.config = config

        # Get position-specific quantile if not provided
        if target_quantile is None:
            self.target_quantile = config.get_quantile_target(market_type)
        else:
            self.target_quantile = target_quantile

        # Initialize models
        self.models = {}
        self.scaler = StandardScaler()
        self.calibrator = IsotonicRegression(out_of_bounds='clip')

        # Training metadata
        self.feature_names = None
        self.oof_predictions = None
        self.cv_scores = {}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        game_ids_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        sample_weight: np.ndarray = None
    ):
        """
        Train model ensemble with GroupKFold cross-validation.

        Args:
            X_train: Training features
            y_train: Training targets
            game_ids_train: Game IDs for GroupKFold
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            sample_weight: Sample weights (for historical vs recent weighting)
        """
        print(f"\n{'='*60}")
        print(f"Training {self.market_type} model (τ={self.target_quantile:.2f})")
        print(f"{'='*60}")

        self.feature_names = X_train.columns.tolist()

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

        # Initialize out-of-fold predictions
        oof_preds = np.zeros(len(X_train))

        # GroupKFold cross-validation
        gkf = GroupKFold(n_splits=self.config.N_SPLITS)

        fold_scores = []

        print(f"\nRunning {self.config.N_SPLITS}-fold GroupKFold CV...")

        for fold_idx, (train_idx, val_idx) in enumerate(
            gkf.split(X_train_df, y_train, game_ids_train)
        ):
            print(f"\nFold {fold_idx + 1}/{self.config.N_SPLITS}")

            # Split data
            X_tr = X_train_df.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_vl = X_train_df.iloc[val_idx]
            y_vl = y_train.iloc[val_idx]

            # Sample weights
            if sample_weight is not None:
                sw_tr = sample_weight[train_idx]
            else:
                sw_tr = None

            # Train XGBoost Quantile
            xgb_quantile = self._train_xgb_quantile(X_tr, y_tr, X_vl, y_vl, sw_tr)

            # Train LightGBM Quantile
            lgb_quantile = self._train_lgb_quantile(X_tr, y_tr, X_vl, y_vl, sw_tr)

            # Ensemble predictions for this fold
            xgb_pred = xgb_quantile.predict(X_vl)
            lgb_pred = lgb_quantile.predict(X_vl)

            weights = self.config.ENSEMBLE_WEIGHTS
            fold_pred = weights['xgb_quantile'] * xgb_pred + weights['lgb_quantile'] * lgb_pred

            # Store OOF predictions
            oof_preds[val_idx] = fold_pred

            # Evaluate fold
            mae = np.mean(np.abs(y_vl - fold_pred))
            fold_scores.append(mae)

            print(f"  Fold MAE: {mae:.4f}")

        # Final models on full training data
        print("\nTraining final models on full data...")

        self.models['xgb_quantile'] = self._train_xgb_quantile(
            X_train_df, y_train, X_val, y_val, sample_weight
        )

        self.models['lgb_quantile'] = self._train_lgb_quantile(
            X_train_df, y_train, X_val, y_val, sample_weight
        )

        # Mean model as fallback
        self.models['xgb_mean'] = self._train_xgb_mean(
            X_train_df, y_train, X_val, y_val, sample_weight
        )

        # Fit isotonic calibration on OOF predictions
        if self.config.ISOTONIC_CALIBRATION_ENABLED:
            print("\nFitting isotonic calibration...")
            self.calibrator.fit(oof_preds, y_train)
            self.oof_predictions = oof_preds

        # Store CV scores
        self.cv_scores = {
            'mean_mae': np.mean(fold_scores),
            'std_mae': np.std(fold_scores),
            'fold_scores': fold_scores
        }

        print(f"\n✅ Training complete!")
        print(f"   CV MAE: {self.cv_scores['mean_mae']:.4f} ± {self.cv_scores['std_mae']:.4f}")

    def predict(self, X: pd.DataFrame, return_components: bool = False) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Features
            return_components: If True, return dict with all model components

        Returns:
            Calibrated predictions (or dict if return_components=True)
        """
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_df = pd.DataFrame(X_scaled, columns=self.feature_names)

        # Get predictions from each model
        xgb_q_pred = self.models['xgb_quantile'].predict(X_df)
        lgb_q_pred = self.models['lgb_quantile'].predict(X_df)
        xgb_m_pred = self.models['xgb_mean'].predict(X_df)

        # Ensemble
        weights = self.config.ENSEMBLE_WEIGHTS
        raw_pred = weights['xgb_quantile'] * xgb_q_pred + weights['lgb_quantile'] * lgb_q_pred

        # Calibrate
        if self.config.ISOTONIC_CALIBRATION_ENABLED and self.calibrator is not None:
            calibrated_pred = self.calibrator.predict(raw_pred)
        else:
            calibrated_pred = raw_pred

        if return_components:
            return {
                'prediction': calibrated_pred,
                'raw_quantile': raw_pred,
                'xgb_quantile': xgb_q_pred,
                'lgb_quantile': lgb_q_pred,
                'mean_pred': xgb_m_pred
            }
        else:
            return calibrated_pred

    def _train_xgb_quantile(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        sample_weight: np.ndarray = None
    ) -> XGBRegressor:
        """Train XGBoost quantile model"""
        params = self.config.XGB_PARAMS.copy()
        params['objective'] = 'reg:quantileerror'
        params['quantile_alpha'] = self.target_quantile

        model = XGBRegressor(**params)

        eval_set = [(X_val, y_val)] if X_val is not None else None

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=eval_set,
            verbose=False
        )

        return model

    def _train_lgb_quantile(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        sample_weight: np.ndarray = None
    ) -> LGBMRegressor:
        """Train LightGBM quantile model"""
        params = self.config.LGBM_PARAMS.copy()
        params['objective'] = 'quantile'
        params['alpha'] = self.target_quantile

        model = LGBMRegressor(**params)

        eval_set = [(X_val, y_val)] if X_val is not None else None

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=eval_set
        )

        return model

    def _train_xgb_mean(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        sample_weight: np.ndarray = None
    ) -> XGBRegressor:
        """Train XGBoost mean model (fallback)"""
        params = self.config.XGB_PARAMS.copy()
        params['objective'] = 'reg:squarederror'
        params['n_estimators'] = 300

        model = XGBRegressor(**params)

        eval_set = [(X_val, y_val)] if X_val is not None else None

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=eval_set,
            verbose=False
        )

        return model

    def save(self, filepath: str):
        """Save model to disk"""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'calibrator': self.calibrator,
            'feature_names': self.feature_names,
            'market_type': self.market_type,
            'target_quantile': self.target_quantile,
            'cv_scores': self.cv_scores,
            'oof_predictions': self.oof_predictions
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, filepath: str):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Reconstruct model
        model = cls(
            market_type=model_data['market_type'],
            target_quantile=model_data['target_quantile']
        )

        model.models = model_data['models']
        model.scaler = model_data['scaler']
        model.calibrator = model_data['calibrator']
        model.feature_names = model_data['feature_names']
        model.cv_scores = model_data['cv_scores']
        model.oof_predictions = model_data.get('oof_predictions')

        return model


class TDProbabilityModel:
    """
    Specialized model for touchdown predictions.
    Uses calibrated classification (binary or multi-class).
    """

    def __init__(
        self,
        market_type: str,
        config: Config = Config,
        calibration_method: str = 'isotonic'
    ):
        self.market_type = market_type
        self.config = config
        self.calibration_method = calibration_method

        # Models
        self.models = {}
        self.scaler = StandardScaler()

        # Training metadata
        self.feature_names = None
        self.cv_scores = {}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        game_ids_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        sample_weight: np.ndarray = None
    ):
        """
        Train TD probability model.

        Predicts P(TD ≥ 1) for any-time TD bets.
        """
        print(f"\n{'='*60}")
        print(f"Training {self.market_type} TD Probability Model")
        print(f"{'='*60}")

        self.feature_names = X_train.columns.tolist()

        # Convert to binary (TD or no TD)
        y_train_binary = (y_train > 0).astype(int)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

        # GroupKFold
        gkf = GroupKFold(n_splits=self.config.N_SPLITS)

        oof_probs = np.zeros(len(X_train))
        fold_scores = []

        print(f"\nRunning {self.config.N_SPLITS}-fold GroupKFold CV...")

        for fold_idx, (train_idx, val_idx) in enumerate(
            gkf.split(X_train_df, y_train_binary, game_ids_train)
        ):
            print(f"\nFold {fold_idx + 1}/{self.config.N_SPLITS}")

            X_tr = X_train_df.iloc[train_idx]
            y_tr = y_train_binary.iloc[train_idx]
            X_vl = X_train_df.iloc[val_idx]
            y_vl = y_train_binary.iloc[val_idx]

            # Train XGBClassifier
            xgb_model = XGBClassifier(
                **self.config.XGB_PARAMS,
                objective='binary:logistic',
                eval_metric='logloss'
            )

            xgb_model.fit(X_tr, y_tr, verbose=False)

            # Predict probabilities
            fold_probs = xgb_model.predict_proba(X_vl)[:, 1]
            oof_probs[val_idx] = fold_probs

            # Evaluate
            brier = brier_score(y_vl, fold_probs)
            logloss = log_loss(y_vl, fold_probs)

            fold_scores.append({'brier': brier, 'logloss': logloss})

            print(f"  Brier: {brier:.4f}, LogLoss: {logloss:.4f}")

        # Train final model
        print("\nTraining final model on full data...")

        self.models['xgb_classifier'] = XGBClassifier(
            **self.config.XGB_PARAMS,
            objective='binary:logistic',
            eval_metric='logloss'
        )

        self.models['xgb_classifier'].fit(X_train_df, y_train_binary, verbose=False)

        # Calibrate probabilities
        print(f"\nCalibrating with {self.calibration_method} regression...")

        if self.calibration_method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(oof_probs, y_train_binary)
            self.models['calibrator'] = calibrator

        # Store scores
        avg_brier = np.mean([s['brier'] for s in fold_scores])
        avg_logloss = np.mean([s['logloss'] for s in fold_scores])

        self.cv_scores = {
            'brier': avg_brier,
            'logloss': avg_logloss,
            'fold_scores': fold_scores
        }

        print(f"\n✅ Training complete!")
        print(f"   CV Brier: {avg_brier:.4f}")
        print(f"   CV LogLoss: {avg_logloss:.4f}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict calibrated probabilities.

        Returns:
            Array of P(TD ≥ 1) probabilities
        """
        # Scale
        X_scaled = self.scaler.transform(X)
        X_df = pd.DataFrame(X_scaled, columns=self.feature_names)

        # Raw probabilities
        raw_probs = self.models['xgb_classifier'].predict_proba(X_df)[:, 1]

        # Calibrate
        if 'calibrator' in self.models:
            calibrated_probs = self.models['calibrator'].predict(raw_probs)
        else:
            calibrated_probs = raw_probs

        return calibrated_probs

    def save(self, filepath: str):
        """Save model"""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'market_type': self.market_type,
            'calibration_method': self.calibration_method,
            'cv_scores': self.cv_scores
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, filepath: str):
        """Load model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        model = cls(
            market_type=model_data['market_type'],
            calibration_method=model_data['calibration_method']
        )

        model.models = model_data['models']
        model.scaler = model_data['scaler']
        model.feature_names = model_data['feature_names']
        model.cv_scores = model_data['cv_scores']

        return model


if __name__ == "__main__":
    # Test models
    print("\n" + "="*60)
    print("TESTING MARKET-OPTIMIZED MODELS")
    print("="*60)

    # Generate sample data
    np.random.seed(42)

    n_samples = 500
    n_features = 20

    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    y_train_yards = pd.Series(np.random.gamma(5, 10, n_samples))
    y_train_tds = pd.Series(np.random.poisson(0.3, n_samples))

    # Create fake game IDs
    game_ids = pd.Series([f'game_{i//10}' for i in range(n_samples)])

    # Test MarketOptimizedModel
    print("\n1. Testing MarketOptimizedModel (receiving_yards):")
    yards_model = MarketOptimizedModel('receiving_yards')

    yards_model.train(
        X_train[:400],
        y_train_yards[:400],
        game_ids[:400],
        X_train[400:],
        y_train_yards[400:]
    )

    # Predict
    preds = yards_model.predict(X_train[400:410])
    print(f"\nSample predictions: {preds[:5]}")
    print(f"Sample actuals: {y_train_yards[400:405].values}")

    # Test TDProbabilityModel
    print("\n\n2. Testing TDProbabilityModel:")
    td_model = TDProbabilityModel('receiving_tds')

    td_model.train(
        X_train[:400],
        y_train_tds[:400],
        game_ids[:400],
        X_train[400:],
        y_train_tds[400:]
    )

    # Predict probabilities
    probs = td_model.predict_proba(X_train[400:410])
    print(f"\nSample TD probabilities: {probs[:5]}")
    print(f"Sample actuals (TDs): {y_train_tds[400:405].values}")

    print("\n✅ Model tests complete!")
