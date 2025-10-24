"""
Calibration and Evaluation Module
Comprehensive metrics: Brier, LogLoss, CLV, ROI, edge@k, calibration curves
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.calibration import calibration_curve

from phase1_config import Config
from custom_loss_functions import brier_score, log_loss, expected_calibration_error


class CalibrationEvaluator:
    """
    Comprehensive evaluation for NFL prop predictions.
    Focuses on market-oriented metrics (CLV, ROI, edge@k).
    """

    def __init__(self, config: Config = Config):
        self.config = config

    def evaluate_regression_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: np.ndarray = None
    ) -> Dict:
        """
        Evaluate regression model (yards, receptions, etc.)

        Args:
            y_true: Actual values
            y_pred: Predicted values
            sample_weight: Optional sample weights

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Standard regression metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
        metrics['mse'] = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
        metrics['rmse'] = np.sqrt(metrics['mse'])

        # R²
        if sample_weight is None:
            metrics['r2'] = r2_score(y_true, y_pred)
        else:
            metrics['r2'] = r2_score(y_true, y_pred, sample_weight=sample_weight)

        # Median Absolute Error (robust to outliers)
        metrics['median_ae'] = np.median(np.abs(y_true - y_pred))

        # Mean Absolute Percentage Error
        mask = y_true != 0
        if mask.sum() > 0:
            metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['mape'] = np.nan

        # Residual statistics
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)

        # Bias (systematic over/under-prediction)
        metrics['bias'] = np.mean(y_pred - y_true)

        return metrics

    def evaluate_classification_model(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict:
        """
        Evaluate classification model (TDs, binary outcomes).

        Args:
            y_true: Binary outcomes (0/1)
            y_prob: Predicted probabilities

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Brier score (lower is better, 0 = perfect)
        metrics['brier_score'] = brier_score(y_true, y_prob)

        # Log loss (lower is better)
        metrics['log_loss'] = log_loss(y_true, y_prob)

        # Expected Calibration Error
        metrics['ece'] = expected_calibration_error(y_true, y_prob, n_bins=10)

        # Accuracy at default threshold (0.5)
        y_pred_binary = (y_prob > 0.5).astype(int)
        metrics['accuracy'] = np.mean(y_true == y_pred_binary)

        # AUC-ROC
        from sklearn.metrics import roc_auc_score
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc_roc'] = np.nan

        return metrics

    def calculate_closing_line_value(
        self,
        predictions: pd.DataFrame,
        actual_lines: pd.DataFrame,
        pred_col: str = 'prediction',
        line_col: str = 'line'
    ) -> Dict:
        """
        Calculate Closing Line Value (CLV).

        CLV measures how often your prediction beats the closing line.
        Positive CLV = beating the market.

        Args:
            predictions: DataFrame with predictions and initial lines
            actual_lines: DataFrame with closing lines
            pred_col: Column name for predictions
            line_col: Column name for lines

        Returns:
            Dictionary with CLV metrics
        """
        # Merge predictions with closing lines
        merged = predictions.merge(
            actual_lines[['player_id', 'market', 'week', line_col]],
            on=['player_id', 'market', 'week'],
            suffixes=('_open', '_close')
        )

        if len(merged) == 0:
            return {
                'clv_samples': 0,
                'clv_avg': 0,
                'clv_positive_rate': 0
            }

        # Calculate CLV for each prediction
        # If pred > open_line (we'd bet OVER), did closing line move up? (positive CLV)
        # If pred < open_line (we'd bet UNDER), did closing line move down? (positive CLV)

        clv_values = []

        for idx, row in merged.iterrows():
            pred = row[pred_col]
            open_line = row[f'{line_col}_open']
            close_line = row[f'{line_col}_close']

            if pred > open_line:
                # We'd bet OVER
                # Positive CLV if close_line moved up
                clv = close_line - open_line
            elif pred < open_line:
                # We'd bet UNDER
                # Positive CLV if close_line moved down
                clv = open_line - close_line
            else:
                clv = 0

            clv_values.append(clv)

        clv_array = np.array(clv_values)

        return {
            'clv_samples': len(clv_array),
            'clv_avg': np.mean(clv_array),
            'clv_std': np.std(clv_array),
            'clv_positive_rate': np.mean(clv_array > 0),
            'clv_median': np.median(clv_array)
        }

    def calculate_edge_at_k(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
        lines: pd.DataFrame,
        k_values: List[float] = [0.5, 1.0, 1.5, 2.0, 3.0],
        pred_col: str = 'prediction',
        actual_col: str = 'actual',
        line_col: str = 'line'
    ) -> Dict:
        """
        Calculate edge@k: ROI when betting only on edges ≥ k.

        This is THE key metric for prop betting profitability.

        Args:
            predictions: DataFrame with predictions
            actuals: DataFrame with actual results
            lines: DataFrame with betting lines
            k_values: List of edge thresholds to test
            pred_col: Prediction column
            actual_col: Actual result column
            line_col: Line column

        Returns:
            Dictionary with edge@k metrics for each k
        """
        # Merge all data
        df = predictions.merge(actuals, on=['player_id', 'week', 'season', 'market'])
        df = df.merge(lines, on=['player_id', 'week', 'season', 'market'])

        results = {}

        for k in k_values:
            # Filter to edges >= k
            df['edge'] = np.abs(df[pred_col] - df[line_col])
            df_k = df[df['edge'] >= k].copy()

            if len(df_k) == 0:
                results[f'edge@{k}'] = {
                    'n_bets': 0,
                    'hit_rate': 0,
                    'roi': 0,
                    'avg_edge': 0
                }
                continue

            # Determine recommendation (OVER or UNDER)
            df_k['recommendation'] = np.where(
                df_k[pred_col] > df_k[line_col],
                'OVER',
                'UNDER'
            )

            # Determine if bet won
            df_k['won'] = np.where(
                df_k['recommendation'] == 'OVER',
                df_k[actual_col] > df_k[line_col],
                df_k[actual_col] < df_k[line_col]
            )

            # Calculate ROI (assuming -110 odds)
            n_wins = df_k['won'].sum()
            n_bets = len(df_k)
            hit_rate = n_wins / n_bets

            # ROI = (wins * 0.909 - losses) / total_bets
            # At -110: win $0.909 for every $1 bet
            roi = ((n_wins * 0.909) - (n_bets - n_wins)) / n_bets

            results[f'edge@{k}'] = {
                'n_bets': n_bets,
                'hit_rate': hit_rate,
                'roi': roi,
                'avg_edge': df_k['edge'].mean(),
                'wins': n_wins,
                'losses': n_bets - n_wins
            }

        return results

    def evaluate_by_line_bucket(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
        lines: pd.DataFrame,
        buckets: List[Tuple[float, float]] = None,
        pred_col: str = 'prediction',
        actual_col: str = 'actual',
        line_col: str = 'line'
    ) -> pd.DataFrame:
        """
        Evaluate model performance by line bucket.

        Helps identify if model only works for certain line ranges.

        Args:
            predictions: Predictions DataFrame
            actuals: Actuals DataFrame
            lines: Lines DataFrame
            buckets: List of (min, max) tuples for line buckets
            pred_col, actual_col, line_col: Column names

        Returns:
            DataFrame with metrics per bucket
        """
        if buckets is None:
            buckets = [
                (0, 2.5),
                (2.5, 4.5),
                (4.5, 6.5),
                (6.5, 10),
                (10, float('inf'))
            ]

        # Merge data
        df = predictions.merge(actuals, on=['player_id', 'week', 'season', 'market'])
        df = df.merge(lines, on=['player_id', 'week', 'season', 'market'])

        bucket_results = []

        for min_val, max_val in buckets:
            bucket_df = df[(df[line_col] >= min_val) & (df[line_col] < max_val)]

            if len(bucket_df) == 0:
                continue

            y_true = bucket_df[actual_col].values
            y_pred = bucket_df[pred_col].values

            metrics = self.evaluate_regression_model(y_true, y_pred)
            metrics['bucket'] = f'{min_val}-{max_val}'
            metrics['n_samples'] = len(bucket_df)
            metrics['avg_line'] = bucket_df[line_col].mean()

            bucket_results.append(metrics)

        return pd.DataFrame(bucket_results)

    def generate_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate calibration curve for probability predictions.

        Args:
            y_true: Binary outcomes
            y_prob: Predicted probabilities
            n_bins: Number of bins

        Returns:
            (true_frequencies, predicted_probabilities)
        """
        return calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate Sharpe ratio for betting returns.

        Args:
            returns: Array of bet returns (profit/loss per bet)
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Sharpe ratio
        """
        excess_returns = returns - risk_free_rate
        if len(excess_returns) == 0 or np.std(excess_returns) == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        return sharpe

    def calculate_max_drawdown(
        self,
        equity_curve: np.ndarray
    ) -> Dict:
        """
        Calculate maximum drawdown from equity curve.

        Args:
            equity_curve: Array of cumulative equity values

        Returns:
            Dictionary with drawdown metrics
        """
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax

        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)

        # Recovery time (if recovered)
        recovery_idx = None
        if max_dd_idx < len(equity_curve) - 1:
            post_dd = equity_curve[max_dd_idx+1:]
            recovered = post_dd >= cummax[max_dd_idx]
            if recovered.any():
                recovery_idx = max_dd_idx + 1 + np.argmax(recovered)

        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,
            'max_drawdown_idx': max_dd_idx,
            'recovery_idx': recovery_idx,
            'recovery_time': recovery_idx - max_dd_idx if recovery_idx else None
        }

    def comprehensive_backtest_report(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
        lines: pd.DataFrame,
        bet_history: pd.DataFrame = None
    ) -> Dict:
        """
        Generate comprehensive backtest report.

        Args:
            predictions: Predictions DataFrame
            actuals: Actuals DataFrame
            lines: Lines DataFrame
            bet_history: Optional bet history with sizes and results

        Returns:
            Comprehensive metrics dictionary
        """
        report = {}

        # Merge data
        df = predictions.merge(actuals, on=['player_id', 'week', 'season', 'market'])
        df = df.merge(lines, on=['player_id', 'week', 'season', 'market'])

        # Regression metrics
        y_true = df['actual'].values
        y_pred = df['prediction'].values

        report['regression_metrics'] = self.evaluate_regression_model(y_true, y_pred)

        # Edge@k analysis
        edge_results = self.calculate_edge_at_k(
            predictions, actuals, lines,
            k_values=[0.5, 1.0, 1.5, 2.0, 3.0]
        )
        report['edge_at_k'] = edge_results

        # Line bucket analysis
        bucket_analysis = self.evaluate_by_line_bucket(predictions, actuals, lines)
        report['bucket_analysis'] = bucket_analysis.to_dict('records')

        # Bet history analysis (if available)
        if bet_history is not None and len(bet_history) > 0:
            total_wagered = bet_history['bet_size'].sum()
            total_profit = bet_history['result'].sum()
            n_bets = len(bet_history)
            n_wins = (bet_history['result'] > 0).sum()

            report['betting_metrics'] = {
                'total_bets': n_bets,
                'total_wagered': total_wagered,
                'total_profit': total_profit,
                'roi': (total_profit / total_wagered * 100) if total_wagered > 0 else 0,
                'hit_rate': (n_wins / n_bets) if n_bets > 0 else 0,
                'avg_bet_size': total_wagered / n_bets if n_bets > 0 else 0,
                'avg_profit_per_bet': total_profit / n_bets if n_bets > 0 else 0
            }

            # Sharpe ratio
            if 'result' in bet_history.columns:
                returns = bet_history['result'].values
                report['betting_metrics']['sharpe_ratio'] = self.calculate_sharpe_ratio(returns)

            # Max drawdown (if we have equity curve)
            if 'cumulative_profit' in bet_history.columns:
                equity = bet_history['cumulative_profit'].values
                dd = self.calculate_max_drawdown(equity)
                report['betting_metrics']['max_drawdown'] = dd

        return report


if __name__ == "__main__":
    # Test calibration evaluator
    print("\n" + "="*60)
    print("TESTING CALIBRATION & EVALUATION")
    print("="*60)

    evaluator = CalibrationEvaluator()

    # Test 1: Regression metrics
    print("\n1. Regression Metrics:")
    y_true = np.array([65, 48, 92, 31, 75, 58, 83, 44, 67, 71])
    y_pred = np.array([63, 51, 88, 35, 72, 55, 85, 42, 70, 68])

    reg_metrics = evaluator.evaluate_regression_model(y_true, y_pred)
    print(f"   MAE: {reg_metrics['mae']:.2f}")
    print(f"   RMSE: {reg_metrics['rmse']:.2f}")
    print(f"   R²: {reg_metrics['r2']:.4f}")
    print(f"   Bias: {reg_metrics['bias']:.2f}")

    # Test 2: Classification metrics
    print("\n2. Classification Metrics (TDs):")
    y_true_binary = np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.75, 0.2, 0.3, 0.8, 0.15, 0.65, 0.25, 0.1, 0.85, 0.2])

    class_metrics = evaluator.evaluate_classification_model(y_true_binary, y_prob)
    print(f"   Brier Score: {class_metrics['brier_score']:.4f}")
    print(f"   Log Loss: {class_metrics['log_loss']:.4f}")
    print(f"   ECE: {class_metrics['ece']:.4f}")
    print(f"   Accuracy: {class_metrics['accuracy']:.2%}")

    # Test 3: Edge@k
    print("\n3. Edge@k Analysis:")
    # Simulate data
    np.random.seed(42)
    n = 100

    edge_data = pd.DataFrame({
        'player_id': [f'p{i}' for i in range(n)],
        'week': np.random.randint(1, 9, n),
        'season': 2025,
        'market': 'receiving_yards',
        'prediction': np.random.normal(65, 20, n),
        'actual': np.random.normal(65, 20, n),
        'line': np.random.normal(62, 18, n)
    })

    predictions_df = edge_data[['player_id', 'week', 'season', 'market', 'prediction']]
    actuals_df = edge_data[['player_id', 'week', 'season', 'market', 'actual']]
    lines_df = edge_data[['player_id', 'week', 'season', 'market', 'line']]

    edge_results = evaluator.calculate_edge_at_k(
        predictions_df, actuals_df, lines_df,
        k_values=[1.0, 2.0, 3.0]
    )

    for k, metrics in edge_results.items():
        print(f"\n   {k}:")
        print(f"     Bets: {metrics['n_bets']}")
        print(f"     Hit Rate: {metrics['hit_rate']:.2%}")
        print(f"     ROI: {metrics['roi']*100:.2f}%")

    print("\n✅ Calibration & evaluation tested successfully!")
