"""
Monte Carlo Simulation Engine
10,000+ iterations to estimate hit rates and confidence intervals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
from phase1_config import Config


class MonteCarloEngine:
    """
    Monte Carlo simulation for player prop predictions.
    Generates distribution of outcomes based on model uncertainty.
    """

    def __init__(self, n_iterations: int = None, config: Config = Config):
        self.n_iterations = n_iterations or config.MONTE_CARLO_ITERATIONS
        self.config = config

    def simulate_prop_outcomes(
        self,
        prediction: float,
        line: float,
        std_dev: float = None,
        distribution: str = 'normal',
        distribution_params: Dict = None
    ) -> Dict:
        """
        Run Monte Carlo simulation for a single prop.

        Args:
            prediction: Model's point prediction
            line: Betting line
            std_dev: Standard deviation (if None, estimated)
            distribution: 'normal', 'lognormal', 'poisson', or 'negbinom'
            distribution_params: Additional distribution parameters

        Returns:
            Dictionary with simulation results
        """
        if std_dev is None:
            # Estimate from CV if normal
            cv = distribution_params.get('cv', 0.3) if distribution_params else 0.3
            std_dev = prediction * cv

        # Generate samples
        if distribution == 'normal':
            samples = np.random.normal(prediction, std_dev, self.n_iterations)
            samples = np.maximum(samples, 0)  # Ensure non-negative

        elif distribution == 'lognormal':
            # Convert to log-scale parameters
            sigma_ln = distribution_params.get('sigma_ln', 0.5) if distribution_params else 0.5
            mu_ln = np.log(prediction) - 0.5 * sigma_ln**2
            samples = np.random.lognormal(mu_ln, sigma_ln, self.n_iterations)

        elif distribution == 'poisson':
            lambda_param = max(prediction, 0.1)
            samples = np.random.poisson(lambda_param, self.n_iterations)

        elif distribution == 'negbinom':
            alpha = distribution_params.get('alpha', 0.2) if distribution_params else 0.2
            r = prediction / alpha
            p = r / (r + prediction)
            samples = np.random.negative_binomial(r, p, self.n_iterations)

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Calculate hit rates
        over_count = np.sum(samples > line)
        under_count = np.sum(samples < line)
        push_count = np.sum(samples == line)

        over_rate = over_count / self.n_iterations
        under_rate = under_count / self.n_iterations
        push_rate = push_count / self.n_iterations

        # Confidence intervals
        ci_level = self.config.MC_CONFIDENCE_LEVEL
        lower_pct = (1 - ci_level) / 2
        upper_pct = 1 - lower_pct

        confidence_interval = (
            np.percentile(samples, lower_pct * 100),
            np.percentile(samples, upper_pct * 100)
        )

        # Distribution statistics
        results = {
            'prediction': prediction,
            'line': line,
            'n_iterations': self.n_iterations,
            'over_rate': over_rate,
            'under_rate': under_rate,
            'push_rate': push_rate,
            'mean_simulated': np.mean(samples),
            'median_simulated': np.median(samples),
            'std_simulated': np.std(samples),
            'confidence_interval': confidence_interval,
            'ci_level': ci_level,
            'samples': samples,  # Keep for analysis
            'percentiles': {
                'p10': np.percentile(samples, 10),
                'p25': np.percentile(samples, 25),
                'p50': np.percentile(samples, 50),
                'p75': np.percentile(samples, 75),
                'p90': np.percentile(samples, 90)
            }
        }

        return results

    def simulate_multiple_props(
        self,
        predictions_df: pd.DataFrame,
        prediction_col: str = 'prediction',
        line_col: str = 'line',
        std_col: str = 'std_dev',
        distribution_col: str = 'distribution'
    ) -> pd.DataFrame:
        """
        Run Monte Carlo for multiple props.

        Args:
            predictions_df: DataFrame with predictions and lines
            prediction_col: Column name for predictions
            line_col: Column name for lines
            std_col: Column name for standard deviations (optional)
            distribution_col: Column name for distribution type (optional)

        Returns:
            DataFrame with simulation results
        """
        results = []

        for idx, row in predictions_df.iterrows():
            prediction = row[prediction_col]
            line = row[line_col]
            std_dev = row.get(std_col, None)
            distribution = row.get(distribution_col, 'normal')

            sim_result = self.simulate_prop_outcomes(
                prediction=prediction,
                line=line,
                std_dev=std_dev,
                distribution=distribution
            )

            # Combine with original row
            result = row.to_dict()
            result.update({
                'mc_over_rate': sim_result['over_rate'],
                'mc_under_rate': sim_result['under_rate'],
                'mc_mean': sim_result['mean_simulated'],
                'mc_std': sim_result['std_simulated'],
                'mc_ci_lower': sim_result['confidence_interval'][0],
                'mc_ci_upper': sim_result['confidence_interval'][1],
                'mc_p10': sim_result['percentiles']['p10'],
                'mc_p90': sim_result['percentiles']['p90']
            })

            results.append(result)

        return pd.DataFrame(results)

    def filter_by_hit_rate(
        self,
        simulations_df: pd.DataFrame,
        min_hit_rate: float = None,
        recommendation_col: str = 'recommendation'
    ) -> pd.DataFrame:
        """
        Filter predictions by minimum Monte Carlo hit rate.

        Args:
            simulations_df: DataFrame from simulate_multiple_props()
            min_hit_rate: Minimum hit rate threshold
            recommendation_col: Column indicating OVER or UNDER

        Returns:
            Filtered DataFrame
        """
        if min_hit_rate is None:
            min_hit_rate = self.config.MC_MIN_HIT_RATE

        filtered = simulations_df.copy()

        # Filter based on recommendation
        if recommendation_col in filtered.columns:
            over_mask = (
                (filtered[recommendation_col] == 'OVER') &
                (filtered['mc_over_rate'] >= min_hit_rate)
            )
            under_mask = (
                (filtered[recommendation_col] == 'UNDER') &
                (filtered['mc_under_rate'] >= min_hit_rate)
            )

            filtered = filtered[over_mask | under_mask]

        return filtered

    def calculate_kelly_with_mc(
        self,
        mc_hit_rate: float,
        odds: float = -110,
        edge_buffer: float = 0.02
    ) -> float:
        """
        Calculate Kelly fraction using Monte Carlo hit rate.

        Args:
            mc_hit_rate: Hit rate from Monte Carlo simulation
            odds: American odds (e.g., -110)
            edge_buffer: Conservative buffer to reduce edge

        Returns:
            Kelly fraction
        """
        # Convert American odds to decimal
        if odds < 0:
            decimal_odds = 1 + (100 / abs(odds))
        else:
            decimal_odds = 1 + (odds / 100)

        # Implied probability from odds
        implied_prob = 1 / decimal_odds

        # Edge (with buffer)
        edge = (mc_hit_rate - implied_prob) - edge_buffer

        if edge <= 0:
            return 0.0

        # Kelly fraction: (p * decimal_odds - 1) / (decimal_odds - 1)
        kelly = (mc_hit_rate * decimal_odds - 1) / (decimal_odds - 1)

        # Apply fractional Kelly
        kelly_fraction = kelly * self.config.KELLY_FRACTION

        return max(0, kelly_fraction)

    def estimate_distribution_params(
        self,
        historical_values: np.ndarray,
        distribution: str = 'auto'
    ) -> Dict:
        """
        Estimate distribution parameters from historical data.

        Args:
            historical_values: Array of historical outcomes
            distribution: Distribution type or 'auto' to select best fit

        Returns:
            Dictionary of distribution parameters
        """
        if distribution == 'auto':
            # Simple heuristic: count data vs continuous
            is_count = np.all(historical_values == historical_values.astype(int))

            if is_count:
                mean_val = np.mean(historical_values)
                var_val = np.var(historical_values)

                if var_val > mean_val * 1.2:
                    distribution = 'negbinom'
                else:
                    distribution = 'poisson'
            else:
                distribution = 'lognormal'

        params = {'distribution': distribution}

        if distribution == 'poisson':
            params['lambda'] = np.mean(historical_values)

        elif distribution == 'negbinom':
            mean_val = np.mean(historical_values)
            var_val = np.var(historical_values)
            params['alpha'] = (var_val - mean_val) / (mean_val ** 2) if mean_val > 0 else 0.2

        elif distribution == 'lognormal':
            # Estimate sigma from CV
            mean_val = np.mean(historical_values)
            std_val = np.std(historical_values)
            cv = std_val / mean_val if mean_val > 0 else 0.5
            params['sigma_ln'] = np.sqrt(np.log(cv**2 + 1))

        elif distribution == 'normal':
            params['std'] = np.std(historical_values)
            params['cv'] = np.std(historical_values) / np.mean(historical_values) if np.mean(historical_values) > 0 else 0.3

        return params


if __name__ == "__main__":
    # Test Monte Carlo engine
    print("\n" + "="*60)
    print("TESTING MONTE CARLO ENGINE")
    print("="*60)

    engine = MonteCarloEngine(n_iterations=10000)

    # Test 1: Normal distribution
    print("\n1. Receiving Yards (Normal):")
    result = engine.simulate_prop_outcomes(
        prediction=65,
        line=62.5,
        std_dev=15,
        distribution='normal'
    )

    print(f"   Prediction: {result['prediction']}")
    print(f"   Line: {result['line']}")
    print(f"   Over Rate: {result['over_rate']:.4f}")
    print(f"   Under Rate: {result['under_rate']:.4f}")
    print(f"   95% CI: ({result['confidence_interval'][0]:.1f}, {result['confidence_interval'][1]:.1f})")

    # Test 2: Poisson distribution
    print("\n2. Receptions (Poisson):")
    result = engine.simulate_prop_outcomes(
        prediction=5.5,
        line=5.5,
        distribution='poisson'
    )

    print(f"   Prediction: {result['prediction']}")
    print(f"   Line: {result['line']}")
    print(f"   Over Rate: {result['over_rate']:.4f}")
    print(f"   Under Rate: {result['under_rate']:.4f}")
    print(f"   Percentiles: p25={result['percentiles']['p25']}, p75={result['percentiles']['p75']}")

    # Test 3: Log-normal distribution
    print("\n3. Yards (Log-Normal):")
    result = engine.simulate_prop_outcomes(
        prediction=75,
        line=70.5,
        distribution='lognormal',
        distribution_params={'sigma_ln': 0.4}
    )

    print(f"   Prediction: {result['prediction']}")
    print(f"   Line: {result['line']}")
    print(f"   Over Rate: {result['over_rate']:.4f}")
    print(f"   Mean Simulated: {result['mean_simulated']:.2f}")

    # Test 4: Kelly calculation
    print("\n4. Kelly Sizing:")
    kelly = engine.calculate_kelly_with_mc(
        mc_hit_rate=0.56,
        odds=-110
    )
    print(f"   Hit Rate: 56%")
    print(f"   Kelly Fraction: {kelly:.4f} ({kelly*100:.2f}% of bankroll)")

    # Test 5: Multiple props
    print("\n5. Multiple Props Simulation:")
    props_df = pd.DataFrame({
        'player': ['Player A', 'Player B', 'Player C'],
        'market': ['receiving_yards', 'receptions', 'rushing_yards'],
        'prediction': [70, 6.2, 85],
        'line': [65.5, 5.5, 80.5],
        'std_dev': [18, 1.8, 22],
        'distribution': ['normal', 'poisson', 'lognormal'],
        'recommendation': ['OVER', 'OVER', 'OVER']
    })

    sim_results = engine.simulate_multiple_props(props_df)

    print("\n   Results:")
    print(sim_results[['player', 'market', 'prediction', 'line', 'mc_over_rate']])

    # Filter by hit rate
    filtered = engine.filter_by_hit_rate(sim_results, min_hit_rate=0.54)
    print(f"\n   Filtered to {len(filtered)} props with hit rate ≥ 54%")

    print("\n✅ Monte Carlo engine tested successfully!")
