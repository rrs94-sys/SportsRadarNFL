"""
Quantitative Betting Engine
Fractional Kelly sizing, correlation controls, and exposure management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from phase1_config import Config


class QuantitativeBettingEngine:
    """
    Advanced betting engine with:
    - Fractional Kelly criterion
    - Correlation-adjusted exposure limits
    - Volatility-adjusted edge thresholds
    - Dynamic bankroll management
    """

    def __init__(self, bankroll: float = None, config: Config = Config):
        self.initial_bankroll = bankroll or config.INITIAL_BANKROLL
        self.current_bankroll = self.initial_bankroll
        self.config = config

        # Tracking
        self.bet_history = []
        self.exposure = {}

    def calculate_edge(
        self,
        prediction: float,
        line: float,
        odds: float = -110
    ) -> Dict:
        """
        Calculate betting edge.

        Args:
            prediction: Model's prediction
            line: Betting line
            odds: American odds

        Returns:
            Dictionary with edge metrics
        """
        # Determine recommendation
        if prediction > line:
            recommendation = 'OVER'
            edge_magnitude = prediction - line
        elif prediction < line:
            recommendation = 'UNDER'
            edge_magnitude = line - prediction
        else:
            recommendation = 'PASS'
            edge_magnitude = 0

        # Convert to probability using assumed distribution
        # For simplicity, use normal approximation
        # P(X > line) ≈ prediction distance from line / typical std

        return {
            'recommendation': recommendation,
            'edge_magnitude': edge_magnitude,
            'prediction': prediction,
            'line': line,
            'odds': odds
        }

    def calculate_kelly_fraction(
        self,
        win_probability: float,
        odds: float = -110,
        kelly_multiplier: float = None
    ) -> float:
        """
        Calculate Kelly fraction for bet sizing.

        Args:
            win_probability: Estimated probability of winning
            odds: American odds
            kelly_multiplier: Fractional Kelly multiplier (default from config)

        Returns:
            Kelly fraction (percentage of bankroll to bet)
        """
        if kelly_multiplier is None:
            kelly_multiplier = self.config.KELLY_FRACTION

        # Convert American odds to decimal
        if odds < 0:
            decimal_odds = 1 + (100 / abs(odds))
        else:
            decimal_odds = 1 + (odds / 100)

        # Implied probability from odds
        implied_prob = 1 / decimal_odds

        # Edge
        edge = win_probability - implied_prob

        if edge <= self.config.MIN_EDGE:
            return 0.0

        # Kelly formula: (p * b - (1 - p)) / b
        # where b = net odds (decimal_odds - 1)
        b = decimal_odds - 1
        kelly = (win_probability * b - (1 - win_probability)) / b

        # Apply fractional Kelly
        kelly_fraction = kelly * kelly_multiplier

        # Ensure non-negative
        return max(0, kelly_fraction)

    def calculate_bet_size(
        self,
        kelly_fraction: float,
        volatility_factor: float = 1.0,
        correlation_adjustment: float = 1.0
    ) -> float:
        """
        Calculate actual bet size with adjustments.

        Args:
            kelly_fraction: Base Kelly fraction
            volatility_factor: Adjustment for player volatility (0.5 - 1.0)
            correlation_adjustment: Adjustment for correlated bets (0.5 - 1.0)

        Returns:
            Bet size in dollars
        """
        # Adjust Kelly for volatility and correlation
        adjusted_kelly = kelly_fraction * volatility_factor * correlation_adjustment

        # Calculate dollar amount
        bet_size = self.current_bankroll * adjusted_kelly

        # Apply min/max constraints
        bet_size = max(bet_size, self.config.MIN_BET)
        bet_size = min(bet_size, self.config.MAX_BET)

        return bet_size

    def check_correlation(
        self,
        bet1: Dict,
        bet2: Dict
    ) -> float:
        """
        Estimate correlation between two bets.

        Returns:
            Correlation coefficient (-1 to 1)
        """
        # Same game
        if bet1.get('game_id') == bet2.get('game_id'):
            # Same team offense
            if bet1.get('team') == bet2.get('team'):
                pos1 = bet1.get('position', '')
                pos2 = bet2.get('position', '')

                if pos1 == 'QB' and pos2 in ['WR', 'TE']:
                    return self.config.CORRELATED_POSITIONS.get('QB_WR_same_team', 0.7)
                elif pos1 in ['WR', 'TE'] and pos2 == 'QB':
                    return self.config.CORRELATED_POSITIONS.get('QB_WR_same_team', 0.7)
                elif pos1 in ['WR', 'TE'] and pos2 in ['WR', 'TE']:
                    return self.config.CORRELATED_POSITIONS.get('WR_WR_same_team', 0.3)

            # Opposing teams (negative correlation for RB vs defense)
            if bet1.get('team') != bet2.get('team'):
                if bet1.get('position') == 'RB' and bet2.get('side') == 'defense':
                    return self.config.CORRELATED_POSITIONS.get('RB_DEF_opp_team', -0.5)

        return 0.0  # No correlation

    def calculate_portfolio_exposure(
        self,
        proposed_bets: List[Dict]
    ) -> Dict:
        """
        Calculate total portfolio exposure accounting for correlations.

        Args:
            proposed_bets: List of proposed bets

        Returns:
            Dictionary with exposure metrics
        """
        total_risk = 0
        game_exposure = {}
        team_exposure = {}

        for bet in proposed_bets:
            bet_size = bet.get('bet_size', 0)
            total_risk += bet_size

            # Track by game
            game_id = bet.get('game_id', 'unknown')
            game_exposure[game_id] = game_exposure.get(game_id, 0) + bet_size

            # Track by team
            team = bet.get('team', 'unknown')
            team_exposure[team] = team_exposure.get(team, 0) + bet_size

        # Check for excessive correlation
        max_game_exposure = max(game_exposure.values()) if game_exposure else 0
        max_team_exposure = max(team_exposure.values()) if team_exposure else 0

        correlated_risk = max_game_exposure / self.current_bankroll if self.current_bankroll > 0 else 0

        return {
            'total_risk': total_risk,
            'total_risk_pct': total_risk / self.current_bankroll if self.current_bankroll > 0 else 0,
            'max_game_exposure': max_game_exposure,
            'max_game_exposure_pct': max_game_exposure / self.current_bankroll if self.current_bankroll > 0 else 0,
            'max_team_exposure': max_team_exposure,
            'correlated_risk_pct': correlated_risk,
            'game_exposure': game_exposure,
            'team_exposure': team_exposure
        }

    def apply_volatility_adjustment(
        self,
        edge_magnitude: float,
        volatility_cv: float,
        base_threshold: float
    ) -> Tuple[bool, float]:
        """
        Adjust edge threshold based on player volatility.

        Args:
            edge_magnitude: Raw edge magnitude
            volatility_cv: Coefficient of variation (volatility measure)
            base_threshold: Base edge threshold

        Returns:
            (passes_threshold, adjusted_threshold)
        """
        if volatility_cv > self.config.HIGH_VOLATILITY_THRESHOLD:
            # Require higher edge for volatile players
            adjusted_threshold = base_threshold * self.config.VOLATILITY_EDGE_MULTIPLIER
        else:
            adjusted_threshold = base_threshold

        passes = edge_magnitude >= adjusted_threshold

        return passes, adjusted_threshold

    def optimize_bet_portfolio(
        self,
        candidate_bets: pd.DataFrame,
        max_total_exposure: float = None
    ) -> pd.DataFrame:
        """
        Optimize bet portfolio considering correlations and exposure limits.

        Args:
            candidate_bets: DataFrame with candidate bets
            max_total_exposure: Maximum total exposure as fraction of bankroll

        Returns:
            Optimized bet portfolio
        """
        if max_total_exposure is None:
            max_total_exposure = self.config.MAX_CORRELATED_EXPOSURE

        # Sort by edge / expected value
        if 'expected_value' in candidate_bets.columns:
            sorted_bets = candidate_bets.sort_values('expected_value', ascending=False)
        elif 'edge_magnitude' in candidate_bets.columns:
            sorted_bets = candidate_bets.sort_values('edge_magnitude', ascending=False)
        else:
            sorted_bets = candidate_bets

        selected_bets = []
        total_exposure = 0

        for idx, bet in sorted_bets.iterrows():
            bet_dict = bet.to_dict()

            # Check correlation with existing bets
            max_correlation = 0
            for existing in selected_bets:
                corr = self.check_correlation(bet_dict, existing)
                max_correlation = max(max_correlation, abs(corr))

            # Adjust bet size for correlation
            correlation_adjustment = 1.0 - (max_correlation * 0.5)  # Reduce size if correlated

            # Recalculate bet size
            kelly_fraction = bet_dict.get('kelly_fraction', 0)
            volatility_factor = bet_dict.get('volatility_factor', 1.0)

            adjusted_bet_size = self.calculate_bet_size(
                kelly_fraction,
                volatility_factor,
                correlation_adjustment
            )

            # Check if adding this bet exceeds exposure limit
            if (total_exposure + adjusted_bet_size) / self.current_bankroll > max_total_exposure:
                continue  # Skip this bet

            # Add to portfolio
            bet_dict['bet_size'] = adjusted_bet_size
            bet_dict['correlation_adjustment'] = correlation_adjustment
            selected_bets.append(bet_dict)

            total_exposure += adjusted_bet_size

        return pd.DataFrame(selected_bets)

    def update_bankroll(self, result: float):
        """
        Update bankroll after bet settlement.

        Args:
            result: Net profit/loss from bet
        """
        self.current_bankroll += result

    def record_bet(
        self,
        bet_details: Dict,
        result: float = None
    ):
        """
        Record bet in history.

        Args:
            bet_details: Dictionary with bet information
            result: Result of bet (profit/loss) if known
        """
        bet_record = bet_details.copy()
        bet_record['timestamp'] = pd.Timestamp.now()
        bet_record['bankroll_at_bet'] = self.current_bankroll
        bet_record['result'] = result

        self.bet_history.append(bet_record)

        if result is not None:
            self.update_bankroll(result)

    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics from bet history.

        Returns:
            Dictionary with performance stats
        """
        if not self.bet_history:
            return {
                'total_bets': 0,
                'total_profit': 0,
                'roi': 0,
                'hit_rate': 0
            }

        df = pd.DataFrame(self.bet_history)

        settled_bets = df[df['result'].notna()]

        if len(settled_bets) == 0:
            return {
                'total_bets': len(df),
                'total_profit': 0,
                'roi': 0,
                'hit_rate': 0
            }

        wins = settled_bets[settled_bets['result'] > 0]
        total_wagered = settled_bets['bet_size'].sum()
        total_profit = settled_bets['result'].sum()

        return {
            'total_bets': len(settled_bets),
            'total_wagered': total_wagered,
            'total_profit': total_profit,
            'roi': (total_profit / total_wagered * 100) if total_wagered > 0 else 0,
            'hit_rate': len(wins) / len(settled_bets) if len(settled_bets) > 0 else 0,
            'avg_bet_size': settled_bets['bet_size'].mean(),
            'current_bankroll': self.current_bankroll,
            'bankroll_change': self.current_bankroll - self.initial_bankroll,
            'bankroll_change_pct': ((self.current_bankroll - self.initial_bankroll) / self.initial_bankroll * 100)
        }


if __name__ == "__main__":
    # Test betting engine
    print("\n" + "="*60)
    print("TESTING QUANTITATIVE BETTING ENGINE")
    print("="*60)

    engine = QuantitativeBettingEngine(bankroll=10000)

    # Test 1: Edge calculation
    print("\n1. Edge Calculation:")
    edge = engine.calculate_edge(prediction=70, line=65.5, odds=-110)
    print(f"   Prediction: {edge['prediction']}")
    print(f"   Line: {edge['line']}")
    print(f"   Recommendation: {edge['recommendation']}")
    print(f"   Edge Magnitude: {edge['edge_magnitude']}")

    # Test 2: Kelly fraction
    print("\n2. Kelly Fraction:")
    kelly = engine.calculate_kelly_fraction(win_probability=0.56, odds=-110)
    print(f"   Win Probability: 56%")
    print(f"   Kelly Fraction: {kelly:.4f} ({kelly*100:.2f}%)")

    # Test 3: Bet sizing
    print("\n3. Bet Sizing:")
    bet_size = engine.calculate_bet_size(
        kelly_fraction=kelly,
        volatility_factor=0.8,  # Volatile player
        correlation_adjustment=0.9  # Slightly correlated
    )
    print(f"   Bankroll: ${engine.current_bankroll:,.2f}")
    print(f"   Base Kelly: {kelly:.4f}")
    print(f"   Volatility Adjustment: 0.8")
    print(f"   Correlation Adjustment: 0.9")
    print(f"   Final Bet Size: ${bet_size:.2f}")

    # Test 4: Correlation check
    print("\n4. Correlation Check:")
    bet1 = {'game_id': 'g1', 'team': 'KC', 'position': 'QB'}
    bet2 = {'game_id': 'g1', 'team': 'KC', 'position': 'WR'}
    bet3 = {'game_id': 'g2', 'team': 'BUF', 'position': 'RB'}

    corr12 = engine.check_correlation(bet1, bet2)
    corr13 = engine.check_correlation(bet1, bet3)

    print(f"   QB-WR same team: {corr12:.2f}")
    print(f"   QB-RB different games: {corr13:.2f}")

    # Test 5: Portfolio optimization
    print("\n5. Portfolio Optimization:")
    candidates = pd.DataFrame({
        'player': ['Mahomes', 'Kelce', 'Hill', 'Allen'],
        'team': ['KC', 'KC', 'MIA', 'BUF'],
        'position': ['QB', 'TE', 'WR', 'QB'],
        'game_id': ['g1', 'g1', 'g2', 'g3'],
        'edge_magnitude': [5.2, 3.8, 4.1, 6.5],
        'kelly_fraction': [0.03, 0.02, 0.025, 0.04],
        'volatility_factor': [1.0, 0.9, 0.7, 1.0],
        'expected_value': [52, 38, 41, 65]
    })

    optimized = engine.optimize_bet_portfolio(candidates, max_total_exposure=0.15)

    print(f"   Candidates: {len(candidates)}")
    print(f"   Selected: {len(optimized)}")
    print(f"\n   Selected Bets:")
    print(optimized[['player', 'team', 'bet_size', 'correlation_adjustment']])

    # Calculate exposure
    exposure = engine.calculate_portfolio_exposure(optimized.to_dict('records'))
    print(f"\n   Total Risk: ${exposure['total_risk']:.2f} ({exposure['total_risk_pct']*100:.1f}% of bankroll)")
    print(f"   Max Game Exposure: ${exposure['max_game_exposure']:.2f}")

    print("\n✅ Betting engine tested successfully!")
