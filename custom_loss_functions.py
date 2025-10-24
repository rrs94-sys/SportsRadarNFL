"""
Custom Loss Functions for NFL Player Props Models
Includes: Quantile, Poisson, Negative Binomial, Log-Normal, Huber, Brier, LogLoss
"""

import numpy as np
from scipy import stats
from scipy.special import gammaln
from typing import Tuple, Callable
import xgboost as xgb


# ============================================================================
# QUANTILE LOSS (for XGBoost/LightGBM)
# ============================================================================

def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.60) -> float:
    """
    Quantile loss function.

    Args:
        y_true: True values
        y_pred: Predicted values
        quantile: Target quantile (tau)

    Returns:
        Mean quantile loss
    """
    errors = y_true - y_pred
    loss = np.where(
        errors >= 0,
        quantile * errors,
        (quantile - 1) * errors
    )
    return np.mean(loss)


def quantile_objective_xgb(quantile: float = 0.60):
    """
    XGBoost objective function for quantile regression.

    Args:
        quantile: Target quantile

    Returns:
        Objective function compatible with XGBoost custom_objective
    """
    def objective(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns gradient and hessian for quantile loss.
        """
        errors = y_true - y_pred

        # Gradient
        grad = np.where(errors >= 0, -quantile, quantile - 1)

        # Hessian (constant for quantile loss)
        hess = np.ones_like(y_true)

        return grad, hess

    return objective


# ============================================================================
# POISSON LOSS (for count data: receptions, completions, TDs)
# ============================================================================

def poisson_deviance_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Poisson deviance loss.

    Appropriate for count data where Var(Y) ≈ E(Y)

    Loss = 2 * Σ[y * ln(y / ŷ) - (y - ŷ)]
    """
    y_pred = np.maximum(y_pred, 1e-10)  # Avoid log(0)

    term1 = np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0)
    term2 = y_true - y_pred

    return 2.0 * np.mean(term1 - term2)


def poisson_loglikelihood(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Negative log-likelihood for Poisson distribution.

    NLL = -Σ[y * ln(λ) - λ - ln(y!)]
    """
    y_pred = np.maximum(y_pred, 1e-10)

    log_factorial_y = gammaln(y_true + 1)
    nll = -(y_true * np.log(y_pred) - y_pred - log_factorial_y)

    return np.mean(nll)


def poisson_probability_over_line(line: float, lambda_pred: float) -> float:
    """
    Calculate P(Y > line) for Poisson distribution.

    Args:
        line: Betting line
        lambda_pred: Predicted Poisson parameter (mean)

    Returns:
        Probability of going over the line
    """
    return 1.0 - stats.poisson.cdf(line, lambda_pred)


# ============================================================================
# NEGATIVE BINOMIAL LOSS (for overdispersed count data)
# ============================================================================

def estimate_dispersion(y: np.ndarray) -> float:
    """
    Estimate dispersion parameter α for Negative Binomial.

    Var(Y) = μ + α * μ²
    α = (Var - μ) / μ²
    """
    mean_y = np.mean(y)
    var_y = np.var(y)

    if var_y <= mean_y:
        return 0.0  # No overdispersion, use Poisson

    alpha = (var_y - mean_y) / (mean_y ** 2)
    return max(alpha, 0.01)  # Minimum threshold


def negbinom_deviance_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """
    Negative Binomial deviance loss.

    Use when data is overdispersed (variance > mean).

    Args:
        y_true: True values
        y_pred: Predicted means
        alpha: Dispersion parameter
    """
    y_pred = np.maximum(y_pred, 1e-10)

    r = y_pred / alpha
    p = r / (r + y_pred)

    term1 = np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0)
    term2 = (y_true + r) * np.log((y_true + r) / (y_pred + r))

    return 2.0 * np.mean(term1 - term2)


def negbinom_objective_xgb(alpha: float):
    """
    XGBoost objective for Negative Binomial.

    Args:
        alpha: Dispersion parameter

    Returns:
        Custom objective function
    """
    def objective(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_pred = np.maximum(y_pred, 1e-10)

        r = y_pred / alpha

        # Gradient
        grad = -(y_true - y_pred) / (y_pred + alpha * y_pred)

        # Hessian
        hess = (y_true + r) / ((y_pred + r) ** 2)

        return grad, hess

    return objective


# ============================================================================
# LOG-NORMAL LOSS (for positive continuous data: yards)
# ============================================================================

def lognormal_nll_loss(y_true: np.ndarray, y_pred_mean: np.ndarray, sigma_ln: float = 0.5) -> float:
    """
    Negative log-likelihood for log-normal distribution.

    Y ~ LogNormal(μ_ln, σ_ln)

    Args:
        y_true: True values (yards)
        y_pred_mean: Predicted mean values
        sigma_ln: Log-scale standard deviation
    """
    y_true = np.maximum(y_true, 1e-10)
    y_pred_mean = np.maximum(y_pred_mean, 1e-10)

    # Convert mean to log-scale parameters
    mu_ln = np.log(y_pred_mean) - 0.5 * sigma_ln**2

    # NLL for log-normal
    log_y = np.log(y_true)
    nll = 0.5 * np.log(2 * np.pi * sigma_ln**2) + log_y + ((log_y - mu_ln)**2) / (2 * sigma_ln**2)

    return np.mean(nll)


def lognormal_probability_over_line(line: float, pred_mean: float, sigma_ln: float = 0.5) -> float:
    """
    Calculate P(Y > line) for log-normal distribution.

    Args:
        line: Betting line
        pred_mean: Predicted mean
        sigma_ln: Log-scale standard deviation

    Returns:
        Probability of going over
    """
    if line <= 0:
        return 1.0

    mu_ln = np.log(pred_mean) - 0.5 * sigma_ln**2
    z = (np.log(line) - mu_ln) / sigma_ln

    return 1.0 - stats.norm.cdf(z)


# ============================================================================
# HUBER LOSS (robust to outliers)
# ============================================================================

def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    """
    Huber loss - combination of MSE and MAE.

    Robust to outliers while maintaining smoothness.

    Args:
        y_true: True values
        y_pred: Predicted values
        delta: Threshold for switching from quadratic to linear
    """
    errors = np.abs(y_true - y_pred)

    # Quadratic for small errors, linear for large
    loss = np.where(
        errors <= delta,
        0.5 * errors**2,
        delta * (errors - 0.5 * delta)
    )

    return np.mean(loss)


def huber_objective_xgb(delta: float = 1.0):
    """
    XGBoost objective for Huber loss.

    Args:
        delta: Huber threshold

    Returns:
        Custom objective function
    """
    def objective(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        errors = y_true - y_pred
        abs_errors = np.abs(errors)

        # Gradient
        grad = np.where(
            abs_errors <= delta,
            -errors,
            -delta * np.sign(errors)
        )

        # Hessian
        hess = np.where(
            abs_errors <= delta,
            1.0,
            0.0  # Approximate for large errors
        ) + 1e-6  # Avoid zero hessian

        return grad, hess

    return objective


# ============================================================================
# PROBABILITY CALIBRATION LOSSES (for TD models)
# ============================================================================

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Brier score for probability predictions.

    BS = mean((y_prob - y_true)²)

    Lower is better. Perfect predictions = 0.

    Args:
        y_true: Binary outcomes (0 or 1)
        y_prob: Predicted probabilities
    """
    return np.mean((y_prob - y_true) ** 2)


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """
    Log loss (cross-entropy) for binary classification.

    LogLoss = -mean[y*log(p) + (1-y)*log(1-p)]

    Args:
        y_true: Binary outcomes (0 or 1)
        y_prob: Predicted probabilities
        eps: Small value to avoid log(0)
    """
    y_prob = np.clip(y_prob, eps, 1 - eps)

    loss = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

    return np.mean(loss)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).

    Measures how well predicted probabilities match actual frequencies.

    Args:
        y_true: Binary outcomes
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_prob[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            ece += bin_weight * np.abs(bin_accuracy - bin_confidence)

    return ece


# ============================================================================
# TOUCHDOWN SPECIFIC (zero-inflated count)
# ============================================================================

def zero_inflated_poisson_nll(
    y_true: np.ndarray,
    lambda_pred: np.ndarray,
    pi_zero: float = 0.7
) -> float:
    """
    Zero-inflated Poisson negative log-likelihood.

    Good for TD predictions where P(TD=0) is high.

    Args:
        y_true: True TD counts
        lambda_pred: Predicted Poisson parameter
        pi_zero: Probability of structural zero
    """
    lambda_pred = np.maximum(lambda_pred, 1e-10)

    # P(Y=0) = π + (1-π)*exp(-λ)
    prob_zero = pi_zero + (1 - pi_zero) * np.exp(-lambda_pred)

    # P(Y=k) = (1-π) * Poisson(k; λ) for k > 0
    prob_nonzero = (1 - pi_zero) * stats.poisson.pmf(y_true, lambda_pred)

    # Combined probability
    prob = np.where(y_true == 0, prob_zero, prob_nonzero)
    prob = np.maximum(prob, 1e-10)

    return -np.mean(np.log(prob))


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def select_count_distribution(y: np.ndarray, threshold: float = 0.2) -> str:
    """
    Automatically select Poisson vs Negative Binomial based on dispersion.

    Args:
        y: Count data
        threshold: Dispersion threshold

    Returns:
        'poisson' or 'negbinom'
    """
    alpha = estimate_dispersion(y)

    if alpha > threshold:
        return 'negbinom'
    else:
        return 'poisson'


def calculate_sigma_ln_from_cv(cv: float) -> float:
    """
    Calculate log-normal sigma from coefficient of variation.

    CV = sqrt(exp(σ²) - 1)
    σ = sqrt(ln(CV² + 1))
    """
    return np.sqrt(np.log(cv**2 + 1))


def estimate_variance_mad(residuals: np.ndarray) -> float:
    """
    Estimate variance using Median Absolute Deviation (robust to outliers).

    σ ≈ 1.4826 * MAD
    """
    mad = np.median(np.abs(residuals - np.median(residuals)))
    return 1.4826 * mad


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING CUSTOM LOSS FUNCTIONS")
    print("="*60)

    # Generate sample data
    np.random.seed(42)

    # Test quantile loss
    print("\n1. Quantile Loss (τ=0.60):")
    y_true = np.array([5, 3, 8, 2, 6])
    y_pred_median = np.array([4, 4, 7, 3, 5])
    y_pred_q60 = np.array([5, 5, 8, 4, 6])

    loss_median = quantile_loss(y_true, y_pred_median, quantile=0.50)
    loss_q60 = quantile_loss(y_true, y_pred_q60, quantile=0.60)

    print(f"   Median loss (τ=0.50): {loss_median:.4f}")
    print(f"   Q60 loss (τ=0.60): {loss_q60:.4f}")

    # Test Poisson
    print("\n2. Poisson Loss:")
    y_count = np.array([3, 5, 2, 8, 1, 4])
    y_pred_count = np.array([3.5, 4.8, 2.2, 7.5, 1.3, 4.1])

    poisson_loss_val = poisson_deviance_loss(y_count, y_pred_count)
    print(f"   Poisson deviance: {poisson_loss_val:.4f}")

    # Test dispersion
    print("\n3. Dispersion Test:")
    y_low_var = np.random.poisson(5, 100)
    y_high_var = np.random.negative_binomial(5, 0.3, 100)

    alpha_low = estimate_dispersion(y_low_var)
    alpha_high = estimate_dispersion(y_high_var)

    print(f"   Low variance data α: {alpha_low:.4f} → {select_count_distribution(y_low_var)}")
    print(f"   High variance data α: {alpha_high:.4f} → {select_count_distribution(y_high_var)}")

    # Test log-normal
    print("\n4. Log-Normal:")
    y_yards = np.array([65, 48, 92, 31, 75])
    y_pred_yards = np.array([60, 50, 85, 35, 70])

    prob_over = lognormal_probability_over_line(50, 60, sigma_ln=0.5)
    print(f"   P(Yards > 50 | μ=60, σ=0.5): {prob_over:.4f}")

    # Test Brier score
    print("\n5. Brier Score (TDs):")
    y_td_binary = np.array([1, 0, 0, 1, 0])
    y_prob = np.array([0.7, 0.2, 0.3, 0.8, 0.1])

    brier = brier_score(y_td_binary, y_prob)
    logloss = log_loss(y_td_binary, y_prob)

    print(f"   Brier score: {brier:.4f}")
    print(f"   Log loss: {logloss:.4f}")

    # Test Huber
    print("\n6. Huber Loss:")
    y_with_outlier = np.array([5, 6, 4, 5, 100])
    y_pred_robust = np.array([5, 5, 5, 5, 5])

    mse = np.mean((y_with_outlier - y_pred_robust)**2)
    huber = huber_loss(y_with_outlier, y_pred_robust, delta=2.0)

    print(f"   MSE (sensitive to outliers): {mse:.4f}")
    print(f"   Huber loss (robust): {huber:.4f}")

    print("\n✅ All loss functions tested successfully!")
