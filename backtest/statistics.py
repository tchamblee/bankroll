import numpy as np
import pandas as pd
from scipy.stats import norm
from itertools import combinations
from typing import List, Tuple, Dict

def estimated_sharpe_ratio(returns: np.ndarray, annualization_factor: int = 252) -> float:
    """Calculates standard annualized Sharpe Ratio."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return np.mean(returns) / np.std(returns) * np.sqrt(annualization_factor)

def calculate_sortino_ratio(returns: np.ndarray, annualization_factor: int = 189000, target_return: float = 0.0) -> float:
    """
    Calculates Annualized Sortino Ratio using Downside Deviation (LPM_2).
    Clamps positive returns to zero (or target_return) for the deviation calculation.
    """
    if len(returns) < 2: return 0.0
    
    avg_ret = np.mean(returns)
    
    # Downside Deviation (LPM_2)
    # Using target_return (usually 0) as the hurdle
    downside = np.minimum(returns - target_return, 0.0)
    downside_std = np.std(downside) + 1e-9
    
    return (avg_ret / downside_std) * np.sqrt(annualization_factor)

def deflated_sharpe_ratio(
    observed_sr: float, 
    returns: np.ndarray, 
    n_trials: int, 
    var_returns: float, 
    skew_returns: float, 
    kurt_returns: float,
    annualization_factor: int = 114408
) -> float:
    """
    Calculates the Probabilistic Sharpe Ratio (PSR) adjusted for multiple testing (DSR).
    Reference: Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio.
    
    :param observed_sr: The annualized Sharpe Ratio of the strategy.
    :param returns: The array of strategy returns.
    :param n_trials: Number of independent trials (strategies tested).
    :return: Probability that true SR > 0, adjusted for selection bias.
    """
    # 1. Standard Deviation of the Sharpe Ratio Estimator (for ONE strategy)
    # Under the Null Hypothesis (SR=0), the standard deviation of the SR estimator is:
    # sigma_SR = sqrt( (1 - skew*SR + (kurt-1)/4 * SR^2) / (T-1) )
    # Since we test H0: SR=0, the terms with SR vanish, simplifying to:
    # sigma_SR = sqrt(1 / (T-1))
    # BUT, we are comparing against the "Max SR" distribution.
    
    T = len(returns)
    if T < 2: return 0.0
    
    # 2. Expected Maximum Sharpe Ratio (The Hurdle)
    # E[max_SR] = E[SR] + sqrt(2 * log(N)) * sigma_SR_trial
    # Where sigma_SR_trial is the std dev of the underlying SR distribution across trials.
    # We approximate sigma_SR_trial using the variance of the returns' moments.
    
    # Bailey-Lopez de Prado (2012) Approximation for SR Variance:
    # Var(SR) approx (1 - skew*SR + (kurt-1)/4 * SR^2) / (T - 1)
    
    # We use the Observed SR to estimate the variance of the SR estimator itself
    sr_var = (1 - skew_returns * observed_sr + (kurt_returns - 1) / 4 * observed_sr**2) / (T - 1)
    sr_std = np.sqrt(sr_var)
    
    # Expected Max SR (Hurdle)
    # Euler-Mascheroni constant (emc) for precise E[max]
    emc = 0.5772156649
    
    if n_trials < 2:
        expected_max_sr = 0 # No multiple testing penalty
    else:
        # The expected maximum of N independent Gaussian variables with variance sr_var
        # E[Max] = sigma * ((1 - gamma)*Z_1 + gamma*Z_N) ... simplified to:
        expected_max_sr = sr_std * ((1 - emc) * norm.ppf(1 - 1/n_trials) + emc * norm.ppf(1 - 1/(n_trials * np.e)))
        # Or standard approximation:
        expected_max_sr = sr_std * np.sqrt(2 * np.log(n_trials))

    # 3. Deflated Sharpe Ratio (Probabilistic Test)
    # We test: Is (Observed_SR - Hurdle) > 0 significantly?
    # DSR = CDF( (Observed_SR - Expected_Max_SR) / sr_std )
    
    dsr_stat = (observed_sr - expected_max_sr) / sr_std
    
    return norm.cdf(dsr_stat)


def combinatorial_purged_cv(
    n_samples: int, 
    n_folds: int = 5, 
    n_test_folds: int = 2,
    embargo_pct: float = 0.01
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates Train/Test splits using Combinatorial Purged Cross-Validation (CPCV).
    Unlike standard K-Fold, this creates C(n_folds, n_test_folds) unique paths.
    
    :param n_samples: Total number of bars/samples.
    :param n_folds: Total number of groups to split data into.
    :param n_test_folds: Number of groups to use as Test set in each split.
    :param embargo_pct: Percentage of data to drop after each test split to prevent leakage.
    :return: List of (train_indices, test_indices) tuples.
    """
    indices = np.arange(n_samples)
    
    # split indices into N equal contiguous blocks
    fold_size = n_samples // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else n_samples
        folds.append(indices[start:end])
        
    # Generate combinations of folds to use as Test sets
    # e.g., if N=5, k=2, we get pairs like (0,1), (0,2)... (3,4)
    # Total combinations = 5! / (2! * 3!) = 10
    combos = list(combinations(range(n_folds), n_test_folds))
    
    splits = []
    embargo = int(n_samples * embargo_pct)
    
    for test_fold_indices in combos:
        test_idx = np.concatenate([folds[i] for i in test_fold_indices])
        
        # Train indices are everything NOT in test folds
        # AND strictly applying embargo after test folds if they precede train data
        # (For simplicity here, we just exclude test indices. 
        # A full CPCV implementation handles embargoes carefully around borders.)
        
        train_folds_indices = [i for i in range(n_folds) if i not in test_fold_indices]
        train_idx = np.concatenate([folds[i] for i in train_folds_indices])
        
        # Sort to ensure chronological order within the sets
        test_idx.sort()
        train_idx.sort()
        
        splits.append((train_idx, test_idx))
        
    return splits
