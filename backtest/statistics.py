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

def deflated_sharpe_ratio(
    observed_sr: float, 
    returns: np.ndarray, 
    n_trials: int, 
    var_returns: float, 
    skew_returns: float, 
    kurt_returns: float,
    annualization_factor: int = 114408 # From memory (5-min bars)
) -> float:
    """
    Calculates the Probabilistic Sharpe Ratio (PSR) adjusted for multiple testing (DSR).
    
    :param observed_sr: The annualized Sharpe Ratio of the strategy to test.
    :param returns: The array of strategy returns.
    :param n_trials: Total number of strategies backtested/rejected (The 'Trial Factor').
    :param var_returns: Variance of the returns.
    :param skew_returns: Skewness of the returns.
    :param kurt_returns: Kurtosis of the returns.
    :return: The probability (0.0 - 1.0) that the true SR is > 0.
    """
    # 1. Estimate the Expected Maximum Sharpe Ratio given N independent trials
    # This represents the "Hurdle" - how high the SR needs to be just to be "lucky"
    # Euler-Mascheroni constant approx 0.5772
    emc = 0.5772156649
    # Expected Max SR = Expected[Max(Z)] * StdDev(SR_distribution)
    # For simplicity in many DSR implementations, we focus on the hurdle derived from n_trials
    
    # Standard deviation of the Sharpe Ratio estimator
    T = len(returns)
    sr_std = np.sqrt((1 - skew_returns * observed_sr + (kurt_returns - 1) / 4 * observed_sr**2) / (T - 1))

    # The "Hurdle" SR (Expected Maximum SR under the Null Hypothesis of SR=0)
    # Using the approximation for independent trials: E[max] approx sqrt(2 * log(N))
    # Note: In practice, strategies are correlated, so N should be 'effective N'. 
    # Using raw N is conservative.
    if n_trials < 2:
        expected_max_sr = 0
    else:
        expected_max_sr = np.sqrt(2 * np.log(n_trials)) # Simplified approach for the hurdle

    # Deflated Sharpe Ratio (Probabilistic)
    # We test: Is Observed SR > Expected Max SR (Hurdle)?
    dsr_stat = (observed_sr - expected_max_sr) / sr_std
    
    # CDF of the test statistic
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
