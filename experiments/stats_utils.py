from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller


def verify_assumptions_t_test(group1: Union[List[float], np.ndarray],
                              group2: Union[List[float], np.ndarray],
                              alpha: float = 0.05) -> str:
    """
    Verify assumptions for t-test (Normality and Homogeneity of Variance).
    Returns the recommended test type: 'standard_t', 'welch_t', or 'mann_whitney'.

    Priority 1: Statistical Assumption Verification
    """
    # 1. Normality Test (Shapiro-Wilk)
    # Note: Shapiro-Wilk is sensitive to N. For N > 50, visual inspection is often better,
    # but for automated pipelines, we'll use it with a caveat or fallback.
    try:
        _, p_norm1 = stats.shapiro(group1)
        _, p_norm2 = stats.shapiro(group2)
    except ValueError:
        # Fallback for very small N or constant input
        p_norm1, p_norm2 = 1.0, 1.0

    if p_norm1 < alpha or p_norm2 < alpha:
        print(f"WARNING: Non-normal distribution detected (p1={p_norm1:.4f}, p2={p_norm2:.4f})")
        print("RECOMMENDATION: Use Mann-Whitney U test")
        return "mann_whitney"

    # 2. Homogeneity of Variance (Levene's test)
    _, p_levene = stats.levene(group1, group2)

    if p_levene < alpha:
        print(f"WARNING: Unequal variances detected (p={p_levene:.4f})")
        print("RECOMMENDATION: Use Welch's t-test")
        return "welch_t"

    return "standard_t"

def verify_chi_square_assumptions(contingency_table: np.ndarray) -> bool:
    """
    Verify assumptions for Chi-square test (Expected frequencies >= 5).
    Returns True if assumptions are met, False otherwise.

    Priority 1: Statistical Assumption Verification
    """
    try:
        result = stats.contingency.expected_freq(contingency_table)
        # result is the expected array directly in newer scipy versions?
        # Actually stats.contingency.expected_freq returns the array.
        expected = result
    except Exception:
        # Fallback for older scipy or different signature
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    if np.any(expected < 5):
        count_low = np.sum(expected < 5)
        print(f"WARNING: Expected frequency < 5 detected in {count_low} cells")
        print("RECOMMENDATION: Use Fisher's exact test or increase N")
        return False

    return True

def verify_stationarity(time_series: Union[List[float], np.ndarray, pd.Series],
                       alpha: float = 0.05) -> bool:
    """
    Verify stationarity of a time series using Augmented Dickey-Fuller (ADF) test.
    Returns True if stationary (p < alpha), False otherwise.

    Priority 1: Statistical Assumption Verification
    """
    # Drop NaNs if any
    if isinstance(time_series, pd.Series):
        ts = time_series.dropna()
    else:
        ts = np.array(time_series)
        ts = ts[~np.isnan(ts)]

    # ADF test requires some variation
    if np.std(ts) == 0:
        print("WARNING: Time series is constant. Cannot test stationarity.")
        return True # Technically stationary but useless for Granger

    try:
        result = adfuller(ts)
        p_value = result[1]

        if p_value > alpha:
            print(f"WARNING: Non-stationary series detected (p={p_value:.4f})")
            print("RECOMMENDATION: Difference series or use cointegration")
            return False

        return True
    except Exception as e:
        print(f"WARNING: ADF test failed: {e}")
        return False

def holm_bonferroni_correction(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, bool]:
    """
    Apply Holm-Bonferroni correction to a dictionary of p-values.
    Returns a dictionary mapping experiment names to boolean significance (True/False).

    Priority 2: Multiple Comparison Correction
    """
    sorted_items = sorted(p_values.items(), key=lambda x: x[1])
    sorted_p = [p for name, p in sorted_items]
    sorted_names = [name for name, p in sorted_items]

    n = len(p_values)
    results = {}

    for i, p in enumerate(sorted_p):
        alpha_adjusted = alpha / (n - i)
        is_significant = p < alpha_adjusted
        results[sorted_names[i]] = is_significant

        # If one fails, all subsequent (larger p) also fail in Holm-Bonferroni step-down?
        # Actually Holm-Bonferroni stops rejecting at the first non-rejection.
        if not is_significant:
            # Mark all remaining as False
            for j in range(i + 1, n):
                results[sorted_names[j]] = False
            break

    return results

def check_time_series_length(series: Union[List, np.ndarray], min_obs: int = 50) -> bool:
    """
    Verify sufficient observations for time series analysis.

    Priority 4: Time Series Length Specification
    """
    n = len(series)
    if n < min_obs:
        print(f"ERROR: Insufficient data for analysis: {n} < {min_obs}")
        return False
    return True
