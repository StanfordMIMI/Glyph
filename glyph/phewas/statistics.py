"""
Statistical utilities for PheWAS: odds ratios, prevalence, etc.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
from typing import Dict, Tuple, Optional


def compute_feature_prevalence(feature_values: np.ndarray) -> int:
    """
    Compute feature prevalence (count of samples where feature > 0).
    
    Args:
        feature_values: Array of feature activations
        
    Returns:
        Count of samples where feature > 0
    """
    return int(np.sum(feature_values > 0))


def compute_outcome_prevalence(outcome_values: np.ndarray) -> float:
    """
    Compute outcome prevalence (proportion of positive outcomes).
    
    Args:
        outcome_values: Array of binary outcome values
        
    Returns:
        Proportion of samples with outcome == 1
    """
    observed = np.asarray(outcome_values)[np.isin(outcome_values, [0, 1])]
    return float(np.mean(observed)) if len(observed) else np.nan


def compute_odds_ratio(
    feature_values: np.ndarray,
    outcome_values: np.ndarray,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute odds ratio between a single feature and outcome using logistic regression.
    
    The odds ratio is exp(coefficient) from a logistic regression of outcome ~ feature.
    
    Args:
        feature_values: Array of feature activations (n_samples,)
        outcome_values: Array of binary outcome values (n_samples,)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with:
            - odds_ratio: exp(coefficient)
            - odds_ratio_ci_lower: Lower 95% CI
            - odds_ratio_ci_upper: Upper 95% CI
            - p_value: Two-tailed p-value
            - auc: ROC AUC score
    """
    # Reshape for sklearn (features should already be normalized by caller)
    X = feature_values.reshape(-1, 1)
    y = outcome_values
    
    # Fit logistic regression (no penalty for unbiased coefficients)
    model = LogisticRegression(max_iter=1000, random_state=seed, penalty=None)
    model.fit(X, y)
    
    # Extract coefficient
    coef = model.coef_[0][0]
    
    # Predicted probabilities
    pred_probs = model.predict_proba(X)[:, 1]
    
    # Compute AUC
    auc = roc_auc_score(y, pred_probs)
    
    # Compute standard error using Fisher Information
    w = pred_probs * (1 - pred_probs)
    X_design = np.column_stack([np.ones(X.shape[0]), X])
    fisher_info = X_design.T @ (X_design * w[:, None])
    
    cov_matrix = np.linalg.pinv(fisher_info)
    var_coef = cov_matrix[1, 1]
    se = np.sqrt(max(var_coef, 1e-10))  # Ensure non-negative
    
    # Z-score and p-value
    z_score = coef / se
    p_value = 2 * norm.sf(np.abs(z_score))
    
    # 95% confidence intervals
    z_critical = 1.96
    coef_ci_lower = coef - z_critical * se
    coef_ci_upper = coef + z_critical * se
    
    return {
        'odds_ratio': float(np.exp(coef)),
        'odds_ratio_ci_lower': float(np.exp(coef_ci_lower)),
        'odds_ratio_ci_upper': float(np.exp(coef_ci_upper)),
        'p_value': float(p_value),
        'auc': float(auc),
        'coefficient': float(coef),
        'standard_error': float(se),
    }


def compute_conditional_probability(
    feature_values: np.ndarray,
    outcome_values: np.ndarray,
) -> Dict[str, float]:
    """
    Compute P(outcome | feature > 0).
    
    Args:
        feature_values: Array of feature activations
        outcome_values: Array of binary outcome values
        
    Returns:
        Dictionary with p_y_given_f, c_f, c_yf
    """
    # Feature activates
    f_active = (feature_values > 0) & np.isin(outcome_values, [0, 1])
    c_f = int(f_active.sum())
    
    # Outcome positive when feature activates
    c_yf = int(((outcome_values > 0) & f_active).sum())
    
    # P(y | f > 0)
    p_y_given_f = c_yf / c_f if c_f > 0 else 0.0
    
    return {
        'c_f': c_f,
        'c_yf': c_yf,
        'p_y_given_f': float(p_y_given_f),
    }
