"""
PheWAS (Phenome-Wide Association Study) implementation for hypothesis generation.

Systematically tests associations between SAE features and outcomes,
computing odds ratios and ranking hypotheses.
"""

import json
import numpy as np
import warnings
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from glyph.phewas.statistics import (
    compute_feature_prevalence,
    compute_outcome_prevalence,
    compute_odds_ratio,
    compute_conditional_probability,
)


class PheWAS:
    """
    PheWAS study for discovering feature-outcome associations.
    
    For each SAE feature and outcome, computes:
    - Feature prevalence (c_f): count where feature > 0
    - Outcome prevalence (p_y): proportion of positive outcomes
    - Odds ratio: from logistic regression
    - AUC: predictive power of single feature
    
    Features with c_f < min_activations are skipped.
    Results are ranked by odds ratio on train set.
    
    Args:
        cfg: Hydra config with phewas settings
        
    Example:
        >>> phewas = PheWAS(cfg)
        >>> results = phewas.run(sae_features, outcomes, sample_ids)
        >>> print(results.head())
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.min_activations = cfg.phewas.min_activations
        self.seed = cfg.seed
    
    def normalize_features(
        self,
        features: Dict[str, torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        """
        Normalize SAE features using train set statistics.
        
        Subtracts train mean and divides by train std for each feature.
        
        Args:
            features: Dict mapping split names to feature tensors
            
        Returns:
            Dict mapping split names to normalized feature arrays
        """
        # Convert to numpy
        features_np = {
            split: f.cpu().numpy() if isinstance(f, torch.Tensor) else f
            for split, f in features.items()
        }
        
        # Compute train statistics
        train = np.where(np.isfinite(features_np['train']), features_np['train'], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            train_mean = np.nanmean(train, axis=0)
            train_std = np.nanstd(train, axis=0)
        train_mean = np.nan_to_num(train_mean)
        
        # Avoid division by zero
        train_std = np.where(train_std > 1e-8, train_std, 1.0)
        
        # Normalize all splits using train statistics
        normalized = {
            split: (f - train_mean) / train_std
            for split, f in features_np.items()
        }
        
        return normalized
    
    def run(
        self,
        sae_features: Dict[str, torch.Tensor],
        outcomes: Dict[str, Dict[str, np.ndarray]],
        modality: str,
        top_k: int,
        alive_mask: Optional[torch.Tensor] = None,
        verbose: bool = True,
        *,
        confounders=None,
        adjustment=None,
        confounder_cols=None,
    ) -> pd.DataFrame:
        """
        Run PheWAS study for a single modality/top_k configuration.
        
        Args:
            sae_features: Dict mapping split names to SAE feature tensors
            outcomes: Dict mapping split names to dicts of outcome arrays
            modality: Name of the modality (for labeling)
            top_k: Top-k value used for SAE (for labeling)
            alive_mask: Optional mask of alive features to use
            verbose: Whether to show progress bar
            
        Returns:
            DataFrame with hypothesis statistics, one row per (feature, outcome, split)
        """
        if (adjustment is None) != (confounders is None):
            raise ValueError('Provide both an adjustment estimator and aligned confounders')
        if adjustment is not None:
            for split, values in sae_features.items():
                if split not in confounders or len(confounders[split]) != len(values):
                    raise ValueError(f'Confounders are not aligned for split {split}')
        for split, split_outcomes in outcomes.items():
            for name, values in split_outcomes.items():
                values = np.asarray(values)
                observed = values[np.isfinite(values) & (values != -1)]
                if not np.isin(observed, [0, 1]).all():
                    raise ValueError(f'PheWAS requires binary 0/1 outcomes; check {name} in {split}')

        # Keep raw features for c_f computation (sparse activations)
        features_raw = {
            split: f.cpu().numpy() if isinstance(f, torch.Tensor) else f
            for split, f in sae_features.items()
        }
        
        # Normalize features for odds ratio computation
        features_norm = self.normalize_features(sae_features)
        
        # Get feature indices to analyze
        n_features = features_norm['train'].shape[1]
        if alive_mask is not None:
            if isinstance(alive_mask, torch.Tensor):
                alive_mask = alive_mask.cpu().numpy()
            feature_indices = np.where(alive_mask)[0]
        else:
            feature_indices = np.arange(n_features)
        
        # Get outcome columns
        outcome_cols = list(outcomes['train'].keys())
        splits = list(features_norm.keys())
        
        # Results list
        results = []
        
        # Progress bar
        total_iterations = len(feature_indices) * len(outcome_cols)
        pbar = tqdm(total=total_iterations, desc="PheWAS", disable=not verbose)
        
        for outcome_col in outcome_cols:
            for feature_idx in feature_indices:
                # First check train c_f on RAW features (sparse activations)
                train_cf = compute_feature_prevalence(features_raw['train'][:, feature_idx])
                
                if train_cf < self.min_activations:
                    pbar.update(1)
                    continue
                
                # Compute statistics for each split
                for split in splits:
                    feature_values_norm = features_norm[split][:, feature_idx]
                    feature_values_raw = features_raw[split][:, feature_idx]
                    outcome_values = outcomes[split][outcome_col]
                    
                    # Feature statistics - use RAW features for c_f count
                    c_f = compute_feature_prevalence(feature_values_raw)
                    
                    # Skip if insufficient activations
                    if c_f < self.min_activations:
                        continue
                    
                    # Outcome statistics
                    p_y = compute_outcome_prevalence(outcome_values)
                    
                    cond_prob = compute_conditional_probability(feature_values_raw, outcome_values)
                    
                    # Odds ratio and AUC (only if both classes present)
                    # Use NORMALIZED features for numerical stability in logistic regression
                    valid_mask = np.isin(outcome_values, [0, 1]) & np.isfinite(feature_values_norm)
                    ov_valid = outcome_values[valid_mask]
                    fv_valid = feature_values_norm[valid_mask]
                    if len(np.unique(ov_valid)) >= 2:
                        or_stats = compute_odds_ratio(
                            fv_valid,
                            ov_valid,
                            seed=self.seed
                        )
                    else:
                        or_stats = {
                            'odds_ratio': np.nan,
                            'odds_ratio_ci_lower': np.nan,
                            'odds_ratio_ci_upper': np.nan,
                            'p_value': np.nan,
                            'auc': np.nan,
                            'coefficient': np.nan,
                            'standard_error': np.nan,
                        }
                    
                    # Compile row
                    row = {
                        'feature_idx': feature_idx,
                        'feature_name': f'feature_{feature_idx}',
                        'outcome': outcome_col,
                        'split': split,
                        'modality': modality,
                        'top_k': top_k,
                        'c_f': c_f,
                        'p_y': p_y,
                        'c_yf': cond_prob['c_yf'],
                        'p_y_given_f': cond_prob['p_y_given_f'],
                        'odds_ratio': or_stats['odds_ratio'],
                        'odds_ratio_ci_lower': or_stats['odds_ratio_ci_lower'],
                        'odds_ratio_ci_upper': or_stats['odds_ratio_ci_upper'],
                        'odds_ratio_p_value': or_stats['p_value'],
                        'auc': or_stats['auc'],
                    }
                    if adjustment is not None:
                        row.update(adjustment.estimate(
                            feature_values_norm, outcome_values, confounders[split]))
                        row['adjustment_covariates'] = json.dumps(list(confounder_cols or []))
                    results.append(row)
                
                pbar.update(1)
        
        pbar.close()
        
        if not results:
            print("  Warning: No features met minimum activation threshold")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Compute ranks based on train odds ratio
        df = self._compute_ranks(df)
        
        return df
    
    def _compute_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute train odds ratio rank for each hypothesis.
        
        Ranks are computed within each (outcome, modality, top_k) group,
        then copied to all splits for that feature.
        """
        if df.empty:
            return df
        
        # Get train rows
        train_df = df[df['split'] == 'train'].copy()
        
        if train_df.empty:
            df['train_odds_ratio_rank'] = np.nan
            return df
        
        # Compute rank within each (outcome, modality, top_k) group
        train_df['train_odds_ratio_rank'] = train_df.groupby(
            ['outcome', 'modality', 'top_k']
        )['odds_ratio'].rank(ascending=False, method='dense')
        
        # Create mapping from (feature_idx, outcome, modality, top_k) to rank
        rank_map = train_df.set_index(
            ['feature_idx', 'outcome', 'modality', 'top_k']
        )['train_odds_ratio_rank'].to_dict()
        
        # Apply ranks to all rows
        df['train_odds_ratio_rank'] = df.apply(
            lambda row: rank_map.get(
                (row['feature_idx'], row['outcome'], row['modality'], row['top_k']),
                np.nan
            ),
            axis=1
        )
        
        return df

    @staticmethod
    def _add_multiple_testing_corrections(df: pd.DataFrame) -> pd.DataFrame:
        """Add global and within-modality Bonferroni-adjusted p-values.

        The discovery family is defined by the hypotheses evaluated on the
        training split.  Its test counts are then reused for every split so a
        train/validation/test row for the same hypothesis has comparable
        correction metadata.
        """
        if df.empty:
            return df

        df = df.copy()
        train = df[df['split'] == 'train']
        n_global = len(train)
        modality_counts = train.groupby('modality', dropna=False).size().to_dict()

        df['bonferroni_n_tests_global'] = int(n_global)
        df['bonferroni_n_tests_modality'] = (
            df['modality'].map(modality_counts).fillna(0).astype(int)
        )

        raw_p = pd.to_numeric(df.get('odds_ratio_p_value', pd.Series(np.nan, index=df.index)), errors='coerce')
        df['odds_ratio_p_value_bonferroni'] = (raw_p * n_global).clip(upper=1.0) if n_global else np.nan
        df['odds_ratio_p_value_bonferroni_modality'] = (
            raw_p * df['bonferroni_n_tests_modality'].replace(0, np.nan)
        ).clip(upper=1.0)

        # NaN raw p-values remain NaN after multiplication.  Explicit flags
        # make downstream consumers independent of threshold reimplementation.
        df['bonferroni_significant'] = df['odds_ratio_p_value_bonferroni'] < 0.05
        df['bonferroni_significant_modality'] = (
            df['odds_ratio_p_value_bonferroni_modality'] < 0.05
        )
        for prefix in ('adjusted', 'matched_unadjusted'):
            p_column = f'{prefix}_odds_ratio_p_value'
            if p_column in df:
                p = pd.to_numeric(df[p_column], errors='coerce')
                df[f'{p_column}_bonferroni'] = (p * n_global).clip(upper=1.0) if n_global else np.nan
                df[f'{p_column}_bonferroni_modality'] = (
                    p * df['bonferroni_n_tests_modality'].replace(0, np.nan)).clip(upper=1.0)
                df[f'{prefix}_bonferroni_significant'] = df[f'{p_column}_bonferroni'] < 0.05
        return df
    
    def run_all(
        self,
        sae_features: Dict[str, Dict[str, torch.Tensor]],
        alive_features: Dict[Tuple[str, int], torch.Tensor],
        outcomes: Dict[str, Dict[str, np.ndarray]],
        verbose: bool = True,
        *,
        confounders=None,
        adjustment=None,
        confounder_cols=None,
    ) -> pd.DataFrame:
        """
        Run PheWAS for all modalities and top_k values.
        
        Args:
            sae_features: Dict[modality][split] -> feature tensor
            alive_features: Dict[(modality, top_k)] -> alive mask
            outcomes: Dict[split][outcome_col] -> outcome array
            verbose: Whether to show progress
            
        Returns:
            Combined DataFrame with all hypotheses
        """
        all_results = []
        
        for (modality, top_k), alive_mask in alive_features.items():
            print(f"\n  Running PheWAS for {modality} (k={top_k})...")
            
            # Get features for this modality
            features = sae_features[modality]
            
            # Run PheWAS
            df = self.run(
                sae_features=features,
                outcomes=outcomes,
                modality=modality,
                top_k=top_k,
                alive_mask=alive_mask,
                verbose=verbose,
                confounders=confounders,
                adjustment=adjustment,
                confounder_cols=confounder_cols,
            )
            
            if not df.empty:
                all_results.append(df)
        
        if not all_results:
            return pd.DataFrame()
        
        combined = pd.concat(all_results, ignore_index=True)
        return self._add_multiple_testing_corrections(combined)
