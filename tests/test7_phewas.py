"""
Unit tests for PheWAS (Phenome-Wide Association Study) implementation.
"""

import pytest
import torch
import numpy as np
import pandas as pd

from glyph.phewas import PheWAS, compute_odds_ratio, compute_feature_prevalence, compute_outcome_prevalence
from glyph.phewas.statistics import compute_conditional_probability
from glyph.config import load_config


class TestStatistics:
    """Test cases for PheWAS statistics functions."""

    def test_compute_feature_prevalence(self):
        """Test feature prevalence computation."""
        feature = np.array([0, 0.5, 0, 1.0, 0.3, 0, 0])
        c_f = compute_feature_prevalence(feature)
        assert c_f == 3  # Three values > 0

    def test_compute_outcome_prevalence(self):
        """Test outcome prevalence computation."""
        outcome = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        p_y = compute_outcome_prevalence(outcome)
        assert p_y == 0.5  # 4/8 = 0.5

    def test_compute_conditional_probability(self):
        """Test P(y|f) computation."""
        feature = np.array([0, 1.0, 0, 1.0, 0.5, 0, 0, 0])  # 3 activations
        outcome = np.array([0, 1, 0, 1, 0, 0, 0, 1])  # 3 positive outcomes
        
        result = compute_conditional_probability(feature, outcome)
        
        assert result['c_f'] == 3  # 3 feature activations
        assert result['c_yf'] == 2  # 2 times outcome=1 when feature>0
        assert abs(result['p_y_given_f'] - 2/3) < 1e-6

    def test_compute_odds_ratio_basic(self):
        """Test odds ratio computation with synthetic data."""
        np.random.seed(42)
        n = 200
        
        # Create feature that correlates with outcome
        feature = np.random.randn(n)
        prob = 1 / (1 + np.exp(-feature))  # Logistic transform
        outcome = (np.random.rand(n) < prob).astype(float)
        
        result = compute_odds_ratio(feature, outcome, seed=42)
        
        assert 'odds_ratio' in result
        assert 'odds_ratio_ci_lower' in result
        assert 'odds_ratio_ci_upper' in result
        assert 'p_value' in result
        assert 'auc' in result
        
        # Odds ratio should be > 1 since feature positively correlates with outcome
        assert result['odds_ratio'] > 1.0
        
        # CI lower should be less than odds ratio
        assert result['odds_ratio_ci_lower'] < result['odds_ratio']
        
        # CI upper should be greater than odds ratio
        assert result['odds_ratio_ci_upper'] > result['odds_ratio']
        
        # AUC should be > 0.5 (better than random)
        assert result['auc'] > 0.5

    def test_compute_odds_ratio_negative_correlation(self):
        """Test odds ratio with negatively correlated feature."""
        np.random.seed(42)
        n = 200
        
        # Create feature that negatively correlates with outcome
        feature = np.random.randn(n)
        prob = 1 / (1 + np.exp(feature))  # Negative correlation
        outcome = (np.random.rand(n) < prob).astype(float)
        
        result = compute_odds_ratio(feature, outcome, seed=42)
        
        # Odds ratio should be < 1 for negative correlation
        assert result['odds_ratio'] < 1.0


class TestPheWAS:
    """Test cases for PheWAS class."""

    def test_phewas_initialization(self):
        """Test PheWAS initializes correctly."""
        cfg = load_config(overrides=[
            "phewas.min_activations=25",
        ])
        
        phewas = PheWAS(cfg)
        
        assert phewas.min_activations == 25
        assert phewas.seed == 42

    def test_normalize_features(self):
        """Test feature normalization."""
        cfg = load_config()
        phewas = PheWAS(cfg)
        
        # Create features with known statistics
        train_features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        val_features = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        
        features = {'train': train_features, 'val': val_features}
        normalized = phewas.normalize_features(features)
        
        # Train should have mean ~0 and std ~1
        assert abs(normalized['train'].mean()) < 0.1
        assert abs(normalized['train'].std() - 1.0) < 0.2  # Close to 1

    def test_phewas_run_synthetic(self):
        """Test PheWAS run with synthetic data."""
        np.random.seed(42)
        
        cfg = load_config(overrides=[
            "phewas.min_activations=10",  # Lower threshold for test
        ])
        
        phewas = PheWAS(cfg)
        
        # Create synthetic SAE features (100 samples, 50 features)
        n_train, n_val, n_test = 100, 30, 30
        n_features = 50
        
        sae_features = {
            'train': torch.randn(n_train, n_features).abs(),  # Positive values
            'val': torch.randn(n_val, n_features).abs(),
            'test': torch.randn(n_test, n_features).abs(),
        }
        
        # Create synthetic outcomes (binary)
        outcomes = {
            'train': {'outcome1': np.random.randint(0, 2, n_train).astype(float)},
            'val': {'outcome1': np.random.randint(0, 2, n_val).astype(float)},
            'test': {'outcome1': np.random.randint(0, 2, n_test).astype(float)},
        }
        
        # Run PheWAS
        results = phewas.run(
            sae_features=sae_features,
            outcomes=outcomes,
            modality='test_modality',
            top_k=10,
            alive_mask=None,
            verbose=False,
        )
        
        # Check results structure
        assert isinstance(results, pd.DataFrame)
        
        if len(results) > 0:
            assert 'feature_idx' in results.columns
            assert 'outcome' in results.columns
            assert 'split' in results.columns
            assert 'odds_ratio' in results.columns
            assert 'auc' in results.columns
            assert 'c_f' in results.columns
            assert 'train_odds_ratio_rank' in results.columns

    def test_phewas_min_activations_filter(self):
        """Test that features with low activations are filtered."""
        cfg = load_config(overrides=[
            "phewas.min_activations=50",  # High threshold
        ])
        
        phewas = PheWAS(cfg)
        
        # Create features where most values are 0 (sparse)
        n = 100
        sparse_features = torch.zeros(n, 10)
        sparse_features[:30, :] = 1.0  # Only 30 samples have activations
        
        sae_features = {
            'train': sparse_features,
            'val': sparse_features[:30],
            'test': sparse_features[:30],
        }
        
        outcomes = {
            'train': {'outcome1': np.random.randint(0, 2, n).astype(float)},
            'val': {'outcome1': np.random.randint(0, 2, 30).astype(float)},
            'test': {'outcome1': np.random.randint(0, 2, 30).astype(float)},
        }
        
        results = phewas.run(
            sae_features=sae_features,
            outcomes=outcomes,
            modality='test',
            top_k=5,
            verbose=False,
        )
        
        # Should have no results since c_f=30 < min_activations=50
        assert len(results) == 0

    def test_multiple_testing_corrections(self):
        """Global and modality-level adjusted p-values use train test counts."""
        rows = []
        for split in ('train', 'test'):
            for modality, p_values in {'ehr': (0.01, 0.20), 'image': (0.03,)}.items():
                for feature_idx, p_value in enumerate(p_values):
                    rows.append({
                        'split': split,
                        'modality': modality,
                        'odds_ratio_p_value': p_value,
                        'feature_idx': feature_idx,
                    })

        corrected = PheWAS._add_multiple_testing_corrections(pd.DataFrame(rows))
        ehr = corrected[(corrected['split'] == 'train') & (corrected['modality'] == 'ehr')]
        image = corrected[(corrected['split'] == 'train') & (corrected['modality'] == 'image')]

        assert (corrected['bonferroni_n_tests_global'] == 3).all()
        assert (ehr['bonferroni_n_tests_modality'] == 2).all()
        assert (image['bonferroni_n_tests_modality'] == 1).all()
        assert ehr.iloc[0]['odds_ratio_p_value_bonferroni'] == pytest.approx(0.03)
        assert ehr.iloc[0]['odds_ratio_p_value_bonferroni_modality'] == pytest.approx(0.02)
        assert image.iloc[0]['odds_ratio_p_value_bonferroni'] == pytest.approx(0.09)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
