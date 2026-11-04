"""
Unit tests for metrics module.

Tests that metric computation functions execute without errors
given valid numpy array inputs.
"""

import pytest
import numpy as np

from glyph.metrics.statistics import (
    compute_auc,
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_all_metrics,
)


class TestMetricsComputation:
    """Test that metrics compute without errors for valid inputs."""

    def test_all_metrics_compute_successfully(self):
        """Test that all metrics compute without errors given valid numpy arrays."""
        # Create sample data
        np.random.seed(42)
        n_samples = 100

        # Binary classification
        y_true = np.random.randint(0, 2, n_samples)
        y_pred_proba = np.random.rand(n_samples)
        y_pred_labels = (y_pred_proba >= 0.5).astype(int)

        # Test individual metric functions
        auc = compute_auc(y_true, y_pred_proba)
        assert isinstance(auc, (float, np.floating))
        assert 0.0 <= auc <= 1.0

        accuracy = compute_accuracy(y_true, y_pred_labels)
        assert isinstance(accuracy, (float, np.floating))
        assert 0.0 <= accuracy <= 1.0

        precision = compute_precision(y_true, y_pred_labels)
        assert isinstance(precision, (float, np.floating))
        assert 0.0 <= precision <= 1.0

        recall = compute_recall(y_true, y_pred_labels)
        assert isinstance(recall, (float, np.floating))
        assert 0.0 <= recall <= 1.0

        f1 = compute_f1(y_true, y_pred_labels)
        assert isinstance(f1, (float, np.floating))
        assert 0.0 <= f1 <= 1.0

        # Test batch computation
        metrics = compute_all_metrics(y_true, y_pred_proba)
        assert isinstance(metrics, dict)
        assert 'auc' in metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics

        # All metrics should be valid floats between 0 and 1
        for metric_name, metric_value in metrics.items():
            assert isinstance(metric_value, (float, np.floating))
            assert 0.0 <= metric_value <= 1.0

    def test_metrics_with_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9])

        metrics = compute_all_metrics(y_true, y_pred_proba)

        # Perfect predictions should yield high metrics
        assert metrics['auc'] == 1.0
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0

    def test_auc_requires_two_classes(self):
        """Test that AUC raises error when only one class present."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.1, 0.5, 0.8, 0.9])

        with pytest.raises(ValueError, match="y_true must contain at least 2 classes"):
            compute_auc(y_true, y_pred)

    def test_metrics_with_custom_threshold(self):
        """Test compute_all_metrics with custom threshold."""
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.3, 0.4, 0.6, 0.7])

        # Test with different thresholds
        metrics_low = compute_all_metrics(y_true, y_pred_proba, threshold=0.5)
        metrics_high = compute_all_metrics(y_true, y_pred_proba, threshold=0.65)

        # Both should execute without error and return valid metrics
        assert all(0.0 <= v <= 1.0 for v in metrics_low.values())
        assert all(0.0 <= v <= 1.0 for v in metrics_high.values())

        # Thresholds should affect metrics differently
        assert metrics_low['recall'] != metrics_high['recall']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
