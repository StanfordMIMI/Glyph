"""
Test outcome type detection and handling for both binary and regression tasks.
"""

import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import sys

# Add glyph to path
sys.path.insert(0, '/dataNAS/people/slmoro/MechSciDev')

from glyph.config import load_config
from glyph.dataloader.dataset import GlyphDataset
from glyph.metrics.statistics import compute_auc, compute_r2, compute_rmse
from glyph.metrics.linear_probe import run_linear_probe


def create_test_data_binary():
    """Create test metadata CSV with binary outcome."""
    tmp_dir = Path(tempfile.mkdtemp())

    # Create dummy image files
    for i in range(100):
        img_path = tmp_dir / f"image_{i}.jpg"
        img_path.touch()

    # Create metadata with binary outcome (0 and 1)
    metadata = pd.DataFrame({
        'sample_id': [f'sample_{i}' for i in range(100)],
        'image_path': [str(tmp_dir / f"image_{i}.jpg") for i in range(100)],
        'outcome_binary': np.random.choice([0, 1], size=100)  # Binary: 0 or 1
    })

    metadata_path = tmp_dir / 'metadata_binary.csv'
    metadata.to_csv(metadata_path, index=False)

    return tmp_dir, metadata_path


def create_test_data_regression():
    """Create test metadata CSV with regression outcome."""
    tmp_dir = Path(tempfile.mkdtemp())

    # Create dummy image files
    for i in range(100):
        img_path = tmp_dir / f"image_{i}.jpg"
        img_path.touch()

    # Create metadata with continuous outcome
    metadata = pd.DataFrame({
        'sample_id': [f'sample_{i}' for i in range(100)],
        'image_path': [str(tmp_dir / f"image_{i}.jpg") for i in range(100)],
        'outcome_continuous': np.random.randn(100) * 10 + 50  # Continuous values
    })

    metadata_path = tmp_dir / 'metadata_regression.csv'
    metadata.to_csv(metadata_path, index=False)

    return tmp_dir, metadata_path


def test_outcome_type_detection_binary():
    """Test that binary outcomes are correctly detected."""
    print("\n" + "="*80)
    print("TEST 1: Binary Outcome Type Detection")
    print("="*80)

    tmp_dir, metadata_path = create_test_data_binary()

    try:
        # Load config with binary outcome
        cfg = load_config(overrides=[
            f"data.metadata={metadata_path}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome_binary]",
            "data.split_ratios=[0.7,0.15,0.15]",
            "seed=42"
        ])

        # Create dataset
        dataset = GlyphDataset(cfg, split='train')

        # Check outcome type detection
        assert hasattr(dataset, 'outcome_types'), "Dataset should have outcome_types attribute"
        assert 'outcome_binary' in dataset.outcome_types, "outcome_binary should be in outcome_types"
        assert dataset.outcome_types['outcome_binary'] == 'binary', \
            f"Expected 'binary', got '{dataset.outcome_types['outcome_binary']}'"

        print("✓ Binary outcome correctly detected")
        print(f"  Outcome types: {dataset.outcome_types}")

        # Check that outcomes are extracted correctly
        sample = dataset[0]
        assert 'outcomes' in sample, "Sample should have outcomes"
        assert 'outcome_binary' in sample['outcomes'], "Sample should have outcome_binary"
        assert sample['outcomes']['outcome_binary'] in [0, 1], "Outcome should be 0 or 1"

        print("✓ Binary outcomes correctly extracted from samples")
        print(f"  Sample outcome: {sample['outcomes']['outcome_binary']}")

        return True

    finally:
        shutil.rmtree(tmp_dir)


def test_outcome_type_detection_regression():
    """Test that regression outcomes are correctly detected."""
    print("\n" + "="*80)
    print("TEST 2: Regression Outcome Type Detection")
    print("="*80)

    tmp_dir, metadata_path = create_test_data_regression()

    try:
        # Load config with continuous outcome
        cfg = load_config(overrides=[
            f"data.metadata={metadata_path}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome_continuous]",
            "data.split_ratios=[0.7,0.15,0.15]",
            "seed=42"
        ])

        # Create dataset
        dataset = GlyphDataset(cfg, split='train')

        # Check outcome type detection
        assert hasattr(dataset, 'outcome_types'), "Dataset should have outcome_types attribute"
        assert 'outcome_continuous' in dataset.outcome_types, "outcome_continuous should be in outcome_types"
        assert dataset.outcome_types['outcome_continuous'] == 'regression', \
            f"Expected 'regression', got '{dataset.outcome_types['outcome_continuous']}'"

        print("✓ Regression outcome correctly detected")
        print(f"  Outcome types: {dataset.outcome_types}")

        # Check that outcomes are extracted correctly
        sample = dataset[0]
        assert 'outcomes' in sample, "Sample should have outcomes"
        assert 'outcome_continuous' in sample['outcomes'], "Sample should have outcome_continuous"
        assert isinstance(sample['outcomes']['outcome_continuous'], (int, float)), \
            "Outcome should be numeric"

        print("✓ Regression outcomes correctly extracted from samples")
        print(f"  Sample outcome: {sample['outcomes']['outcome_continuous']:.3f}")

        return True

    finally:
        shutil.rmtree(tmp_dir)


def test_linear_probe_binary():
    """Test linear probe with binary classification."""
    print("\n" + "="*80)
    print("TEST 3: Linear Probe with Binary Classification")
    print("="*80)

    np.random.seed(42)

    # Create synthetic data for binary classification
    n_train, n_val, n_test = 100, 20, 20
    n_features = 50

    # Create features with some signal
    X_train = np.random.randn(n_train, n_features)
    X_val = np.random.randn(n_val, n_features)
    X_test = np.random.randn(n_test, n_features)

    # Create labels with signal (first feature correlates with outcome)
    y_train = (X_train[:, 0] + np.random.randn(n_train) * 0.5 > 0).astype(float)
    y_val = (X_val[:, 0] + np.random.randn(n_val) * 0.5 > 0).astype(float)
    y_test = (X_test[:, 0] + np.random.randn(n_test) * 0.5 > 0).astype(float)

    # Run linear probe with binary outcome type
    results = run_linear_probe(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        outcome_type='binary',
        lr_grid=[1e-3, 3e-3],
        wd_grid=[0.0, 1e-4],
        max_epochs=100,
        patience=10,
        verbose=False
    )

    # Check results structure
    assert 'outcome_type' in results, "Results should contain outcome_type"
    assert results['outcome_type'] == 'binary', "Outcome type should be binary"
    assert 'metric_name' in results, "Results should contain metric_name"
    assert results['metric_name'] == 'AUC', "Metric name should be AUC for binary"
    assert 'test_metric' in results, "Results should contain test_metric"
    assert 'all_metrics' in results, "Results should contain all_metrics"

    test_auc = results['test_metric']
    print(f"✓ Linear probe ran successfully with binary outcome")
    print(f"  Test AUC: {test_auc:.4f}")
    print(f"  Train AUC: {results['all_metrics']['train']:.4f}")
    print(f"  Val AUC: {results['all_metrics']['val']:.4f}")

    # AUC should be > 0.5 (better than random) due to signal
    assert test_auc > 0.5, f"Test AUC should be > 0.5, got {test_auc:.4f}"

    print("✓ Binary classification metrics are valid")

    return True


def test_linear_probe_regression():
    """Test linear probe with regression."""
    print("\n" + "="*80)
    print("TEST 4: Linear Probe with Regression")
    print("="*80)

    np.random.seed(42)

    # Create synthetic data for regression
    n_train, n_val, n_test = 100, 20, 20
    n_features = 50

    # Create features
    X_train = np.random.randn(n_train, n_features)
    X_val = np.random.randn(n_val, n_features)
    X_test = np.random.randn(n_test, n_features)

    # Create continuous labels with signal (linear combination of first 3 features)
    y_train = 2 * X_train[:, 0] + 3 * X_train[:, 1] - X_train[:, 2] + np.random.randn(n_train) * 0.5
    y_val = 2 * X_val[:, 0] + 3 * X_val[:, 1] - X_val[:, 2] + np.random.randn(n_val) * 0.5
    y_test = 2 * X_test[:, 0] + 3 * X_test[:, 1] - X_test[:, 2] + np.random.randn(n_test) * 0.5

    # Run linear probe with regression outcome type
    results = run_linear_probe(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        outcome_type='regression',
        lr_grid=[1e-3, 3e-3],
        wd_grid=[0.0, 1e-4],
        max_epochs=100,
        patience=10,
        verbose=False
    )

    # Check results structure
    assert 'outcome_type' in results, "Results should contain outcome_type"
    assert results['outcome_type'] == 'regression', "Outcome type should be regression"
    assert 'metric_name' in results, "Results should contain metric_name"
    assert results['metric_name'] == 'R²', "Metric name should be R² for regression"
    assert 'test_metric' in results, "Results should contain test_metric"
    assert 'all_metrics' in results, "Results should contain all_metrics"

    test_r2 = results['test_metric']
    print(f"✓ Linear probe ran successfully with regression outcome")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Train R²: {results['all_metrics']['train']:.4f}")
    print(f"  Val R²: {results['all_metrics']['val']:.4f}")

    # R² should be positive due to strong signal
    assert test_r2 > 0, f"Test R² should be > 0, got {test_r2:.4f}"

    print("✓ Regression metrics are valid")

    # Test RMSE computation
    y_pred = results['test_preds']
    rmse = compute_rmse(y_test, y_pred)
    print(f"  Test RMSE: {rmse:.4f}")

    return True


def test_metrics():
    """Test metric computation functions."""
    print("\n" + "="*80)
    print("TEST 5: Metric Computation Functions")
    print("="*80)

    # Test AUC
    y_true_binary = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred_probs = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.85, 0.15])
    auc = compute_auc(y_true_binary, y_pred_probs)
    print(f"✓ AUC computed: {auc:.4f}")
    assert 0 <= auc <= 1, "AUC should be between 0 and 1"

    # Test R²
    y_true_continuous = np.array([1.5, 2.3, 3.1, 4.8, 5.2, 6.1])
    y_pred_continuous = np.array([1.6, 2.1, 3.3, 4.5, 5.4, 5.9])
    r2 = compute_r2(y_true_continuous, y_pred_continuous)
    print(f"✓ R² computed: {r2:.4f}")
    assert r2 <= 1, "R² should be <= 1"

    # Test RMSE
    rmse = compute_rmse(y_true_continuous, y_pred_continuous)
    print(f"✓ RMSE computed: {rmse:.4f}")
    assert rmse >= 0, "RMSE should be >= 0"

    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("OUTCOME TYPE DETECTION AND HANDLING TEST SUITE")
    print("="*80)

    tests = [
        test_outcome_type_detection_binary,
        test_outcome_type_detection_regression,
        test_metrics,
        test_linear_probe_binary,
        test_linear_probe_regression,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test_func.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_func.__name__} FAILED with exception:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*80 + "\n")

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
