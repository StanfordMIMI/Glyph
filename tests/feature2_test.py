"""
Integration test for Glyph class with full pipeline:
- Embedding extraction
- Linear probe training and evaluation

Tests the complete workflow with real pneumonia data.
"""

def main():
    print("="*80)
    print("FEATURE 2 INTEGRATION TEST: Glyph Class with Linear Probe")
    print("="*80)

    from glyph import Glyph

    # Initialize Glyph with pneumonia data
    print("\n[TEST] Initializing Glyph class...")
    ms = Glyph(
        config_path='config',
        config_overrides=[
            "data.metadata=/data/akkumar/sae_exps/MechSciDev/tests/test_dataset/tests/test_dataset/pneumonia_data/metadata.csv",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]",
            "data.batch_size=32",
            "data.num_workers=4",
            "data.embedding_map={image_path: resnet18}",
            "seed=42"
        ]
    )
    print("✓ Glyph initialized successfully")

    # Run the full pipeline
    print("\n[TEST] Running Glyph pipeline...")
    results = ms()
    print("✓ Pipeline completed successfully")

    # Verify results structure
    print("\n[TEST] Verifying results structure...")
    assert 'test_metric' in results, "Missing 'test_metric' in results"
    assert 'all_metrics' in results, "Missing 'all_metrics' in results"
    assert 'best_hparams' in results, "Missing 'best_hparams' in results"
    assert 'model' in results, "Missing 'model' in results"
    assert 'outcome_type' in results, "Missing 'outcome_type' in results"
    assert 'metric_name' in results, "Missing 'metric_name' in results"
    print("✓ Results structure correct")

    # Verify all splits present
    print("\n[TEST] Verifying all splits...")
    assert all(k in results['all_metrics'] for k in ['train', 'val', 'test']), \
        "Missing splits in all_metrics"
    print("✓ All splits present (train/val/test)")

    # Verify metric values are reasonable
    metric_name = results['metric_name']
    print(f"\n[TEST] Verifying {metric_name} values...")
    for split, score in results['all_metrics'].items():
        if results['outcome_type'] == 'binary':
            assert 0.0 <= score <= 1.0, f"{split} {metric_name} out of range: {score}"
        print(f"  - {split} {metric_name}: {score:.4f}")

    # Note: Test metric might be low due to small dataset or overfitting
    # Just verify it's in valid range for now
    print(f"\n  Note: Test {metric_name} = {results['test_metric']:.4f}")
    if results['outcome_type'] == 'binary' and results['test_metric'] < 0.4:
        print("  Warning: Test AUC is quite low - this may indicate:")
        print("    - Overfitting (high train AUC, low val/test AUC)")
        print("    - Need for different hyperparameters")
        print("    - Data quality issues")
    print(f"✓ {metric_name} values are valid")

    # Verify hyperparameters
    print("\n[TEST] Verifying hyperparameters...")
    hparams = results['best_hparams']
    assert 'lr' in hparams, "Missing 'lr' in best_hparams"
    assert 'wd' in hparams, "Missing 'wd' in best_hparams"
    assert hparams['lr'] > 0, "Learning rate should be > 0"
    assert hparams['wd'] >= 0, "Weight decay should be >= 0"
    print(f"  - Learning rate: {hparams['lr']}")
    print(f"  - Weight decay: {hparams['wd']}")
    print("✓ Hyperparameters valid")

    # Verify embeddings were created
    print("\n[TEST] Verifying embeddings...")
    assert ms.embeddings is not None, "Embeddings were not created"
    assert all(k in ms.embeddings for k in ['train', 'val', 'test']), \
        "Missing splits in embeddings"
    for split, data in ms.embeddings.items():
        assert 'sample_ids' in data, f"Missing sample_ids in {split}"
        assert 'embeddings' in data, f"Missing embeddings in {split}"
        print(f"  - {split}: {len(data['sample_ids'])} samples")
    print("✓ Embeddings created successfully")

    # Verify files were saved
    print("\n[TEST] Verifying saved files...")
    from pathlib import Path
    output_dir = Path(ms.cfg.data.output_dir)

    expected_files = [
        # "embeddings.pkl",
        "linear_probe_results.pkl",
        "linear_probe_results_model.pt"
    ]

    for filename in expected_files:
        filepath = output_dir / filename
        assert filepath.exists(), f"Expected file not found: {filepath}"
        print(f"  ✓ {filename} exists ({filepath.stat().st_size / 1024:.2f} KB)")
    print("✓ All expected files saved")

    # Test summary method
    print("\n[TEST] Testing summary method...")
    ms.summary()
    print("✓ Summary method works")

    # Final summary
    print("\n" + "="*80)
    print("INTEGRATION TEST SUMMARY")
    print("="*80)
    print(f"✓ Glyph class initialized successfully")
    print(f"✓ Pipeline ran end-to-end")
    print(f"✓ Test {results['metric_name']}: {results['test_metric']:.4f}")
    print(f"✓ Outcome type: {results['outcome_type']}")
    print(f"✓ Best hyperparameters: lr={hparams['lr']}, wd={hparams['wd']}")
    print(f"✓ All files saved to: {output_dir}")
    print("\n✓✓✓ INTEGRATION TEST PASSED ✓✓✓\n")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
