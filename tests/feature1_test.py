"""
Integration test for embedding extraction pipeline.

Tests the full workflow:
1. Load dataset from pneumonia data
2. Extract embeddings using ResNet18
3. Save embeddings to disk
4. Verify caching works on second run
"""

import torch
from pathlib import Path

from glyph.config import load_config
from glyph.dataloader import get_dataloaders
from glyph.models import EmbeddingExtractor


def main():
    print("="*80)
    print("FEATURE 1 INTEGRATION TEST: Embedding Extraction Pipeline")
    print("="*80)

    # Step 1: Load configuration and dataset
    print("\n[STEP 1] Loading configuration and dataset...")
    cfg = load_config(overrides=[
        "data.metadata=/data/akkumar/sae_exps/MechSciDev/tests/test_dataset/tests/test_dataset/pneumonia_data/metadata.csv",
        "data.input_cols=[image_path]",
        "data.outcome_cols=[outcome]",
        "data.batch_size=32",
        "data.num_workers=4",
        "data.cache_dir=./cache",
        "data.split_ratios=[0.7,0.15,0.15]",
        "seed=42"
    ])

    print(f"✓ Config loaded")
    print(f"  - Metadata: {cfg.data.metadata}")
    print(f"  - Input cols: {cfg.data.input_cols}")
    print(f"  - Outcome cols: {cfg.data.outcome_cols}")
    print(f"  - Batch size: {cfg.data.batch_size}")
    print(f"  - Seed: {cfg.seed}")

    # Create dataloaders
    print("\n[STEP 1.1] Creating dataloaders for all splits...")
    dataloaders = get_dataloaders(cfg)

    print(f"✓ Dataloaders created")
    for split, loader in dataloaders.items():
        print(f"  - {split}: {len(loader.dataset)} samples, {len(loader)} batches")

    # Step 2: Initialize embedding extractor
    print("\n[STEP 2] Initializing embedding extractor with ResNet18...")
    embedding_map = {'image_path': 'resnet18'}
    extractor = EmbeddingExtractor(cfg, embedding_map, cache_embeddings=True)

    print(f"✓ EmbeddingExtractor initialized")
    print(f"  - Embedding map: {embedding_map}")
    print(f"  - Cache directory: {extractor.cache_dir}")
    print(f"  - Device: {extractor._get_device()}")
    print(f"  - Models loaded: {list(extractor.models.keys())}")
    for col, model in extractor.models.items():
        print(f"    - {col}: {type(model).__name__} (dim={model.get_embedding_dim()})")

    # Step 3: Extract embeddings (first run - should compute)
    print("\n[STEP 3] Extracting embeddings (FIRST RUN - should compute)...")
    print("-"*80)

    all_embeddings = extractor.extract_all_splits(dataloaders, show_progress=True)

    print("\n✓ Embeddings extracted for all splits")
    for split, result in all_embeddings.items():
        print(f"\n  [{split.upper()}]")
        print(f"    - Number of samples: {len(result['sample_ids'])}")
        print(f"    - Sample IDs (first 5): {result['sample_ids'][:5]}")
        print(f"    - Modalities: {list(result['embeddings'].keys())}")
        for modality, emb_tensor in result['embeddings'].items():
            print(f"      - {modality} shape: {emb_tensor.shape}")
            print(f"      - {modality} dtype: {emb_tensor.dtype}")
            print(f"      - {modality} device: {emb_tensor.device}")
            print(f"      - {modality} mean: {emb_tensor.mean().item():.4f}")
            print(f"      - {modality} std: {emb_tensor.std().item():.4f}")
            print(f"      - {modality} min/max: [{emb_tensor.min().item():.4f}, {emb_tensor.max().item():.4f}]")

    # Step 4: Save embeddings to disk
    print("\n[STEP 4] Saving embeddings to disk...")
    output_dir = Path(cfg.data.output_dir) / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, result in all_embeddings.items():
        output_path = output_dir / f"{split}_embeddings.pt"
        torch.save(result, output_path)
        print(f"✓ Saved {split} embeddings to: {output_path}")
        print(f"  - File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Step 5: Verify caching works (second run - should load from cache)
    print("\n[STEP 5] Testing cache functionality (SECOND RUN - should load from cache)...")
    print("-"*80)

    # Create new extractor instance
    extractor2 = EmbeddingExtractor(cfg, embedding_map, cache_embeddings=True)

    # Extract again - should use cache
    all_embeddings_cached = extractor2.extract_all_splits(dataloaders, show_progress=False)

    print("\n✓ Second extraction completed (should have used cache)")

    # Step 6: Verify cached results match original
    print("\n[STEP 6] Verifying cached results match original...")

    all_match = True
    for split in all_embeddings.keys():
        original = all_embeddings[split]
        cached = all_embeddings_cached[split]

        # Check sample IDs match
        ids_match = original['sample_ids'] == cached['sample_ids']

        # Check embeddings match
        emb_match = torch.allclose(
            original['embeddings']['image_path'],
            cached['embeddings']['image_path'],
            rtol=1e-5,
            atol=1e-8
        )

        if ids_match and emb_match:
            print(f"✓ {split}: Cached results match original")
        else:
            print(f"✗ {split}: Cached results DO NOT match original!")
            print(f"  - IDs match: {ids_match}")
            print(f"  - Embeddings match: {emb_match}")
            all_match = False

    if all_match:
        print("\n✓ All cached results verified!")
    else:
        print("\n✗ Cache verification failed!")
        return False

    # Step 7: Final summary
    print("\n" + "="*80)
    print("INTEGRATION TEST SUMMARY")
    print("="*80)

    total_samples = sum(len(r['sample_ids']) for r in all_embeddings.values())
    total_files = len(all_embeddings)
    cache_files = list(extractor.cache_dir.glob("*.pt"))
    output_files = list(output_dir.glob("*.pt"))

    print(f"✓ Total samples processed: {total_samples}")
    print(f"✓ Splits processed: {list(all_embeddings.keys())}")
    print(f"✓ Embedding dimension: {all_embeddings['train']['embeddings']['image_path'].shape[1]}")
    print(f"✓ Cache files created: {len(cache_files)}")
    for cf in cache_files:
        print(f"  - {cf.name}")
    print(f"✓ Output files created: {len(output_files)}")
    for of in output_files:
        print(f"  - {of.name}")
    print(f"\n✓✓✓ INTEGRATION TEST PASSED ✓✓✓\n")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
