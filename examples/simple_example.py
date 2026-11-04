"""
Simple example demonstrating Glyph usage on the pneumonia dataset.

Run from project root:
    python examples/simple_example.py
"""

from glyph import Glyph


# User configuration (similar to user.yaml)
USER_CONFIG = [
    # Data paths
    "data.metadata=/dataNAS/people/rholland/pneumonia_data/metadata.csv",
    
    # Input/output columns
    "data.input_cols=[image_path]",
    "data.outcome_cols=[outcome]",
    
    # Embedding model mapping
    "data.embedding_map={image_path: resnet18}",
    
    # DataLoader settings
    "data.batch_size=32",
    "data.num_workers=4",
    
    # Output paths
    "data.cache_dir=./cache",
    "data.output_dir=./outputs",
    
    # Reproducibility
    "seed=42",
    
    # SAE settings (reduce for faster testing)
    "sae.top_ks=[20]",
    "sae.matryoshka=[128, 512, 2048]",
    "sae.max_steps=1000",
]


if __name__ == "__main__":
    # Initialize and run
    ms = Glyph(config_path='config', config_overrides=USER_CONFIG)
    ms.fit()
    ms.hypothesis_search()
    
    # Access results from ms object
    print("\n--- Accessing results from ms object ---")
    print(f"Embeddings: {list(ms.embeddings.keys())}")
    print(f"Embedding linear probe: {list(ms.linear_probe_results.keys())}")
    print(f"SAE models: {list(ms.sae_models.keys())}")
    print(f"SAE alive features: {list(ms.sae_alive_features.keys())}")
    print(f"SAE linear probe: {list(ms.sae_linear_probe_results.keys())}")
    print(f"PheWAS hypotheses: {len(ms.phewas_results)} total")
