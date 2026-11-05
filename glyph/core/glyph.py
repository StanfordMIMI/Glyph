"""
Main Glyph class - orchestrates the entire hypothesis generation pipeline.
"""

import json
import numpy as np
import torch
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from glyph.metrics.linear_probe import fit_linear_probe_from_embeddings
from glyph.config import load_config
from glyph.dataloader import get_dataloaders
from glyph.models import EmbeddingExtractor
from glyph.mechinterp.sae import SAE, reconstruction_metrics

class Glyph:
    """
    Glyph main class for discovering interpretable features and generating hypotheses.

    Example usage:
        >>> from glyph import Glyph
        >>> ms = Glyph('config')
        >>> results = ms()
        >>> print(f"Test AUC: {results['test_auc']:.4f}")
    """

    def __init__(self, config_path: str, config_overrides: Optional[List[str]] = None):
        """
        Initialize Glyph pipeline.

        Args:
            config_path: Path to configuration YAML file or config name
            config_overrides: Optional list of config overrides (e.g., ['seed=42', 'data.batch_size=64'])
        """
        print("="*80)
        print("INITIALIZING GLYPH PIPELINE")
        print("="*80)

        # Load configuration
        print("\n[1/3] Loading configuration...")
        self.cfg = load_config(config_name=config_path, overrides=config_overrides or [])
        print(f"✓ Config loaded: {config_path}")
        print(f"  - Seed: {self.cfg.seed}")
        print(f"  - Device: {self.cfg.device}")
        print(f"  - Metadata: {self.cfg.data.metadata}")

        # Create dataloaders
        print("\n[2/3] Creating dataloaders...")
        self.dataloaders = get_dataloaders(self.cfg, allow_missing_inputs=True)
        print(f"✓ Dataloaders created:")
        for split, loader in self.dataloaders.items():
            print(f"  - {split}: {len(loader.dataset)} samples")

        # Extract outcomes from dataloaders (all outcome columns)
        self.outcomes = self._extract_outcomes()

        # Get outcome types from train dataset
        train_dataset = self.dataloaders['train'].dataset
        self.outcome_types = train_dataset.outcome_types

        # Display outcome information for all outcome columns
        print(f"✓ Outcomes extracted:")
        for outcome_col in self.cfg.data.outcome_cols:
            outcome_type = self.outcome_types[outcome_col]
            print(f"  - {outcome_col} (type: {outcome_type}):")
            for split in self.outcomes.keys():
                y = self.outcomes[split][outcome_col]
                valid = np.isfinite(y) & (y != -1)
                n_missing = int((~valid).sum())
                y_valid = y[valid]
                unique, counts = np.unique(y_valid, return_counts=True)
                missing_str = f", missing={n_missing}" if n_missing > 0 else ""
                if outcome_type == 'binary':
                    print(f"      {split}: {dict(zip(unique.astype(int), counts))}{missing_str}")
                else:
                    print(f"      {split}: mean={np.mean(y_valid):.3f}, std={np.std(y_valid):.3f}{missing_str}")

        # Models are loaded only when raw extraction is requested by fit().
        self.embedding_extractor = None
        self._output_root = Path(self.cfg.data.output_dir)

        # Placeholders for results
        self.embeddings = None
        self.linear_probe_results = None
        self.sae_models = None
        self.sae_features = None
        self.sae_alive_features = None
        self.sae_linear_probe_results = None
        self.phewas_results = None
        self.autointerp_results = None
        print("\n" + "="*80)
        print("INITIALIZATION COMPLETE")
        print("="*80 + "\n")

    def _extract_outcomes(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract outcome arrays from dataloaders for all outcome columns.

        Returns:
            Dict mapping split names to dicts of outcome column names to outcome arrays
            e.g., {'train': {'outcome1': array, 'outcome2': array}, ...}
        """
        outcomes = {split: {col: [] for col in self.cfg.data.outcome_cols} 
                    for split in self.dataloaders.keys()}

        for split, loader in self.dataloaders.items():
            for batch in loader:
                for outcome_col in self.cfg.data.outcome_cols:
                    batch_outcomes = batch['outcomes'][outcome_col]
                    if isinstance(batch_outcomes, torch.Tensor):
                        outcomes[split][outcome_col].extend(batch_outcomes.tolist())
                    else:
                        outcomes[split][outcome_col].extend(batch_outcomes)
            
            # Convert lists to numpy arrays
            for outcome_col in self.cfg.data.outcome_cols:
                outcomes[split][outcome_col] = np.array(outcomes[split][outcome_col])

        return outcomes

    def _extract_outcomes_aligned_with_embeddings(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract outcomes aligned with embedding sample_ids order.
        
        This is necessary because cached embeddings may have a different sample order
        than a fresh iteration through the dataloader (due to shuffling).
        
        Returns:
            Dict mapping split names to dicts of outcome column names to outcome arrays,
            where the order matches self.embeddings[split]['sample_ids']
        """
        # Load metadata to look up outcomes by sample_id
        metadata = pd.read_csv(self.cfg.data.metadata, dtype={"sample_id": str})
        
        # Handle unnamed index column
        if metadata.columns[0].startswith('Unnamed'):
            metadata = metadata.drop(columns=[metadata.columns[0]])
        
        # Determine sample_id column
        if 'sample_id' in metadata.columns:
            id_col = 'sample_id'
        else:
            id_col = metadata.columns[0]
        
        # Create lookup dict: sample_id -> {outcome_col: value}
        outcome_lookup = {}
        for _, row in metadata.iterrows():
            sample_id = str(row[id_col])
            outcome_lookup[str(sample_id)] = {
                col: row[col] for col in self.cfg.data.outcome_cols
            }
        
        # Extract outcomes in the same order as embeddings
        outcomes = {}
        for split, emb_data in self.embeddings.items():
            sample_ids = emb_data['sample_ids']
            outcomes[split] = {col: [] for col in self.cfg.data.outcome_cols}
            
            for sid in sample_ids:
                for col in self.cfg.data.outcome_cols:
                    outcomes[split][col].append(outcome_lookup[str(sid)][col])
            
            # Convert to numpy arrays
            for col in self.cfg.data.outcome_cols:
                outcomes[split][col] = np.array(outcomes[split][col])
        
        return outcomes

    def __call__(self):
        """
        Run the full pipeline:
        1. Extract embeddings
        2. Compute embedding AUC with linear probe
        3. Fit SAE on embeddings
        4. Compute SAE feature AUC with linear probe
        5. Run PheWAS hypothesis search
        6. Run AutoInterp (if configured)

        Returns:
            Dictionary with all results
        """
        self.fit()
        self.hypothesis_search()
        
        return {
            'linear_probe': self.linear_probe_results,
            'sae_linear_probe': self.sae_linear_probe_results,
            'phewas': self.phewas_results,
            'autointerp': self.autointerp_results,
        }
    
    def fit(self, *, pretrained=None, embeddings_path=None, recompute_embeddings=True):
        """Create a new fitted run, computing raw embeddings unless paths are supplied.

        pretrained maps input columns to named HF SAE bundles. An explicit
        embeddings_path (file, directory, or column-to-file mapping) skips raw
        extraction. recompute_embeddings=False requires such an explicit path.
        """
        previous = getattr(self, "run_id", None)
        try:
            return self._fit(pretrained=pretrained, embeddings_path=embeddings_path,
                             recompute_embeddings=recompute_embeddings)
        except Exception:
            if getattr(self, "run_id", None) != previous:
                path = self.run_dir / "run.json"
                if path.exists():
                    manifest = json.loads(path.read_text())
                    manifest["status"] = "failed"
                    path.write_text(json.dumps(manifest, indent=2))
            raise

    def _fit(self, *, pretrained=None, embeddings_path=None, recompute_embeddings=True):
        """
        Fit the Glyph pipeline (Steps 1-4):
        1. Extract embeddings from raw data
        2. Compute embedding AUC with linear probe
        3. Fit SAE on embeddings
        4. Compute SAE feature AUC with linear probe
        
        Returns:
            self (for method chaining)
        """
        print("="*80)
        print("RUNNING GLYPH FIT")
        print("="*80)

        from glyph.core.runs import start_run, fit_embeddings
        start_run(self, pretrained, embeddings_path, recompute_embeddings)
        fit_embeddings(self, pretrained, embeddings_path, recompute_embeddings)

        # Re-extract outcomes aligned with embedding sample_ids
        self.outcomes = self._extract_outcomes_aligned_with_embeddings()

        # Save embeddings
        print("\nSaving embeddings...")
        self._save_embeddings()

        # Get embedding map
        embedding_map = self.cfg.data.get('embedding_map', {'image_path': 'resnet18'})
        if hasattr(embedding_map, '_content'):
            embedding_map = dict(embedding_map)
        

        # Step 2: Linear probe on embeddings (skippable - set
        # skip_embeddings_probe=True to jump straight to SAE training when
        # the raw-embedding baseline AUC isn't needed again)
        self.linear_probe_results = {}
        if not self.cfg.get('skip_embeddings_probe', False):
            print("\n[STEP 2] Running linear probe on embeddings...")
            print("-"*80)

            for outcome_col in self.cfg.data.outcome_cols:
                print(f"\n  Processing outcome: {outcome_col}")
                outcome_type = self.outcome_types[outcome_col]

                outcomes_for_col = {
                    split: self.outcomes[split][outcome_col]
                    for split in self.outcomes.keys()
                }

                for modality in embedding_map.keys():
                    result = fit_linear_probe_from_embeddings(
                        embeddings_dict=self.embeddings,
                        outcomes_dict=outcomes_for_col,
                        outcome_type=outcome_type,
                        modality=modality,
                        device=self.cfg.device,
                        seed=self.cfg.seed,
                        verbose=True
                    )
                    self.linear_probe_results[(outcome_col, modality)] = result

            self._save_results(self.linear_probe_results, "linear_probe_results.pkl")
        else:
            print("\n[STEP 2] Skipping linear probe on embeddings (skip_embeddings_probe=True)")

        # Step 3: Fit SAE on embeddings
        print("\n[STEP 3] Training Sparse Autoencoders...")
        print("-"*80)

        self.sae_models = {}
        self.sae_features = {}
        self.sae_alive_features = {}

        from glyph.core.runs import fit_dictionaries
        fit_dictionaries(self, pretrained)

        # Save SAE features
        print("\nSaving SAE features...")
        self._save_sae_features()
        from glyph.core.runs import save_run
        save_run(self)

        # Step 4: Linear probe on SAE features
        print("\n[STEP 4] Running linear probe on SAE features...")
        print("-"*80)

        self.sae_linear_probe_results = {}
        for outcome_col in self.cfg.data.outcome_cols:
            print(f"\n  Processing outcome: {outcome_col}")
            outcome_type = self.outcome_types[outcome_col]
            
            outcomes_for_col = {
                split: self.outcomes[split][outcome_col] 
                for split in self.outcomes.keys()
            }
            
            for modality in embedding_map.keys():
                sae = self.sae_models[modality]
                
                for top_k in sae.models.keys():
                    # Prepare SAE features dict in same format as embeddings
                    sae_emb_dict = {
                        split: {
                            'sample_ids': self.embeddings[split]['sample_ids'],
                            'embeddings': {modality: self.sae_features[modality][split]}
                        }
                        for split in self.embeddings.keys()
                    }
                    
                    result = fit_linear_probe_from_embeddings(
                        embeddings_dict=sae_emb_dict,
                        outcomes_dict=outcomes_for_col,
                        outcome_type=outcome_type,
                        modality=modality,
                        device=self.cfg.device,
                        seed=self.cfg.seed,
                        verbose=True
                    )
                    self.sae_linear_probe_results[(outcome_col, modality, top_k)] = result

        self._save_results(self.sae_linear_probe_results, "sae_linear_probe_results.pkl")

        print("\n" + "="*80)
        print("FIT COMPLETE")
        print("="*80 + "\n")
        
        from glyph.core.runs import complete_run
        complete_run(self)
        return self

    @classmethod
    def load_run(cls, path, *, device="auto"):
        from glyph.core.runs import load_run
        return load_run(cls, path, device=device)

    def encode_embeddings(self, embeddings, *, modality):
        """Apply this run's saved normalization and dictionary to raw FM embeddings."""
        from glyph.core.runs import encode_embeddings
        return encode_embeddings(self, embeddings, modality)

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        """Load a published dictionary; raw inputs are encoded later by measure()."""
        from glyph.core.measurement import from_pretrained
        return from_pretrained(cls, name, **kwargs)

    def measure(self, code=None, *, metadata=None, sample_id=None, split=None,
                modality=None, embeddings_path=None, recompute=False):
        """Measure one concept or the whole space, indexed by sample_id."""
        selected = None
        if hasattr(self, '_measurement_bundle'):
            from glyph.core.measurement import validate_code, prepare
            validate_code(self._measurement_bundle, code)
            selected = prepare(self, metadata, sample_id=sample_id, split=split,
                               embeddings_path=embeddings_path, recompute=recompute)
            split = None
        elif metadata is not None or embeddings_path is not None or recompute:
            raise ValueError('Use Glyph.from_pretrained() to measure new data, or fit() first')
        from glyph.core.concepts import measure
        result = measure(self, code, split=split, modality=modality)
        if selected is not None:
            return result.loc[selected]
        if sample_id is not None:
            return result.loc[[str(sample_id)]]
        return result

    def hypothesis_search(self, *, adjust_confounders=False, confounder_cols=None,
                          adjustment_device='cpu'):
        """
        Run PheWAS hypothesis search and optional AutoInterp on fitted SAE features.
        
        Steps:
        5. Run PheWAS study
        6. Run AutoInterp (if API key is configured)
        
        Must be called after fit().

        Set adjust_confounders=True and provide metadata confounder_cols to add
        FMR orthogonal adjustment. It defaults to CPU independently of cfg.device.
        adjustment_device='cuda' explicitly requests GPU acceleration.
        
        Returns:
            self (for method chaining)
        """
        from glyph.phewas import PheWAS

        if not self.cfg.data.outcome_cols:
            raise ValueError("Association testing requires outcome labels. Use fit() and visualize() for exploration without labels.")
        if self.sae_features is None:
            raise RuntimeError("Run fit() before hypothesis_search().")

        adjustment = None
        confounders = None
        if not adjust_confounders and (confounder_cols is not None or adjustment_device != 'cpu'):
            raise ValueError('Set adjust_confounders=True to use confounder adjustment options')
        if adjust_confounders:
            from glyph.phewas.confounder_adjustment import FMRConfounderAdjustment
            confounders, adjustment_manifest = self._prepare_confounders(confounder_cols)
            settings = dict(self.cfg.phewas.get('confounder_adjustment', {}))
            adjustment = FMRConfounderAdjustment(
                device=adjustment_device, seed=self.cfg.seed, **settings)
        
        print("="*80)
        print("RUNNING HYPOTHESIS SEARCH")
        print("="*80)
        if adjustment is not None:
            print(f"Confounder adjustment enabled: FMR orthogonal, "
                  f"depth {adjustment.params['max_depth']}, device {adjustment_device}")
        
        print("\n[STEP 5] Running PheWAS study...")
        print("-"*80)
        
        phewas = PheWAS(self.cfg)
        self.phewas_results = phewas.run_all(
            sae_features=self.sae_features,
            alive_features=self.sae_alive_features,
            outcomes=self.outcomes,
            verbose=True,
            confounders=confounders,
            adjustment=adjustment,
            confounder_cols=confounder_cols,
        )
        
        # Save hypotheses
        print("\nSaving PheWAS results...")
        self._save_phewas_results()
        if adjustment is not None:
            adjustment_manifest.update(
                method=adjustment.method, device=adjustment_device,
                settings={**adjustment.params, 'n_splits': adjustment.n_splits,
                          'inner_splits': adjustment.inner_splits,
                          'inner_seed_rule': 'seed + outer_fold_index + 1',
                          'folds': 'stratified', 'xgboost_version': adjustment.xgb.__version__},
                feature_units='per Discovery-train standard deviation',
                observation_unit='study',
            )
            manifest_path = Path(self.cfg.data.output_dir) / 'confounder_adjustment.json'
            manifest_path.write_text(json.dumps(adjustment_manifest, indent=2))
        else:
            manifest_path = Path(self.cfg.data.output_dir) / 'confounder_adjustment.json'
            manifest_path.unlink(missing_ok=True)

        # Step 6: AutoInterp (if configured)
        from glyph.interpretation.settings import ready
        llm_ready = ready(self.cfg)
        run_autointerp = (
            llm_ready
            and self.phewas_results is not None
            and len(self.phewas_results) > 0
        )

        if run_autointerp:
            print("\n[STEP 6] Running AutoInterp...")
            print("-"*80)
            self._run_autointerp()
        else:
            if not llm_ready:
                print("\n[STEP 6] Skipping AutoInterp (disabled or no LLM configured)")
            else:
                print("\n[STEP 6] Skipping AutoInterp (no PheWAS results)")

        if self.cfg.visualization.get('auto_generate', False) and not self.phewas_results.empty:
            print("\n[STEP 7] Building interactive concept map...")
            print("-"*80)
            self.visualize()

        # Print summary
        self._print_summary()

        return self

    def _prepare_confounders(self, confounder_cols):
        from glyph.phewas.confounder_adjustment import encode_confounders

        if isinstance(confounder_cols, str) or not confounder_cols:
            raise ValueError('Provide a nonempty list of metadata confounder_cols')
        columns = list(confounder_cols)
        if len(set(columns)) != len(columns):
            raise ValueError('confounder_cols must not contain duplicate columns')
        if set(columns) & set(self.cfg.data.outcome_cols):
            raise ValueError('Outcome columns cannot also be used as confounders')
        header = pd.read_csv(self.cfg.data.metadata, nrows=0).columns
        header = [c for c in header if not c.startswith('Unnamed')]
        id_col = 'sample_id' if 'sample_id' in header else header[0]
        metadata = pd.read_csv(self.cfg.data.metadata, dtype={id_col: str})
        missing = set(columns) - set(metadata.columns)
        if missing:
            raise ValueError(f'Confounder columns missing from metadata: {sorted(missing)}')
        if metadata[id_col].isna().any() or metadata[id_col].duplicated().any():
            raise ValueError('Confounder alignment requires unique, nonmissing study IDs')
        if self.embeddings is None:
            raise ValueError('Run fit() or hypothesis_search_from_csv() before adjustment')
        lookup = metadata.set_index(id_col)
        frames = {}
        for split, data in self.embeddings.items():
            ids = [str(sid) for sid in data['sample_ids']]
            if not pd.Index(ids).isin(lookup.index).all():
                raise ValueError(f'Embedding study IDs for {split} are missing from metadata')
            frames[split] = lookup.loc[ids, columns]
        matrices, encoded_columns, categories = encode_confounders(frames)
        return matrices, dict(covariates=columns, encoded_covariates=encoded_columns,
                              training_categories=categories,
                              missing_values='complete-case exclusion; no imputation',
                              unseen_categories='excluded using training category levels')

    def hypothesis_search_from_csv(self, csv_dir=None, *, adjust_confounders=False,
                                   confounder_cols=None, adjustment_device='cpu'):
        """
        Like hypothesis_search(), but skips fit() entirely and loads a
        previously-saved {modality}_concepts_supported.csv (written by
        _save_filtered_concepts_csv() at the end of a prior fit() run)
        instead of the full dense SAE feature pickle - the CSV only has
        the concepts that already met phewas.min_activations, so this is
        much cheaper when re-running PheWAS with a different outcome set
        against an SAE that's already trained.

        Call on a fresh Glyph instance (after __init__, without fit()).
        """
        csv_dir = Path(csv_dir) if csv_dir else Path(self.cfg.data.output_dir)

        embedding_map = self.cfg.data.get('embedding_map', {'image_path': 'resnet18'})
        if hasattr(embedding_map, '_content'):
            embedding_map = dict(embedding_map)

        header = pd.read_csv(self.cfg.data.metadata, nrows=0).columns
        header = [c for c in header if not c.startswith('Unnamed')]
        id_col = 'sample_id' if 'sample_id' in header else header[0]
        metadata = pd.read_csv(self.cfg.data.metadata, dtype={id_col: str})
        if metadata[id_col].isna().any() or metadata[id_col].duplicated().any():
            raise ValueError('CSV resume requires unique, nonmissing metadata IDs')
        outcome_lookup = metadata.set_index(id_col)

        self.sae_features = {}
        self.sae_alive_features = {}
        self.outcomes = {}
        self.embeddings = {}
        selection_path = csv_dir / 'sae_selection.json'
        selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}

        for modality in embedding_map.keys():
            csv_path = csv_dir / f"{modality}_concepts_supported.csv"
            print(f"Loading filtered concepts from {csv_path} (skipping fit())...")
            df = pd.read_csv(csv_path, dtype={'sample_id': str})
            import re
            concept_cols = [c for c in df.columns if re.fullmatch(r"Concept_\d+", c)]
            if not concept_cols:
                raise ValueError(f"No Concept_<index> columns in {csv_path}")
            feature_ids = [int(c.split('_')[1]) for c in concept_cols]
            if len(set(feature_ids)) != len(feature_ids):
                raise ValueError("Concept CSV contains duplicate concept indices")
            if df['sample_id'].isna().any() or df['sample_id'].duplicated().any():
                raise ValueError("Concept CSV needs unique, nonmissing study IDs")
            if not set(df['split']).issubset({'train', 'val', 'test'}):
                raise ValueError("Concept CSV split labels must be train, val, or test")
            if max(feature_ids) >= max(self.cfg.sae.matryoshka):
                raise ValueError("Concept index exceeds the configured dictionary size")
            width = max(max(feature_ids) + 1, selection.get(modality, {}).get("dictionary_size", 0))
            if width > max(self.cfg.sae.matryoshka):
                raise ValueError("Saved dictionary size exceeds configured dictionary size")
            df['sample_id'] = df['sample_id'].astype(str)
            print(f"  {modality}: {len(concept_cols)} supported concepts, {len(df)} samples")

            self.sae_features[modality] = {}
            if modality in selection:
                top_k = selection[modality]['selected_top_k']
            elif len(self.cfg.sae.top_ks) == 1:
                top_k = self.cfg.sae.top_ks[0]
            else:
                raise ValueError('CSV resume needs sae_selection.json or one explicit sae.top_ks value')
            for split in df['split'].unique():
                sub = df[df['split'] == split]
                ids = sub['sample_id'].tolist()
                if split in self.embeddings and self.embeddings[split]['sample_ids'] != ids:
                    raise ValueError('Concept CSVs must use the same study order across modalities')
                self.embeddings[split] = {'sample_ids': ids, 'embeddings': {}}
                values = torch.tensor(sub[concept_cols].values, dtype=torch.float32)
                if not torch.isfinite(values).all():
                    raise ValueError("Concept CSV activations must be finite")
                restored = torch.zeros(len(sub), width, dtype=torch.float32)
                restored[:, feature_ids] = values
                self.sae_features[modality][split] = restored
                sub_outcomes = outcome_lookup.loc[sub['sample_id']]
                self.outcomes.setdefault(split, {})
                for col in self.cfg.data.outcome_cols:
                    self.outcomes[split][col] = sub_outcomes[col].fillna(-1).to_numpy(dtype=float)

            # already filtered to supported concepts - all columns count as alive
            restored_mask = torch.zeros(width, dtype=torch.bool)
            restored_mask[feature_ids] = True
            self.sae_alive_features[(modality, top_k)] = restored_mask

        self.hypothesis_search(adjust_confounders=adjust_confounders,
                               confounder_cols=confounder_cols,
                               adjustment_device=adjustment_device)
        return self

    def _run_autointerp(self):
        """Run AutoInterp on PheWAS results."""
        from glyph.interpretation import AutoInterp
        
        autointerp = AutoInterp(self.cfg)
        
        # Build sample_ids dict from embeddings
        sample_ids = {
            split: data['sample_ids']
            for split, data in self.embeddings.items()
        }
        
        output_dir = str(Path(self.cfg.data.output_dir))
        
        self.autointerp_results = autointerp.run_all(
            sae_features=self.sae_features,
            sample_ids=sample_ids,
            hypotheses=self.phewas_results,
            output_dir=output_dir,
            verbose=True,
        )
        
        if self.autointerp_results is not None and len(self.autointerp_results) > 0:
            print(f"\n  AutoInterp complete: {len(self.autointerp_results)} interpretations")
        else:
            print("\n  AutoInterp: No interpretations generated")

    def _save_embeddings(self):
        """
        Save embeddings in spec format: {modality}_embeddings.pt
        
        Each file contains a dict with:
        - 'sample_id': list of sample IDs
        - '{modality}_embedding': tensor of shape (n_samples, embedding_dim)
        - 'split': list indicating which split each sample belongs to
        """
        output_dir = Path(self.cfg.data.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get embedding map
        embedding_map = self.cfg.data.get('embedding_map', {'image_path': 'resnet18'})
        if hasattr(embedding_map, '_content'):
            embedding_map = dict(embedding_map)

        for modality in embedding_map.keys():
            # Collect embeddings from all splits
            all_sample_ids = []
            all_embeddings = []
            all_splits = []
            
            for split, data in self.embeddings.items():
                n_samples = len(data['sample_ids'])
                all_sample_ids.extend(data['sample_ids'])
                all_embeddings.append(data['embeddings'][modality])
                all_splits.extend([split] * n_samples)
            
            # Stack embeddings
            combined_embeddings = torch.cat(all_embeddings, dim=0)
            
            # Create output dict matching spec format
            # Get clean modality name (e.g., 'image' from 'image_path')
            modality_name = modality.replace('_path', '')
            output_data = {
                'sample_id': all_sample_ids,
                f'{modality_name}_embedding': combined_embeddings,
                'split': all_splits,
                'encoder': str(embedding_map[modality]),
                'schema_version': 1,
            }
            
            # Save as {modality}_embeddings.pt
            output_path = output_dir / f"{modality_name}_embeddings.pt"
            torch.save(output_data, output_path)
            print(f"  ✓ Saved {modality_name} embeddings to: {output_path}")
            print(f"    - Shape: {combined_embeddings.shape}")

    def _save_sae_features(self):
        """
        Save SAE features and alive features to disk.
        
        Saves:
        - sae_features.pkl: Sparse activations per split/modality/top_k
        - sae_alive_features.pkl: Alive feature masks per modality/top_k
        """
        output_dir = Path(self.cfg.data.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        selection = {modality: sae.selection_report for modality, sae in self.sae_models.items()}
        (output_dir / 'sae_selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False))
        
        # Save SAE features
        sae_features_path = output_dir / "sae_features.pkl"
        sae_data = {}
        for modality, splits in self.sae_features.items():
            sae_data[modality] = {
                split: features.cpu() for split, features in splits.items()
            }
        with open(sae_features_path, 'wb') as f:
            pickle.dump(sae_data, f)
        print(f"  ✓ SAE features saved to: {sae_features_path}")
        
        # Save alive features
        alive_path = output_dir / "sae_alive_features.pkl"
        alive_data = {
            key: mask.cpu() for key, mask in self.sae_alive_features.items()
        }
        with open(alive_path, 'wb') as f:
            pickle.dump(alive_data, f)
        print(f"  ✓ Alive features saved to: {alive_path}")

        self._save_filtered_concepts_csv()

    def _save_filtered_concepts_csv(self):
        """
        Save one CSV per modality containing only concepts that meet
        phewas.min_activations (a stricter subset of "alive" = fired at
        least once) - PheWAS never uses the other columns anyway, so a
        resumed hypothesis_search() shouldn't have to load the full dense
        [n_samples, n_concepts] tensor just to discard most of it.

        Columns: sample_id, split, Concept_<original index> for each
        supported concept. The original index is preserved in the column
        name so results stay traceable back to the full dictionary.
        """
        output_dir = Path(self.cfg.data.output_dir)
        min_activations = self.cfg.phewas.min_activations

        for modality, splits in self.sae_features.items():
            train_features = splits.get('train', next(iter(splits.values())))
            n_concepts = train_features.shape[1]

            counts = torch.zeros(n_concepts, dtype=torch.long)
            for split_features in splits.values():
                counts += (split_features.cpu() > 1e-8).sum(dim=0)
            supported = torch.where(counts >= min_activations)[0].tolist()

            print(f"  {modality}: {len(supported)}/{n_concepts} concepts meet "
                  f"min_activations={min_activations} (vs {int((counts > 0).sum())} merely alive)")
            if not supported:
                continue

            rows, sample_ids, split_col = [], [], []
            for split, features in splits.items():
                rows.append(features[:, supported].cpu().numpy())
                sample_ids.extend(self.embeddings[split]['sample_ids'])
                split_col.extend([split] * len(self.embeddings[split]['sample_ids']))
            matrix = np.concatenate(rows, axis=0)

            df = pd.DataFrame(matrix, columns=[f"Concept_{i}" for i in supported])
            df.insert(0, "sample_id", sample_ids)
            df.insert(1, "split", split_col)

            csv_path = output_dir / f"{modality}_concepts_supported.csv"
            df.to_csv(csv_path, index=False)
            print(f"  ✓ Filtered concept CSV saved to: {csv_path} ({df.shape})")

    def _save_phewas_results(self):
        """Save PheWAS results to hypotheses.pkl."""
        output_dir = Path(self.cfg.data.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        hypotheses_path = output_dir / "hypotheses.pkl"
        self.phewas_results.to_pickle(hypotheses_path)
        print(f"  ✓ Hypotheses saved to: {hypotheses_path}")
        print(f"    - Total hypotheses: {len(self.phewas_results)}")

    def _save_results(self, data: Any, filename: str):
        """
        Save results to output directory.

        Args:
            data: Data to save
            filename: Output filename
        """
        output_dir = Path(self.cfg.data.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / filename

        # Handle torch models separately
        if isinstance(data, dict) and 'model' in data:
            # Save model state dict separately
            model_path = output_dir / filename.replace('.pkl', '_model.pt')
            torch.save(data['model'].state_dict(), model_path)
            print(f"  ✓ Model saved to: {model_path}")

            # Save rest of data without model
            data_copy = {k: v for k, v in data.items() if k != 'model'}
            with open(output_path, 'wb') as f:
                pickle.dump(data_copy, f)
        else:
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)

        print(f"  ✓ Results saved to: {output_path}")

    def _print_summary(self):
        """Print pipeline results summary."""
        print("\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80)

        if self.embeddings:
            total_samples = sum(len(e['sample_ids']) for e in self.embeddings.values())
            print(f"\n[Embeddings]")
            print(f"  Total samples: {total_samples}")
            for split, data in self.embeddings.items():
                for modality, emb_tensor in data['embeddings'].items():
                    emb_shape = emb_tensor.shape
                    print(f"  {split:5s}: {emb_shape[0]:4d} samples × {emb_shape[1]:4d} dims ({modality})")

        if self.linear_probe_results:
            print(f"\n[Embedding Linear Probe Results]")
            for (outcome_col, modality), result in self.linear_probe_results.items():
                metric_name = result.get('metric_name', 'AUC')
                test_score = result['all_metrics']['test']
                print(f"  {outcome_col} ({modality}): Test {metric_name} = {test_score:.4f}")

        if self.sae_alive_features:
            print(f"\n[SAE Summary]")
            for (modality, top_k), mask in self.sae_alive_features.items():
                n_alive = mask.sum().item()
                n_total = len(mask)
                print(f"  {modality} (k={top_k}): {n_alive}/{n_total} alive features")

        if self.sae_linear_probe_results:
            print(f"\n[SAE Linear Probe Results]")
            for (outcome_col, modality, top_k), result in self.sae_linear_probe_results.items():
                metric_name = result.get('metric_name', 'AUC')
                test_score = result['all_metrics']['test']
                print(f"  {outcome_col} ({modality}, k={top_k}): Test {metric_name} = {test_score:.4f}")

        if self.phewas_results is not None and len(self.phewas_results) > 0:
            print(f"\n[PheWAS Results]")
            print(f"  Total hypotheses: {len(self.phewas_results)}")
            if 'adjustment_status' in self.phewas_results:
                counts = self.phewas_results['adjustment_status'].value_counts()
                n_ok = int(counts.get('ok', 0))
                print(f"  Confounder adjustment: {n_ok} fitted, "
                      f"{len(self.phewas_results) - n_ok} unestimable")
            
            # Show outcome class balance
            print(f"\n  Outcome class balance (train):")
            for outcome_col in self.cfg.data.outcome_cols:
                y = self.outcomes['train'][outcome_col]
                valid = np.isfinite(y) & (y != -1)
                y_valid = y[valid]
                n_pos = int((y_valid > 0).sum())
                n_neg = int((y_valid == 0).sum())
                n_valid = int(valid.sum())
                n_missing = len(y) - n_valid
                missing_str = f", {n_missing} missing (-1)" if n_missing > 0 else ""
                print(f"    {outcome_col}: {n_pos}/{n_valid} positive ({100*n_pos/max(n_valid, 1):.1f}%), "
                      f"{n_neg}/{n_valid} negative ({100*n_neg/max(n_valid, 1):.1f}%){missing_str}")
            
            # Show top hypotheses by train odds ratio (with test OR for comparison)
            # Only include hypotheses that also have test results
            train_results = self.phewas_results[self.phewas_results['split'] == 'train']
            test_results = self.phewas_results[self.phewas_results['split'] == 'test']
            
            # Filter to hypotheses with matching test results
            identity = ['modality', 'top_k', 'feature_idx', 'outcome']
            test_keys = set(test_results[identity].itertuples(index=False, name=None))
            train_with_test = train_results[
                train_results.apply(lambda r: tuple(r[key] for key in identity) in test_keys, axis=1)
            ]
            
            if len(train_with_test) > 0:
                n_print = self.cfg.phewas.top_n_print
                top_hyp = train_with_test.nlargest(n_print, 'odds_ratio')[
                    ['feature_name', 'feature_idx', 'top_k', 'outcome', 'modality', 'odds_ratio', 'odds_ratio_p_value',
                     'odds_ratio_p_value_bonferroni', 'auc', 'c_f']
                ]

                n_tests = len(train_results)

                def stars(p_corrected):
                    if np.isnan(p_corrected): return '   '
                    if p_corrected < 0.001:   return '***'
                    if p_corrected < 0.01:    return '** '
                    if p_corrected < 0.05:    return '*  '
                    return '   '

                def fmt_or(or_val, p_val, p_corrected):
                    s = stars(p_corrected)
                    p_str = f"{p_val:.1e}" if not np.isnan(p_val) else "n/a"
                    return f"{or_val:.2f} ({p_str}{' ' + s.strip() if s.strip() else ''})"

                # Get embedding train AUC once for the header
                emb_auc_header = 0.5
                if self.linear_probe_results:
                    key = (top_hyp.iloc[0]['outcome'], top_hyp.iloc[0]['modality'])
                    if key in self.linear_probe_results:
                        emb_auc_header = self.linear_probe_results[key]['all_metrics']['train']

                print(f"\n  Top {n_print} hypotheses (by train odds ratio):  [embedding train AUC = {emb_auc_header:.3f}]")
                print(f"  {'─'*125}")
                print(f"  {'Feature':<15} {'Outcome':<20} {'Train OR (p-val)':>22} {'Test OR (p-val)':>22} {'feat/emb AUC':>14} {'Count':>8}")
                print(f"  {'─'*125}")

                for _, row in top_hyp.iterrows():
                    test_row = test_results[
                        (test_results['feature_name'] == row['feature_name']) &
                        (test_results['outcome'] == row['outcome']) &
                        (test_results['modality'] == row['modality']) &
                        (test_results['top_k'] == row['top_k'])
                    ]
                    test_or  = test_row['odds_ratio'].values[0] if len(test_row) > 0 else float('nan')
                    test_p   = test_row['odds_ratio_p_value'].values[0]    if len(test_row) > 0 else float('nan')
                    test_p_corrected = test_row['odds_ratio_p_value_bonferroni'].values[0] if len(test_row) > 0 else float('nan')

                    emb_auc = 0.5
                    if self.linear_probe_results:
                        key = (row['outcome'], row['modality'])
                        if key in self.linear_probe_results:
                            emb_auc = self.linear_probe_results[key]['all_metrics']['train']

                    pct_auc = 100 * (row['auc'] - 0.5) / (emb_auc - 0.5) if emb_auc > 0.5 else float('nan')
                    pct_str = f"({pct_auc:.0f}%)" if not np.isnan(pct_auc) else "(n/a)"
                    auc_str = f"{row['auc']:.3f}/{emb_auc:.3f} {pct_str}"

                    outcome_str = str(row['outcome'])[:19]
                    print(f"  {row['feature_name']:<15} {outcome_str:<20} "
                          f"{fmt_or(row['odds_ratio'], row['odds_ratio_p_value'], row['odds_ratio_p_value_bonferroni']):>22} "
                          f"{fmt_or(test_or, test_p, test_p_corrected):>22} "
                          f"{auc_str:>14} {int(row['c_f']):>8}")

                print(f"  {'─'*125}")
                print(f"  Bonferroni-adjusted p-values (N={n_tests}): *** p_adj<0.001  ** p_adj<0.01  * p_adj<0.05")

        if self.autointerp_results is not None and len(self.autointerp_results) > 0:
            print(f"\n[AutoInterp Results]")
            print(f"  Total interpretations: {len(self.autointerp_results)}")

            directions = [("Risk features (high OR)", "risk"), ("Protective features (low OR)", "protective")]
            if 'direction' not in self.autointerp_results:
                # Published descriptions belong to concepts, not to outcome-specific directions.
                directions = []
            for direction_label, direction_key in directions:
                subset = self.autointerp_results[self.autointerp_results['direction'] == direction_key]
                if subset.empty:
                    continue
                print(f"\n  [{direction_label}]")
                for _, row in subset.iterrows():
                    acc = row.get('test_accuracy', 0.0)
                    n = row.get('test_n_total', 0)
                    if acc <= 0.5:
                        interp = 'Undocumented AI concept'
                    else:
                        interp = row['interpretation']
                        if len(interp) > 60:
                            interp = interp[:60] + "..."
                    print(f"  {row['feature_name']:<15} {str(row['outcome']):<20} "
                          f"acc={acc:.0%} ({n} samples)  {interp}")

        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80 + "\n")

    def visualize(self, outcome=None, split=None, output_path=None, *, serve=False, open_browser=True, port=0, image_examples=None):
        """
        Build interactive concept maps of SAE features.

        Nodes are alive SAE features; edges connect highly correlated features.
        The generated, self-contained HTML supports pan, zoom, search, selectable
        outcomes, concept interpretations, and per-concept association wheels.
        Region overlays are intentionally omitted because they require a separate
        semantic naming step.

        Can be called after fit() for exploration without association results.

        Args:
            outcome: outcome selected when the HTML opens (default: first outcome in config)
            split: optional partition; defaults to train for fitted runs or the measured cohort
            image_examples: number of high-activation image examples per concept;
                defaults to visualization.image_examples (2). Set 0 to omit images.
            output_path: optional HTML path. Only valid when visualizing one
                modality/top-k pair. By default, files are written under output_dir.

        Returns:
            List of generated HTML paths.
        """
        from glyph.visualization import (
            build_interactive_concept_network,
            plot_concept_network,
        )

        if not self.sae_features:
            raise RuntimeError("Run fit() or measure() before visualize().")
        if split is None:
            first = next(iter(self.sae_features.values()))
            split = 'train' if 'train' in first else next(iter(first))

        if outcome is None and self.phewas_results is not None and self.cfg.data.outcome_cols:
            outcome = self.cfg.data.outcome_cols[0]

        embedding_map = self.cfg.data.get('embedding_map', {'image_path': 'resnet18'})
        if hasattr(embedding_map, '_content'):
            embedding_map = dict(embedding_map)

        output_dir = Path(self.cfg.data.output_dir)
        viz_cfg = self.cfg.visualization
        configurations = list(self.sae_alive_features.items())
        if output_path is not None and len(configurations) != 1:
            raise ValueError("output_path can only be used with one modality/top-k configuration")

        generated = []

        for modality in embedding_map.keys():
            for (mod, top_k), alive_mask in self.sae_alive_features.items():
                if mod != modality:
                    continue

                if viz_cfg.get('interactive', True):
                    model = (getattr(self, 'sae_models', None) or {}).get(modality)
                    published = getattr(model, 'selection_report', {}).get('pretrained')
                    concept_prefix = None
                    if published:
                        from glyph.zoo.registry import resolve
                        concept_prefix = {'image': 'I', 'findings': 'R', 'labs': 'L'}[resolve(published).modality]
                    path = Path(output_path) if output_path else (
                        output_dir / f"concept_map_{modality}_k{top_k}_{split}.html"
                    )
                    print(f"\nBuilding interactive concept map: {modality} k={top_k}, split={split}")
                    from glyph.visualization.image_examples import collect_image_examples
                    image_count=viz_cfg.get('image_examples',2) if image_examples is None else image_examples
                    frame=getattr(self,'_measurement_frame',None)
                    metadata_path=self.cfg.data.get('metadata')
                    if frame is None and image_count and metadata_path and Path(metadata_path).is_file():
                        frame=pd.read_csv(metadata_path,dtype=str)
                        if len(frame.columns) and frame.columns[0].startswith('Unnamed'):
                            frame=frame.drop(columns=[frame.columns[0]])
                        if 'sample_id' not in frame:frame=frame.rename(columns={frame.columns[0]:'sample_id'})
                    samples=((getattr(self,'embeddings',None) or {}).get(split) or {}).get('sample_ids',[])
                    gallery=collect_image_examples(self.sae_features[modality][split],
                        samples,frame,modality,
                        feature_indices=np.flatnonzero(alive_mask.cpu().numpy() if hasattr(alive_mask,'cpu') else alive_mask),
                        split=split,count=image_count,thumbnail_size=viz_cfg.get('thumbnail_size',256))
                    generated.append(build_interactive_concept_network(
                        sae_features=self.sae_features[modality],
                        alive_mask=alive_mask,
                        phewas_results=self.phewas_results,
                        output_path=path,
                        modality=modality,
                        top_k=top_k,
                        split=split,
                        initial_outcome=outcome,
                        correlation_percentile=viz_cfg.correlation_percentile,
                        bonferroni_alpha=viz_cfg.bonferroni_alpha,
                        seed=self.cfg.seed,
                        autointerp_results=self.autointerp_results,
                        umap_n_neighbors=viz_cfg.get('umap_n_neighbors', 20),
                        umap_min_dist=viz_cfg.get('umap_min_dist', 0.1),
                        umap_spread=viz_cfg.get('umap_spread', 1.0),
                        umap_repulsion_strength=viz_cfg.get('umap_repulsion_strength', 1.0),
                        view=viz_cfg.get('view', 'simple'),
                        wheel_label_limit=viz_cfg.get('wheel_label_limit', 20),
                        concept_prefix=concept_prefix,
                        image_examples=gallery,
                        outcome_names=viz_cfg.get("outcome_names", {}),
                    ))

                if viz_cfg.get('static_jpg', False) and self.phewas_results is not None:
                    jpg_path = output_dir / f"concept_network_{modality}_k{top_k}_{outcome}_{split}.jpg"
                    print(f"\nPlotting static concept network: {modality} k={top_k}, outcome={outcome}, split={split}")
                    plot_concept_network(
                        modality=modality,
                        top_k=top_k,
                        sae_features=self.sae_features[modality],
                        alive_mask=alive_mask,
                        phewas_results=self.phewas_results,
                        outcome=outcome,
                        output_path=jpg_path,
                        split=split,
                        correlation_percentile=viz_cfg.correlation_percentile,
                        bonferroni_alpha=viz_cfg.bonferroni_alpha,
                        figsize=tuple(viz_cfg.figsize),
                        dpi=viz_cfg.dpi,
                        layout_k=viz_cfg.layout_k,
                        seed=self.cfg.seed,
                        autointerp_results=self.autointerp_results,
                    )

        self._html_paths = generated
        if serve:
            self.serve(port=port,open_browser=open_browser)
        return generated

    def serve(self, *, port=0, open_browser=True):
        """Host previously generated maps; close the returned server to stop it."""
        from glyph.visualization.hosting import serve_html
        paths = getattr(self, '_html_paths', None) or sorted(Path(self.cfg.data.output_dir).glob('concept_map_*.html'))
        previous = getattr(self, 'viewer', None)
        if previous is not None:
            previous.close()
        self.viewer = serve_html(paths,port=port,open_browser=open_browser)
        return self.viewer

    def summary(self):
        """Print summary of results."""
        self._print_summary()
