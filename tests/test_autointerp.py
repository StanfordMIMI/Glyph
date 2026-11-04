"""
Integration test for AutoInterp with CheXpert data.

Tests the AutoInterp pipeline using:
- Real CheXpert metadata (with report text)
- Synthetic SAE features and PheWAS hypotheses
- Mocked Gemini API (no API key needed)

Modeled after chexpert2_test.py.
"""

import sys
import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

METADATA_PATH = "/data/akkumar/sae_exps/data/metadata.csv"
OUTPUT_DIR = "/tmp/autointerp_test_outputs"


def build_mock_genai():
    """Build a mock google.generativeai module with realistic responses."""
    mock_genai = MagicMock()

    call_idx = [0]

    fit_responses = [
        "This feature detects chest X-rays showing cardiomegaly with stable retrocardiac opacification.",
        "This feature identifies reports mentioning endotracheal tube placement and ventilator support.",
        "This feature captures findings related to pleural effusion and pulmonary edema.",
        "This feature detects cases with support devices such as PA catheters and pacemakers.",
        "This feature identifies reports with bilateral lung opacities and consolidation.",
    ]

    def generate_content(prompt, generation_config=None):
        resp = MagicMock()
        idx = call_idx[0]
        call_idx[0] += 1

        if 'MATCH_COUNT' in prompt:
            n = 3 + (idx % 5)
            lines = []
            for i in range(10):
                lines.append("YES" if i < n else "NO")
            lines.append(f"MATCH_COUNT: {n}")
            resp.text = "\n".join(lines)
        else:
            resp.text = fit_responses[idx % len(fit_responses)]

        return resp

    mock_model = MagicMock()
    mock_model.generate_content.side_effect = generate_content
    mock_genai.GenerativeModel.return_value = mock_model
    mock_genai.types.GenerationConfig.return_value = {}

    return mock_genai


def build_synthetic_data(metadata_path):
    """Build synthetic SAE features and PheWAS hypotheses from real metadata."""
    metadata = pd.read_csv(metadata_path)
    if metadata.columns[0].startswith('Unnamed'):
        metadata = metadata.drop(columns=[metadata.columns[0]])

    id_col = 'sample_id'
    split_col = 'split'

    splits = {}
    for split_name in ['train', 'val', 'test']:
        split_df = metadata[metadata[split_col] == split_name]
        splits[split_name] = split_df[id_col].tolist()

    print(f"  Split sizes: train={len(splits['train'])}, "
          f"val={len(splits['val'])}, test={len(splits['test'])}")

    n_features = 256
    top_k = 20
    modality = 'image_path'

    np.random.seed(42)
    sae_features = {}
    for split_name, sample_ids in splits.items():
        n = len(sample_ids)
        feats = torch.zeros(n, n_features)
        for f_idx in range(n_features):
            active = np.random.choice(n, size=max(1, n // 10), replace=False)
            feats[active, f_idx] = torch.randn(len(active)).abs() + 0.5
        sae_features[split_name] = feats

    # Build hypotheses matching PheWAS output format
    outcome_cols = ['no_finding', 'enlarged_cardiomediastinum']
    hyp_rows = []

    for outcome in outcome_cols:
        all_train_or = {}
        for f_idx in range(n_features):
            train_cf = (sae_features['train'][:, f_idx] > 0).sum().item()
            if train_cf < 25:
                continue
            odds_ratio = float(np.random.lognormal(0, 0.5))
            all_train_or[f_idx] = odds_ratio

        sorted_features = sorted(all_train_or.items(), key=lambda x: -x[1])
        rank_map = {f_idx: rank + 1 for rank, (f_idx, _) in enumerate(sorted_features)}

        for f_idx, odds_ratio in all_train_or.items():
            rank = rank_map[f_idx]
            for split_name in ['train', 'val', 'test']:
                n = len(splits[split_name])
                cf = (sae_features[split_name][:, f_idx] > 0).sum().item()
                if cf < 25:
                    continue
                hyp_rows.append({
                    'feature_idx': f_idx,
                    'feature_name': f'feature_{f_idx}',
                    'outcome': outcome,
                    'split': split_name,
                    'modality': modality,
                    'top_k': top_k,
                    'c_f': cf,
                    'p_y': 0.3,
                    'odds_ratio': odds_ratio if split_name == 'train'
                                  else float(np.random.lognormal(0, 0.5)),
                    'auc': 0.5 + np.random.random() * 0.3,
                    'train_odds_ratio_rank': rank,
                })

    hypotheses = pd.DataFrame(hyp_rows)
    print(f"  Hypotheses: {len(hypotheses)} rows, "
          f"{len(hypotheses[hypotheses['split']=='train'])} train")

    return splits, sae_features, hypotheses, modality, top_k


def main():
    print("=" * 80)
    print("AUTOINTERP INTEGRATION TEST")
    print("=" * 80)

    # Step 1: Verify metadata has report column
    print("\n[TEST 1] Checking metadata...")
    metadata = pd.read_csv(METADATA_PATH, nrows=5)
    assert 'report' in metadata.columns, "Missing 'report' column in metadata"
    assert 'sample_id' in metadata.columns, "Missing 'sample_id' column"
    report_text = str(metadata['report'].iloc[0])
    assert len(report_text) > 10, f"Report text too short: {report_text[:50]}"
    print(f"  report column present, first report: {len(report_text)} chars")
    print("  PASSED")

    # Step 2: Build synthetic data
    print("\n[TEST 2] Building synthetic SAE features and hypotheses...")
    splits, sae_features, hypotheses, modality, top_k = build_synthetic_data(METADATA_PATH)
    print("  PASSED")

    # Step 3: Create AutoInterp with mocked API
    print("\n[TEST 3] Creating AutoInterp with mocked Gemini API...")
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        'seed': 42,
        'device': 'cpu',
        'data': {
            'metadata': METADATA_PATH,
            'input_cols': ['image_path'],
            'outcome_cols': ['no_finding', 'enlarged_cardiomediastinum'],
            'output_dir': OUTPUT_DIR,
            'autointerp_text_map': {'image_path': 'report'},
        },
        'autointerp': {
            'gemini_api_key': 'mock-key-for-testing',
            'model': 'gemini-2.0-flash',
            'temperature': 0.7,
            'max_tokens': 1024,
            'top_n_hypotheses': 5,
            'top_pct_activating': 0.05,
            'n_fit_samples': 20,
            'n_test_samples': 10,
            'autointerp_text_map': {'image_path': 'report'},
            'fit_prompt': (
                'You are an expert scientist analyzing data samples. '
                'Below are {n_samples} data samples that all strongly activate '
                'a particular learned feature in a model.\n\n'
                'Your task: identify the common pattern or concept that these '
                'samples share.\n\n'
                'Here are the samples:\n{samples}\n\n'
                'Based on these samples, provide a concise interpretation '
                '(1-2 sentences) of what this feature detects.\n\n'
                'Your interpretation:'
            ),
            'test_prompt': (
                'You are an expert scientist. A learned feature has been '
                'interpreted as detecting the following pattern:\n\n'
                '"{interpretation}"\n\n'
                'Below are {n_samples} new data samples. For each sample, '
                'determine whether this interpretation applies (YES or NO). '
                'Then provide a count of how many samples match.\n\n'
                'Here are the samples:\n{samples}\n\n'
                'For each sample, write YES or NO on a separate line, then '
                'provide the total count of matching samples on the final line '
                'in the format: "MATCH_COUNT: X"'
            ),
        },
    })

    mock_genai = build_mock_genai()

    # Import and construct AutoInterp with mock
    from glyph.interpretation.autointerp import AutoInterp

    autointerp = AutoInterp.__new__(AutoInterp)
    autointerp.cfg = cfg
    autointerp.top_n = cfg.autointerp.top_n_hypotheses
    autointerp.top_pct = cfg.autointerp.top_pct_activating
    autointerp.n_fit = cfg.autointerp.n_fit_samples
    autointerp.n_test = cfg.autointerp.n_test_samples
    autointerp.seed = cfg.seed
    autointerp.genai = mock_genai
    autointerp.model_name = cfg.autointerp.model
    autointerp.temperature = cfg.autointerp.temperature
    autointerp.max_tokens = cfg.autointerp.max_tokens

    print("  AutoInterp created with mock API")
    print("  PASSED")

    # Step 4: Test _select_top_activating_samples
    print("\n[TEST 4] Testing sample selection logic...")
    rng = np.random.RandomState(42)
    selected = autointerp._select_top_activating_samples(
        feature_idx=0,
        sae_features=sae_features['train'],
        sample_ids=splits['train'],
        n_samples=20,
        rng=rng,
    )
    assert len(selected) > 0, "No samples selected"
    assert len(selected) <= 20, f"Too many samples: {len(selected)}"
    for sid in selected:
        assert sid in splits['train'], f"Selected sample {sid} not in train split"
    print(f"  Selected {len(selected)} samples from train split")
    print("  PASSED")

    # Step 5: Test _format_samples_text with real reports
    print("\n[TEST 5] Testing text formatting with real reports...")
    full_metadata = pd.read_csv(METADATA_PATH)
    if full_metadata.columns[0].startswith('Unnamed'):
        full_metadata = full_metadata.drop(columns=[full_metadata.columns[0]])

    formatted = autointerp._format_samples_text(
        sample_ids=selected[:3],
        metadata=full_metadata,
        text_col='report',
        id_col='sample_id',
    )
    assert 'Sample 1' in formatted
    assert 'Sample 2' in formatted
    assert 'Sample 3' in formatted
    assert len(formatted) > 100, "Formatted text suspiciously short"
    print(f"  Formatted text: {len(formatted)} chars for 3 samples")
    print(f"  Preview: {formatted[:120]}...")
    print("  PASSED")

    # Step 6: Test _parse_test_count
    print("\n[TEST 6] Testing response parsing...")
    assert autointerp._parse_test_count("YES\nNO\nYES\nMATCH_COUNT: 2", 3) == 2
    assert autointerp._parse_test_count("MATCH_COUNT: 5", 10) == 5
    assert autointerp._parse_test_count("MATCH_COUNT: 15", 10) == 10
    assert autointerp._parse_test_count("YES\nYES\nNO\nYES", 4) == 3
    assert autointerp._parse_test_count("NO\nNO\nNO", 3) == 0
    print("  All parsing cases passed")
    print("  PASSED")

    # Step 7: Run full run_for_modality
    print("\n[TEST 7] Running full AutoInterp for modality...")
    sae_features_wrapped = {modality: sae_features}

    results = autointerp.run_for_modality(
        modality=modality,
        sae_features=sae_features,
        sample_ids=splits,
        hypotheses_df=hypotheses,
        verbose=True,
    )

    assert len(results) > 0, "No interpretation results generated"
    print(f"\n  Results: {len(results)} interpretations")

    required_cols = [
        'feature_idx', 'feature_name', 'outcome', 'modality', 'top_k',
        'train_odds_ratio_rank', 'odds_ratio', 'interpretation',
        'fit_sample_ids', 'fit_sample_count',
        'test_sample_ids', 'test_sample_count',
        'test_match_count', 'test_score',
        'fit_prompt', 'fit_response', 'test_prompt', 'test_response',
    ]
    for col in required_cols:
        assert col in results.columns, f"Missing column: {col}"
    print("  All required columns present")

    for _, row in results.iterrows():
        assert row['fit_sample_count'] > 0, "fit_sample_count should be > 0"
        assert isinstance(row['fit_sample_ids'], list), "fit_sample_ids should be a list"
        assert isinstance(row['interpretation'], str), "interpretation should be string"
        assert len(row['interpretation']) > 0, "interpretation should not be empty"
        assert 0.0 <= row['test_score'] <= 1.0, f"test_score out of range: {row['test_score']}"
        assert len(row['fit_prompt']) > 0, "fit_prompt should not be empty"
        assert len(row['fit_response']) > 0, "fit_response should not be empty"
    print("  All row-level assertions passed")
    print("  PASSED")

    # Step 8: Run run_all and check saving
    print("\n[TEST 8] Running run_all with output saving...")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    all_results = autointerp.run_all(
        sae_features=sae_features_wrapped,
        sample_ids=splits,
        hypotheses=hypotheses,
        output_dir=OUTPUT_DIR,
        verbose=True,
    )

    assert len(all_results) > 0, "run_all returned empty results"

    saved_path = Path(OUTPUT_DIR) / "autointerp_results.pkl"
    assert saved_path.exists(), f"Results not saved to {saved_path}"
    print(f"  Results saved to: {saved_path} ({saved_path.stat().st_size / 1024:.1f} KB)")

    modality_saved = Path(OUTPUT_DIR) / "autointerp_results_image.pkl"
    assert modality_saved.exists(), f"Per-modality results not saved to {modality_saved}"
    print(f"  Per-modality results saved: {modality_saved}")

    reloaded = pd.read_pickle(saved_path)
    assert len(reloaded) == len(all_results), "Reloaded results count mismatch"
    print(f"  Reloaded {len(reloaded)} results successfully")
    print("  PASSED")

    # Step 9: Test missing API key error
    print("\n[TEST 9] Testing config validation...")
    cfg_no_key = OmegaConf.create({
        'seed': 42,
        'autointerp': {
            'gemini_api_key': None,
            'model': 'gemini-2.0-flash',
            'temperature': 0.7,
            'max_tokens': 1024,
            'top_n_hypotheses': 5,
            'top_pct_activating': 0.05,
            'n_fit_samples': 20,
            'n_test_samples': 10,
        },
    })
    try:
        with patch.dict('sys.modules', {'google.generativeai': MagicMock()}):
            AutoInterp(cfg_no_key)
        assert False, "Should have raised ValueError for missing API key"
    except ValueError as e:
        assert 'API key' in str(e)
        print(f"  Correctly raised: {e}")
    print("  PASSED")

    # Summary
    print("\n" + "=" * 80)
    print("AUTOINTERP TEST SUMMARY")
    print("=" * 80)
    print(f"  Total interpretations generated: {len(all_results)}")
    print(f"  Outcomes covered: {all_results['outcome'].unique().tolist()}")
    print()

    for _, row in all_results.iterrows():
        interp = row['interpretation']
        if len(interp) > 70:
            interp = interp[:70] + "..."
        print(f"  {row['feature_name']:<15} {str(row['outcome']):<30} "
              f"score={row['test_score']:.0%}  {interp}")

    print()
    print("  ALL TESTS PASSED")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
