from pathlib import Path
import json

import pytest
import torch

from glyph import Glyph
from glyph.config import load_config
from glyph.mechinterp.sae import SAE, TopKSAE, reconstruction_metrics


@pytest.fixture
def controlled_training(monkeypatch):
    calls = []
    scores = {}

    def train(**kwargs):
        k = kwargs['top_k']
        calls.append(k)
        model = TopKSAE(kwargs['input_dim'], kwargs['dict_size'], k)
        return model, torch.ones(kwargs['dict_size'], dtype=torch.bool)

    def metrics(model, embeddings, **kwargs):
        score = scores.get(model.top_k, 0.86)
        return {'r2': score, 'mse': 1 - score}

    monkeypatch.setattr('glyph.mechinterp.sae_trainer.train_sae', train)
    monkeypatch.setattr('glyph.mechinterp.sae.reconstruction_metrics', metrics)
    return scores, calls


def make_sae(candidates, **settings):
    overrides = [f'sae.top_ks={json.dumps(candidates)}', 'sae.matryoshka=[128]']
    overrides.extend(f'sae.selection.{key}={json.dumps(value)}' for key, value in settings.items())
    return SAE(load_config(overrides=overrides), input_dim=4, device='cpu')


def fit(sae):
    return sae.fit(torch.randn(20, 4), torch.randn(12, 4))


def test_sorted_greedy_search_and_per_target_allowances(controlled_training):
    scores, calls = controlled_training
    scores.update({5: 0.6, 10: 0.8, 20: 0.86, 24: 0.91, 35: 0.96})
    sae = make_sae([80, 20, 5, 10, 40, 20])
    selected = fit(sae)
    assert calls == [5, 10, 20, *range(21, 36)]
    assert list(selected) == [35]
    assert sae.selected_top_k == 35
    assert list(sae.alive_features) == [35]
    assert sae.selection_report['baseline_top_k'] == 20
    assert sae.selection_report['target_top_k_limits'] == [20, 40, 60]
    assert sae.selection_report['selected_target'] == 0.95
    assert (sae.embed(torch.randn(6, 4)) != 0).sum(dim=1).tolist() == [35] * 6


def test_rejects_90_percent_that_costs_more_than_double(controlled_training):
    scores, calls = controlled_training
    scores.update({5: 0.6, 10: 0.8, 20: 0.86, 41: 0.94})
    sae = make_sae([5, 10, 20, 80])
    fit(sae)
    assert max(calls) == 60
    assert sae.selected_top_k == 20
    assert sae.selection_report['selected_target'] == 0.85


def test_95_percent_can_qualify_without_intermediate_90_upgrade(controlled_training):
    scores, calls = controlled_training
    scores.update({5: 0.6, 10: 0.8, 20: 0.86, 41: 0.94, 60: 0.95})
    sae = make_sae([5, 10, 20, 80])
    fit(sae)
    assert sae.selected_top_k == 60
    assert sae.selection_report['selected_target'] == 0.95


def test_keeps_smallest_k_for_same_milestone(controlled_training):
    scores, _ = controlled_training
    scores.update({5: 0.6, 10: 0.8, 20: 0.86, 24: 0.90, 25: 0.949})
    sae = make_sae([5, 10, 20, 80])
    fit(sae)
    assert sae.selected_top_k == 24


def test_stops_immediately_if_first_candidate_reaches_highest_target(controlled_training):
    scores, calls = controlled_training
    scores[5] = 0.97
    sae = make_sae([5, 10, 20, 80])
    fit(sae)
    assert calls == [5]
    assert sae.selected_top_k == 5


def test_fallback_uses_best_reconstruction_and_warns(controlled_training):
    scores, calls = controlled_training
    scores.update({5: -3.0, 10: -1.5, 20: -2.0})
    sae = make_sae([5, 10, 20])
    with pytest.warns(UserWarning, match='below 0.85'):
        fit(sae)
    assert calls == [5, 10, 20]
    assert sae.selected_top_k == 10
    assert sae.selection_report['reached_minimum'] is False


def test_fixed_k_and_configurable_search_step(controlled_training):
    scores, calls = controlled_training
    scores[20] = 0.86
    fixed = make_sae([20])
    fit(fixed)
    assert calls == [20]
    calls.clear()
    flexible = make_sae([20, 80], lookahead_step=7)
    fit(flexible)
    assert calls == [20, 27, 34, 40, 41, 48, 55, 60]


def test_allowance_is_configurable(controlled_training):
    scores, calls = controlled_training
    scores.update({20: 0.86, 26: 0.92, 30: 0.95})
    sae = make_sae([20, 80], extra_sparsity_per_target=0.25)
    fit(sae)
    assert max(calls) == 30
    assert sae.selected_top_k == 30
    assert sae.selection_report['target_top_k_limits'] == [20, 25, 30]


def test_constant_validation_is_rejected_before_training(controlled_training):
    _, calls = controlled_training
    sae = make_sae([5, 10])
    with pytest.raises(ValueError, match='constant embeddings'):
        sae.fit(torch.randn(20, 4), torch.ones(12, 4))
    assert calls == []


def test_reconstruction_metrics_match_direct_calculation():
    model = TopKSAE(4, 8, 2)
    x = torch.randn(17, 4)
    with torch.no_grad():
        error = ((x - model(x)['x_hat']) ** 2).sum().item()
        variation = ((x - x.mean(dim=0)) ** 2).sum().item()
    result = reconstruction_metrics(model, x, batch_size=3)
    assert result['r2'] == pytest.approx(1 - error / variation, rel=1e-6)
    assert result['mse'] == pytest.approx(error / x.numel(), rel=1e-6)


def test_real_training_preserves_only_selected_model():
    torch.manual_seed(81)
    cfg = load_config(overrides=['sae.top_ks=[2,4]', 'sae.matryoshka=[8]',
                                 'sae.max_steps=5', 'sae.use_ghost_grads=false'])
    sae = SAE(cfg, input_dim=4, device='cpu')
    with pytest.warns(UserWarning, match='below 0.85'):
        sae.fit(torch.randn(60, 4), torch.randn(20, 4))
    assert len(sae.models) == len(sae.alive_features) == 1
    assert sae.selected_top_k in [2, 4]
    assert len(sae.selection_report['tried']) == 2
    assert (sae.embed(torch.randn(7, 4)) != 0).sum(dim=1).tolist() == [sae.selected_top_k] * 7


def test_selected_k_is_saved_and_used_by_csv_resume(tmp_path, controlled_training, monkeypatch):
    scores, _ = controlled_training
    scores.update({5: 0.6, 10: 0.8, 20: 0.86, 24: 0.90})
    sae = make_sae([5, 10, 20, 80])
    fit(sae)
    obj = Glyph.__new__(Glyph)
    obj.cfg = sae.cfg
    obj.cfg.data.output_dir = str(tmp_path)
    obj.sae_models = {'image': sae}
    obj.sae_features = {'image': {'train': torch.ones(4, 128)}}
    obj.sae_alive_features = {('image', 24): torch.ones(128, dtype=torch.bool)}
    monkeypatch.setattr(obj, '_save_filtered_concepts_csv', lambda: None)
    obj._save_sae_features()
    saved = json.loads((Path(obj.cfg.data.output_dir) / 'sae_selection.json').read_text())
    assert saved['image']['selected_top_k'] == 24
    import pandas as pd
    metadata = tmp_path / 'metadata.csv'
    pd.DataFrame({'sample_id': ['a', 'b'], 'outcome': [0, 1]}).to_csv(metadata, index=False)
    pd.DataFrame({'sample_id': ['a', 'b'], 'split': ['train', 'train'],
                  'Concept_0': [0.2, 0.5]}).to_csv(tmp_path / 'image_concepts_supported.csv', index=False)
    obj.cfg.data.metadata = str(metadata)
    obj.cfg.data.outcome_cols = ['outcome']
    obj.cfg.data.embedding_map = {'image': 'precomputed'}
    monkeypatch.setattr(obj, 'hypothesis_search', lambda **kwargs: obj)
    obj.hypothesis_search_from_csv(tmp_path)
    assert list(obj.sae_alive_features) == [('image', 24)]


def test_pipeline_uses_selected_model_for_every_split_and_probe(tmp_path, controlled_training, monkeypatch):
    from types import SimpleNamespace
    scores, _ = controlled_training
    scores.update({5: 0.6, 10: 0.8, 20: 0.86, 24: 0.90})
    cfg = load_config(overrides=[
        f'data.output_dir={tmp_path}', 'data.embedding_map={image:precomputed}',
        'data.outcome_cols=[outcome]', 'sae.top_ks=[5,10,20,80]',
        'sae.matryoshka=[128]', 'phewas.min_activations=1',
        '+skip_embeddings_probe=true',
    ])
    embeddings = {
        split: {'sample_ids': [f'{split}-{i}' for i in range(n)],
                'embeddings': {'image': torch.randn(n, 4)}}
        for split, n in [('train', 60), ('val', 20), ('test', 20)]
    }
    outcomes = {split: {'outcome': torch.arange(len(data['sample_ids'])).numpy() % 2}
                for split, data in embeddings.items()}
    obj = Glyph.__new__(Glyph)
    obj.cfg = cfg
    obj.outcome_types = {'outcome': 'binary'}
    obj.dataloaders = {}
    obj.embedding_extractor = SimpleNamespace(extract_all_splits=lambda *args, **kw: embeddings)
    monkeypatch.setattr(obj, '_extract_outcomes_aligned_with_embeddings', lambda: outcomes)
    probe_features = []

    def probe(**kwargs):
        probe_features.append(kwargs['embeddings_dict'])
        return {'all_metrics': {'train': 0.5, 'val': 0.5, 'test': 0.5}}

    monkeypatch.setattr('glyph.core.glyph.fit_linear_probe_from_embeddings', probe)
    obj.fit()
    assert list(obj.sae_alive_features) == [('image', 24)]
    assert list(obj.sae_linear_probe_results) == [('outcome', 'image', 24)]
    assert len(probe_features) == 1
    for split in embeddings:
        features = obj.sae_features['image'][split]
        assert (features != 0).sum(dim=1).eq(24).all()
        assert torch.equal(probe_features[0][split]['embeddings']['image'], features)
    saved = json.loads((Path(obj.cfg.data.output_dir) / 'sae_selection.json').read_text())
    assert saved['image']['selected_top_k'] == 24


def test_default_accepts_90_at_double_and_95_at_triple_baseline(controlled_training):
    scores, calls = controlled_training
    scores.update({20: .86, 40: .90, 60: .95})
    sae = make_sae([20, 80])
    fit(sae)
    assert sae.selection_report['target_top_k_limits'] == [20, 40, 60]
    assert sae.selected_top_k == 60
    assert sae.selection_report['selected_target'] == .95


def test_previous_half_allowance_remains_configurable(controlled_training):
    scores, calls = controlled_training
    scores.update({20: .86, 40: .902})
    sae = make_sae([20, 80], extra_sparsity_per_target=.5)
    fit(sae)
    assert sae.selection_report['target_top_k_limits'] == [20, 30, 40]
    assert sae.selected_top_k == 20
