import json

import pandas as pd
import pytest
import torch
import yaml

from glyph.zoo import pretrained
from glyph.zoo.registry import ZooEntry


@pytest.fixture
def published_bundle(tmp_path, monkeypatch):
    entry = ZooEntry('test@1', 'image', 'test encoder', 3, 5, 2,
                     'sae_test.ckpt', 'owner/model', 'a' * 40)
    model = dict(modality='image', encoder='test encoder', input_dim=3,
                 n_concepts=5, top_k=2, activation='relu_topk', weights_file=entry.weights)
    config = dict(model_type='glyph_sparse_autoencoder', format_version=1,
                  normalization_file='normalization.yaml',
                  interpretations_file='interpretations.csv', models={entry.name: model})
    files = tmp_path / 'published';files.mkdir()
    def write_config():
        (files / 'config.json').write_text(json.dumps(config))
    write_config()
    weights = dict(W_enc=torch.arange(15, dtype=torch.float32).reshape(3, 5) / 10,
                   b_enc=torch.zeros(5))
    torch.save(weights, files / entry.weights)
    (files / 'normalization.yaml').write_text(yaml.safe_dump({'image': dict(mean=[0., 0., 0.], std=1., dim_scale=1.)}))
    pd.DataFrame({'inputs': ['image'], 'feature_name': ['Concept_0'],
                  'extracted_interpretation': ['Synthetic test feature']}).to_csv(files / 'interpretations.csv', index=False)
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        assert kwargs['repo_id'] == entry.repo and kwargs['revision'] == entry.revision
        assert not kwargs.get('force_download', False)
        return str(files / kwargs['filename'])
    monkeypatch.setattr('huggingface_hub.hf_hub_download', download)
    monkeypatch.setattr(pretrained, 'resolve', lambda name: entry)
    monkeypatch.setattr(pretrained, 'ASSET_DIR', tmp_path / 'no-packaged-assets')
    return entry, config, files, calls, write_config, weights


def test_configuration_is_used_before_checkpoint_and_encoding_is_unchanged(published_bundle, tmp_path):
    entry, config, files, calls, _, weights = published_bundle
    bundle = pretrained.load_pretrained(entry.name, cache_dir=tmp_path / 'cache')
    assert [c['filename'] for c in calls] == ['config.json', entry.weights, 'normalization.yaml', 'interpretations.csv']
    x = torch.tensor([[1., 2., 3.], [3., 1., 2.]])
    with pytest.warns(UserWarning, match='embedding magnitude'):
        actual = bundle.encode(x, strict=False)
    dense = torch.relu(x @ weights['W_enc'] + weights['b_enc'])
    values, indices = dense.topk(entry.top_k, dim=-1)
    expected = torch.zeros_like(dense).scatter_(-1, indices, values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('field,value', [('input_dim', 4), ('top_k', 3), ('weights_file', '../other.ckpt'), ('activation', 'sigmoid')])
def test_incompatible_config_is_rejected_before_downloading_weights(published_bundle, tmp_path, field, value):
    entry, config, _, calls, write, _ = published_bundle
    config['models'][entry.name][field] = value;write()
    with pytest.raises(ValueError, match=field):
        pretrained.load_pretrained(entry.name, cache_dir=tmp_path / 'cache')
    assert [c['filename'] for c in calls] == ['config.json']


def test_unsupported_configuration_version_fails_clearly(published_bundle, tmp_path):
    entry, config, _, calls, write, _ = published_bundle
    config['format_version'] = 9;write()
    with pytest.raises(ValueError, match='Unsupported pretrained Glyph configuration'):
        pretrained.load_pretrained(entry.name, cache_dir=tmp_path / 'cache')
    assert len(calls) == 1


def test_checkpoint_shape_is_checked_against_published_configuration(published_bundle, tmp_path):
    entry, _, files, _, _, _ = published_bundle
    torch.save({'W_enc': torch.zeros(3, 4), 'b_enc': torch.zeros(4)}, files / entry.weights)
    with pytest.raises(ValueError, match='checkpoint dimensions'):
        pretrained.load_pretrained(entry.name, cache_dir=tmp_path / 'cache')
