import builtins
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import expit

from glyph import Glyph
from glyph.config import load_config
from glyph.phewas import PheWAS
from glyph.phewas.confounder_adjustment import (
    FMRConfounderAdjustment,
    encode_confounders,
)


@pytest.fixture
def estimator():
    pytest.importorskip('xgboost')
    return FMRConfounderAdjustment(n_splits=3, n_estimators=60)


def synthetic_data(n=1800):
    rng = np.random.default_rng(72)
    w = rng.normal(size=(n, 2))
    t = 0.9 * w[:, 0] + rng.normal(size=n)
    y = rng.binomial(1, expit(0.6 * t + 1.5 * w[:, 0] - 0.3 * w[:, 1]))
    return t, y.astype(float), w


def test_adjustment_attenuates_synthetic_confounding(estimator):
    t, y, w = synthetic_data()
    result = estimator.estimate(t, y, w)
    assert result['adjustment_status'] == 'ok'
    assert 1 < result['adjusted_odds_ratio'] < result['matched_unadjusted_odds_ratio']
    assert result['adjusted_odds_ratio_ci_lower'] < result['adjusted_odds_ratio']
    assert result['adjusted_odds_ratio_ci_upper'] > result['adjusted_odds_ratio']
    assert 0 <= result['adjusted_odds_ratio_p_value'] <= 1
    assert result['adjustment_method'] == 'fmr_orthogonal'
    assert result['adjustment_max_depth'] == 2
    assert result['adjustment_device'] == 'cpu'
    assert result['adjustment_n'] == len(y)


def test_missing_values_use_the_same_comparison_population(estimator):
    t, y, w = synthetic_data()
    t[0] = np.nan
    y[1:3] = [-1, np.nan]
    w[3, 0] = np.nan
    w[4, 1] = np.inf
    eligible = np.arange(5, len(y))
    result = estimator.estimate(t, y, w)
    reference = estimator.estimate(t[eligible], y[eligible], w[eligible])
    assert result['adjustment_n_excluded'] == 5
    assert result['adjustment_n_missing_outcome'] == 2
    assert result['adjustment_n_missing_feature'] == 1
    assert result['adjustment_n_missing_confounder'] == 2
    for col in ('adjusted_odds_ratio', 'matched_unadjusted_odds_ratio',
                'adjusted_odds_ratio_p_value', 'adjustment_n_events'):
        assert result[col] == pytest.approx(reference[col])


def test_unestimable_outcomes_have_explicit_status(estimator):
    t, y, w = synthetic_data(100)
    no_events = estimator.estimate(t, np.zeros_like(y), w)
    assert no_events['adjustment_status'] == 'insufficient_outcome_classes'
    assert np.isnan(no_events['adjusted_odds_ratio'])
    constant = estimator.estimate(np.zeros_like(t), y, w)
    assert constant['adjustment_status'] == 'constant_feature'
    y[:] = 0
    y[0] = 1
    rare = estimator.estimate(t, y, w)
    assert rare['adjustment_status'] == 'insufficient_events_for_folds'
    assert np.isnan(rare['adjusted_odds_ratio'])


def test_validation_does_not_silently_change_devices(estimator, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ValueError, match='available CUDA GPU'):
        FMRConfounderAdjustment(device='cuda')
    with pytest.raises(ValueError, match='adjustment_device'):
        FMRConfounderAdjustment(device='auto')
    t, y, w = synthetic_data(100)
    with pytest.raises(ValueError, match='aligned'):
        estimator.estimate(t, y, w[:-1])
    with pytest.raises(ValueError, match='binary'):
        estimator.estimate(t, np.linspace(0, 1, len(y)), w)


def test_missing_extra_explains_install_command(monkeypatch):
    real_import = builtins.__import__

    def missing_xgboost(name, *args, **kwargs):
        if name == 'xgboost':
            raise ImportError('not installed')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', missing_xgboost)
    with pytest.raises(ImportError, match=r'glyph-discovery\[confounders\]'):
        FMRConfounderAdjustment()


def test_gpu_backend_fallback_is_not_silent(monkeypatch):
    xgb = pytest.importorskip('xgboost')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(xgb, 'build_info', lambda: {'USE_CUDA': True})

    class CPUFallback:
        def __init__(self, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            return self

        def get_booster(self):
            return self

        def save_config(self):
            return json.dumps({'learner': {'generic_param': {'device': 'cpu'}}})

    monkeypatch.setattr(xgb, 'XGBClassifier', CPUFallback)
    estimator = FMRConfounderAdjustment(device='cuda', n_splits=3)
    with pytest.raises(RuntimeError, match='fell back to CPU'):
        estimator.estimate(*synthetic_data(100))


def test_categories_use_training_levels_and_keep_missing_rows():
    train = pd.DataFrame({'age': [20, 40, 60], 'sex': ['F', 'M', None]})
    test = pd.DataFrame({'age': [80, 70], 'sex': ['M', 'unseen']})
    matrices, names, levels = encode_confounders({'train': train, 'test': test})
    assert names == ['age', 'sex=F', 'sex=M']
    assert levels == {'sex': ['F', 'M']}
    np.testing.assert_array_equal(matrices['test'][0], [80, 0, 1])
    assert np.isnan(matrices['test'][1, 1:]).all()
    assert np.isnan(matrices['train'][2, 1:]).all()


@pytest.fixture
def pipeline(tmp_path):
    t, y, w = synthetic_data(480)
    ids = [f'study-{i}' for i in range(len(y))]
    metadata = pd.DataFrame({'sample_id': ids, 'age': w[:, 0], 'bmi': w[:, 1],
                             'outcome': y})
    path = tmp_path / 'metadata.csv'
    metadata.sample(frac=1, random_state=7).to_csv(path, index=False)
    obj = Glyph.__new__(Glyph)
    obj.cfg = load_config(overrides=[
        f'data.metadata={path}', f'data.output_dir={tmp_path / "output"}',
        'data.outcome_cols=[outcome]', 'phewas.min_activations=1',
        'phewas.confounder_adjustment.n_splits=3',
        'phewas.confounder_adjustment.n_estimators=20',
        'visualization.auto_generate=false',
    ])
    obj.embeddings = {'train': {'sample_ids': ids, 'embeddings': {}}}
    obj.sae_features = {'image': {'train': torch.tensor(t[:, None])}}
    obj.sae_alive_features = {('image', 2): torch.tensor([True])}
    obj.outcomes = {'train': {'outcome': y}}
    obj.linear_probe_results = None
    obj.sae_linear_probe_results = None
    obj.autointerp_results = None
    obj.phewas_results = None
    return obj, metadata


def test_ordinary_search_never_imports_optional_dependency(pipeline, monkeypatch):
    obj, _ = pipeline
    real_import = builtins.__import__

    def reject_xgboost(name, *args, **kwargs):
        if name == 'xgboost' or name.startswith('xgboost.'):
            raise AssertionError('Unadjusted search imported XGBoost')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', reject_xgboost)
    assert obj.hypothesis_search() is obj
    assert not any(c.startswith('adjusted_') for c in obj.phewas_results.columns)
    assert not (Path(obj.cfg.data.output_dir) / 'confounder_adjustment.json').exists()


def test_opt_in_alignment_outputs_and_corrections(pipeline):
    pytest.importorskip('xgboost')
    obj, metadata = pipeline
    matrix, manifest = obj._prepare_confounders(['age', 'bmi'])
    np.testing.assert_allclose(matrix['train'], metadata[['age', 'bmi']])
    assert manifest['covariates'] == ['age', 'bmi']
    obj.hypothesis_search(adjust_confounders=True, confounder_cols=['age', 'bmi'])
    row = obj.phewas_results.iloc[0]
    assert row['adjustment_status'] == 'ok'
    assert row['adjustment_max_depth'] == 2
    assert json.loads(row['adjustment_covariates']) == ['age', 'bmi']
    assert row['adjusted_odds_ratio_p_value_bonferroni'] == row['adjusted_odds_ratio_p_value']
    output = Path(obj.cfg.data.output_dir)
    saved = pd.read_pickle(output / 'hypotheses.pkl')
    pd.testing.assert_frame_equal(saved, obj.phewas_results)
    settings = json.loads((output / 'confounder_adjustment.json').read_text())
    assert settings['settings']['max_depth'] == 2
    assert settings['device'] == 'cpu'
    assert settings['method'] == 'fmr_orthogonal'
    assert settings['settings']['folds'] == 'stratified'
    assert settings['settings']['inner_seed_rule'] == 'seed + outer_fold_index + 1'
    obj.hypothesis_search()
    assert not (output / 'confounder_adjustment.json').exists()
    assert 'adjustment_method' not in pd.read_pickle(output / 'hypotheses.pkl')


def test_csv_resume_accepts_the_same_adjustment_flag(pipeline, tmp_path):
    pytest.importorskip('xgboost')
    obj, metadata = pipeline
    obj.cfg.data.embedding_map = {'image': 'precomputed'}
    obj.cfg.sae.top_ks = [2]
    values = obj.sae_features['image']['train'].numpy()[:, 0]
    csv = pd.DataFrame({'sample_id': metadata['sample_id'], 'split': 'train',
                        'Concept_0': values})
    csv.to_csv(tmp_path / 'image_concepts_supported.csv', index=False)
    result = obj.hypothesis_search_from_csv(
        tmp_path, adjust_confounders=True, confounder_cols=['age', 'bmi'])
    assert result is obj
    assert obj.phewas_results.iloc[0]['adjustment_status'] == 'ok'


def test_multisplit_adjustment_uses_each_splits_aligned_rows(estimator):
    t, y, w = synthetic_data(900)
    features = {'train': torch.tensor(t[:600, None]), 'test': torch.tensor(t[600:, None])}
    outcomes = {'train': {'outcome': y[:600]}, 'test': {'outcome': y[600:]}}
    cfg = load_config(overrides=['phewas.min_activations=1'])
    result = PheWAS(cfg).run_all(
        {'image': features}, {('image', 2): torch.tensor([True])}, outcomes,
        verbose=False, adjustment=estimator,
        confounders={'train': w[:600], 'test': w[600:]}, confounder_cols=['a', 'b'])
    assert set(result['adjustment_status']) == {'ok'}
    assert result.set_index('split')['adjustment_n'].to_dict() == {'train': 600, 'test': 300}


def test_missing_outcome_counts_and_raw_activation_conditioning():
    cfg = load_config(overrides=['phewas.min_activations=1'])
    features = torch.tensor([[0.0], [1.0], [10.0], [2.0]])
    outcomes = np.array([1.0, 0.0, -1.0, np.nan])
    result = PheWAS(cfg).run({'train': features}, {'train': {'outcome': outcomes}},
                             'image', 2, verbose=False)
    assert result.iloc[0]['p_y'] == 0.5
    assert result.iloc[0]['p_y_given_f'] == 0
    assert result.iloc[0]['c_yf'] == 0


def test_bad_opt_in_arguments_fail_early(pipeline):
    obj, metadata = pipeline
    with pytest.raises(ValueError, match='adjust_confounders=True'):
        obj.hypothesis_search(confounder_cols=['age'])
    with pytest.raises(ValueError, match='nonempty list'):
        obj.hypothesis_search(adjust_confounders=True)
    with pytest.raises(ValueError, match='missing from metadata'):
        obj.hypothesis_search(adjust_confounders=True, confounder_cols=['absent'])
    with pytest.raises(ValueError, match='Outcome columns'):
        obj.hypothesis_search(adjust_confounders=True, confounder_cols=['outcome'])
    metadata = pd.concat([metadata, metadata.iloc[[0]]])
    metadata.to_csv(obj.cfg.data.metadata, index=False)
    with pytest.raises(ValueError, match='unique, nonmissing'):
        obj.hypothesis_search(adjust_confounders=True, confounder_cols=['age'])


def test_adjusted_multiple_testing_keeps_failed_hypotheses_in_family():
    rows = pd.DataFrame({
        'split': ['train', 'train', 'test', 'test'],
        'modality': ['image'] * 4,
        'odds_ratio_p_value': [0.01] * 4,
        'adjusted_odds_ratio_p_value': [0.03, np.nan, 0.02, 0.01],
    })
    result = PheWAS._add_multiple_testing_corrections(rows)
    assert result['bonferroni_n_tests_global'].tolist() == [2] * 4
    np.testing.assert_allclose(result['adjusted_odds_ratio_p_value_bonferroni'],
                               [0.06, np.nan, 0.04, 0.02], equal_nan=True)


@pytest.mark.parametrize('effect', [-0.6, 0.0, 0.6])
def test_orthogonal_estimator_matches_frozen_research_reference(effect):
    import importlib.util
    reference = Path(__file__).parent / 'reference/fmr_orthogonal_20261023.py'
    spec = importlib.util.spec_from_file_location('research_fmr_reference', reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(237)
    w = rng.normal(size=(1000, 2))
    t = .8 * w[:, 0] + rng.normal(size=len(w))
    y = rng.binomial(1, expit(effect * t + w[:, 0] - .4 * w[:, 1]))
    settings = dict(n_splits=3, inner_splits=2, n_estimators=30, max_depth=2)
    expected = module.fmr_orthogonal(t, y, w, random_state=17, **settings)
    actual = FMRConfounderAdjustment(seed=17, **settings).estimate(t, y, w)
    assert actual['adjustment_status'] == 'ok'
    for suffix in ['', '_p_value', '_ci_lower', '_ci_upper']:
        assert np.isfinite(expected['dml_odds_ratio' + suffix])
        assert actual['adjusted_odds_ratio' + suffix] == pytest.approx(
            expected['dml_odds_ratio' + suffix], rel=1e-12, abs=1e-12)


def test_variance_weights_use_only_inner_oof_full_logits_within_each_outer_fold(monkeypatch):
    from scipy.special import expit
    from sklearn.model_selection import StratifiedKFold
    xgb = pytest.importorskip('xgboost')
    t, y, w = synthetic_data(480)
    margins = []
    weighted_fits = []
    original_predict = xgb.XGBClassifier.predict
    original_fit = xgb.XGBRegressor.fit

    def predict(self, values, *args, **kwargs):
        result = original_predict(self, values, *args, **kwargs)
        if kwargs.get('output_margin'):
            margins.append((values[:, 1:].copy(), np.asarray(result).copy()))
        return result

    def fit(self, values, target, *args, **kwargs):
        weights = kwargs.get('sample_weight')
        if weights is not None:
            fold = len(weighted_fits)
            train, heldout = list(StratifiedKFold(3, shuffle=True, random_state=42).split(w, y))[fold]
            np.testing.assert_array_equal(values, w[train])
            logits = {tuple(row): logit for rows, logits in margins[-2:] for row, logit in zip(rows, logits)}
            assert set(logits) == {tuple(row) for row in w[train]}
            assert not set(logits).intersection(tuple(row) for row in w[heldout])
            raw_logits = np.array([logits[tuple(row)] for row in values], dtype=float)
            expected = expit(raw_logits) * expit(-raw_logits)
            expected /= expected.mean()
            np.testing.assert_allclose(weights, expected, rtol=1e-12)
            assert np.mean(weights) == pytest.approx(1.)
            weighted_fits.append(fold)
        return original_fit(self, values, target, *args, **kwargs)

    monkeypatch.setattr(xgb.XGBClassifier, 'predict', predict)
    monkeypatch.setattr(xgb.XGBRegressor, 'fit', fit)
    result = FMRConfounderAdjustment(n_splits=3, inner_splits=2, n_estimators=8).estimate(t, y, w)
    assert result['adjustment_status'] == 'ok'
    assert weighted_fits == [0, 1, 2]
