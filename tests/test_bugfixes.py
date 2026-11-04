import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from glyph import Glyph
from glyph.config import load_config
from glyph.metrics.linear_probe import run_linear_probe, fit_linear_probe_from_embeddings
from glyph.phewas import PheWAS


def probe_data():
    rng = np.random.default_rng(123)
    return [(rng.normal(size=(30, 3)).astype(np.float32), np.arange(30, dtype=np.float32) % 2) for _ in range(3)]


def test_probe_checkpoint_is_a_snapshot(monkeypatch):
    scores = iter([.9, .8, .7])
    monkeypatch.setattr('glyph.metrics.linear_probe.compute_auc', lambda *a: next(scores, .9))
    states = []
    original_step = torch.optim.AdamW.step
    def step(optimizer, *a, **kw):
        result = original_step(optimizer, *a, **kw)
        states.append([v.detach().clone() for group in optimizer.param_groups for v in group['params']])
        return result
    monkeypatch.setattr(torch.optim.AdamW, 'step', step)
    data = probe_data()
    result = run_linear_probe(*[v for pair in data for v in pair], lr_grid=[.1], wd_grid=[0.], max_epochs=3, verbose=False)
    for actual, expected in zip(result['model'].parameters(), states[0]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert any(not torch.equal(a,b) for a,b in zip(states[0],states[-1]))


def test_regression_all_negative_r2_still_selects_model():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(30, 2))
    y = 100 + rng.normal(scale=.1, size=30)
    result = run_linear_probe(X,y,X,y,X,y,outcome_type='regression',lr_grid=[.001],wd_grid=[0.],max_epochs=2,verbose=False)
    assert result['val_metric'] < -1 and result['model'] is not None


def test_probe_missing_rows_are_excluded_and_predictions_keep_row_alignment():
    data = probe_data()
    for X,y in data:
        X[0,0] = np.nan; X[1,1] = np.inf; y[2] = np.nan; y[3] = -1
    result = run_linear_probe(*[v for pair in data for v in pair],lr_grid=[.01],wd_grid=[0.],max_epochs=3,verbose=False)
    assert result['eligible_counts'] == {'train':26,'val':26,'test':26}
    assert np.isnan(result['test_preds'][:2]).all()
    assert np.isfinite(result['test_preds'][2:]).all()
    assert all(torch.isfinite(p).all() for p in result['model'].parameters())


def test_wrapper_marks_unestimable_probe_without_crashing_pipeline():
    embeddings = {s:{'embeddings':{'image':torch.randn(8,2)}} for s in ['train','val','test']}
    labels = {s:np.zeros(8) for s in embeddings}
    result = fit_linear_probe_from_embeddings(embeddings,labels,modality='image',max_epochs=2,verbose=False)
    assert result['status']=='unestimable' and result['model'] is None
    assert 'both classes' in result['reason']


def test_csv_resume_preserves_original_ids_through_real_phewas(tmp_path):
    ids=[f's{i}' for i in range(120)]
    splits=['train']*60+['val']*30+['test']*30
    metadata=pd.DataFrame({'sample_id':ids,'outcome':np.arange(120)%2})
    metadata.to_csv(tmp_path/'metadata.csv',index=False)
    rng=np.random.default_rng(1)
    pd.DataFrame({'sample_id':ids,'split':splits,'Concept_98':rng.uniform(.1,2,120),
                  'Concept_7':rng.uniform(.1,2,120)}).to_csv(tmp_path/'image_concepts_supported.csv',index=False)
    obj=Glyph.__new__(Glyph)
    obj.cfg=load_config(overrides=[f'data.metadata={tmp_path}/metadata.csv',f'data.output_dir={tmp_path}',
                                  'data.embedding_map={image:precomputed}','data.outcome_cols=[outcome]',
                                  'sae.top_ks=[2]','phewas.min_activations=1','visualization.auto_generate=false'])
    obj.linear_probe_results=None;obj.sae_linear_probe_results=None;obj.autointerp_results=None
    obj.hypothesis_search_from_csv(tmp_path)
    assert set(obj.phewas_results.feature_idx)=={7,98}
    assert set(obj.phewas_results.feature_name)=={'feature_7','feature_98'}
    assert obj.sae_features['image']['train'].shape==(60,99)
    assert torch.where(obj.sae_alive_features[('image',2)])[0].tolist()==[7,98]


def test_empty_discovery_family_does_not_create_significance():
    frame=pd.DataFrame({'split':['test'],'modality':['image'],'odds_ratio_p_value':[.01]})
    corrected=PheWAS._add_multiple_testing_corrections(frame)
    assert corrected.odds_ratio_p_value_bonferroni.isna().all()
    assert not corrected.bonferroni_significant.any()


def test_cache_changes_with_metadata_input_assets_and_model_kwargs(tmp_path,monkeypatch):
    from glyph.models.embedding_extractor import EmbeddingExtractor
    from omegaconf import OmegaConf
    metadata=tmp_path/'metadata.csv';asset=tmp_path/'embeddings.bin'
    metadata.write_text('sample_id,scan\na,file-a\n');asset.write_bytes(b'first')
    cfg=load_config(overrides=[f'data.metadata={metadata}',f'data.cache_dir={tmp_path}/cache'])
    cfg.data.embedding_model_kwargs={'scan':{'asset_path':str(asset)}}
    monkeypatch.setattr(EmbeddingExtractor,'_load_models',lambda self:{})
    obj=EmbeddingExtractor(cfg,{'scan':'precomputed'})
    first=obj._get_cache_path('train')
    metadata.write_text('sample_id,scan\nb,file-a\n');second=obj._get_cache_path('train')
    asset.write_bytes(b'second');third=obj._get_cache_path('train')
    OmegaConf.update(cfg,'data.embedding_model_kwargs.scan.extra','changed',force_add=True);fourth=obj._get_cache_path('train')
    assert len({first,second,third,fourth})==4


def test_static_plot_uses_global_correction_and_separate_modalities(tmp_path,monkeypatch):
    from glyph.visualization.concept_network import plot_concept_network
    import networkx as nx
    frame=pd.DataFrame([{'feature_idx':0,'outcome':o,'split':'train','modality':m,'top_k':1,
                         'odds_ratio':2.,'odds_ratio_p_value':.02,'c_f':3}
                        for m in ['image','labs'] for o in ['a','b']])
    captured=[];original=nx.draw_networkx_nodes
    def nodes(graph,*a,**kw):
        captured.append([graph.nodes[n]['significant'] for n in graph])
        return original(graph,*a,**kw)
    monkeypatch.setattr(nx,'draw_networkx_nodes',nodes)
    plot_concept_network({'train':torch.tensor([[1.],[.5],[0.]])},torch.tensor([True]),frame,
                         'a',tmp_path/'plot.jpg',modality='labs',top_k=1)
    assert captured==[[False]] # .02 * four discovery hypotheses > .05
    with pytest.raises(ValueError,match='Specify modality'):
        plot_concept_network({'train':torch.ones(3,1)},torch.tensor([True]),frame,'a',tmp_path/'bad.jpg')


def test_normalization_preserves_other_rows_when_one_feature_value_is_missing():
    phewas=PheWAS(load_config())
    result=phewas.normalize_features({'train':torch.tensor([[1.],[float('nan')],[3.]]),
                                      'test':torch.tensor([[2.],[4.]])})
    assert np.isfinite(result['train'][[0,2]]).all()
    assert np.isfinite(result['test']).all()


def test_summary_matches_test_effects_within_the_same_modality(capsys):
    obj=Glyph.__new__(Glyph);obj.cfg=load_config(overrides=['data.outcome_cols=[o]'])
    obj.embeddings=None;obj.linear_probe_results=None;obj.sae_alive_features=None
    obj.sae_linear_probe_results=None;obj.autointerp_results=None
    obj.outcomes={'train':{'o':np.array([0.,1.])}}
    rows=[]
    for modality,train_or,test_or in [('image',1.5,5.),('labs',1.2,9.)]:
        for split,value in [('train',train_or),('test',test_or)]:
            rows.append({'feature_idx':0,'feature_name':'feature_0','outcome':'o','modality':modality,
                         'top_k':2,'split':split,'odds_ratio':value,'odds_ratio_p_value':.001,
                         'odds_ratio_p_value_bonferroni':.004,'auc':.6,'c_f':30})
    obj.phewas_results=pd.DataFrame(rows);obj._print_summary()
    lines=[line for line in capsys.readouterr().out.splitlines() if 'feature_0' in line]
    assert len(lines)==2
    assert '1.50' in lines[0] and '5.00' in lines[0]
    assert '1.20' in lines[1] and '9.00' in lines[1]


def test_pretrained_registry_selects_numeric_version(monkeypatch):
    from dataclasses import replace
    from glyph.zoo.registry import ZOO,resolve
    entry=resolve('ct-merlin')
    monkeypatch.setitem(ZOO,'ct-merlin@2',replace(entry,name='ct-merlin@2'))
    monkeypatch.setitem(ZOO,'ct-merlin@10',replace(entry,name='ct-merlin@10'))
    assert resolve('ct-merlin').name=='ct-merlin@10'


def test_cache_never_reuses_a_different_dataloader_subset(tmp_path,monkeypatch):
    from glyph.models.embedding_extractor import EmbeddingExtractor
    calls=[]
    class Model:
        def embed_batch(self, values):
            calls.extend(values)
            return torch.tensor([[float(v),1.] for v in values])
    class Loader:
        def __init__(self,key,value):
            self.key,self.value=key,value
            self.dataset=SimpleNamespace(metadata=pd.DataFrame({'sample_id':[key],'scan':[value]}))
        def __iter__(self):
            yield {'sample_id':[self.key],'inputs':{'scan':[self.value]}}
    metadata=tmp_path/'metadata.csv';metadata.write_text('sample_id,scan\na,1\nb,2\n')
    cfg=load_config(overrides=[f'data.metadata={metadata}',f'data.cache_dir={tmp_path}/cache'])
    monkeypatch.setattr(EmbeddingExtractor,'_load_models',lambda self:{'scan':Model()})
    obj=EmbeddingExtractor(cfg,{'scan':'fake'})
    a=obj.extract_embeddings(Loader('a',1),show_progress=False)
    b=obj.extract_embeddings(Loader('b',2),show_progress=False)
    c=obj.extract_embeddings(Loader('b',2),show_progress=False)
    assert a['sample_ids']==['a'] and b['sample_ids']==c['sample_ids']==['b']
    assert calls==[1,2]


@pytest.mark.parametrize("provider", ["vertex", "gemini_api"])
def test_autointerp_uses_current_sdk_for_both_providers(monkeypatch, provider):
    import sys
    from unittest.mock import Mock
    from glyph.interpretation.autointerp import AutoInterp
    client = Mock()
    client.models.generate_content.return_value.text = "interpretation"
    constructor = Mock(return_value=client)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=SimpleNamespace(Client=constructor)))
    llm = SimpleNamespace(provider=provider, model="test-model", temperature=0., max_tokens=20,
                          vertex_project="test-project", vertex_location="us-central1", gemini_api_key="test-key")
    cfg = SimpleNamespace(seed=1, autointerp=SimpleNamespace(top_n_hypotheses=5, bottom_n_hypotheses=0,
                          top_pct_activating=.05, n_fit_samples=20, n_test_samples=10, llm=llm))
    obj = AutoInterp(cfg)
    expected = dict(vertexai=True, project="test-project", location="us-central1") if provider == "vertex" else dict(api_key="test-key")
    constructor.assert_called_once_with(**expected)
    assert obj._call_gemini("synthetic test prompt") == "interpretation"
    client.models.generate_content.assert_called_once_with(model="test-model", contents="synthetic test prompt",
                                                          config=dict(temperature=0., max_output_tokens=20))


@pytest.mark.parametrize('bad_split', ['external', None])
def test_unsupported_metadata_splits_are_rejected_before_rows_are_dropped(tmp_path, bad_split):
    from glyph.dataloader import GlyphDataset
    path = tmp_path / 'metadata.csv'
    pd.DataFrame(dict(sample_id=['a', 'b', 'c'], split=['train', 'test', bad_split])).to_csv(path, index=False)
    cfg = load_config(overrides=[f'data.metadata={path}', 'data.input_cols=[scan]', 'data.outcome_cols=[]'])
    with pytest.raises(ValueError, match='Metadata split labels'):
        GlyphDataset(cfg, split='train', allow_missing_inputs=True)
