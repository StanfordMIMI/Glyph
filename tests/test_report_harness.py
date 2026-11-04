import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / 'skills/glyph-report/scripts/build_report.py'
spec = importlib.util.spec_from_file_location('report_harness', SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.fixture
def tables():
    concepts = pd.DataFrame(dict(concept_id=['a','b','c'], x=[.1,.8,.4], y=[.2,.8,.4], modality=['input']*3,
        interpretation=['size pattern','',''], interpretation_accuracy=[.9,np.nan,np.nan],
        interpretation_accuracy_context=['Original release']*3))
    associations = pd.DataFrame([dict(concept_id=i, outcome='outcome', split=s, odds_ratio=effect,
        odds_ratio_ci_lower=effect*.8, odds_ratio_ci_upper=effect*1.2,
        odds_ratio_p_value_bonferroni=p) for i,effect,p in [('a',2.,.001),('b',.5,.002),('c',1.,1.)]
        for s in ['train','test']])
    # A comparison effect that changes direction must not count as reproduction.
    associations.loc[(associations.concept_id=='b')&(associations.split=='test'),'odds_ratio']=2.
    return concepts, associations


def test_complete_space_and_exact_evidence(tmp_path,tables):
    concepts, associations = tables
    facts = report.build(concepts,associations,tmp_path/'brief','Example',{},preview_html=True)
    assert facts['n_concepts']==3
    assert facts['n_significant_pairs']==2
    assert facts['n_reproduced_hits']==1
    assert facts['zooms'][0]['odds_ratio']==2.
    assert facts['zooms'][0]['interpretation_accuracy_context']=='Original release'
    assert 'Original release accuracy 90%' in (tmp_path/'brief/report.md').read_text()
    assert len(pd.read_csv(tmp_path/'brief/concept_space.csv'))==3
    assert (tmp_path/'brief/report.pdf').read_bytes().startswith(b'%PDF')
    assert (tmp_path/'brief/report.html').is_file()
    assert json.loads((tmp_path/'brief/facts.json').read_text())['n_reproduced_hits']==1


def test_no_hits_is_a_valid_report(tmp_path,tables):
    concepts, associations=tables
    associations['odds_ratio_p_value_bonferroni']=1.
    facts=report.build(concepts,associations,tmp_path/'brief','No hits',{})
    assert facts['n_significant_concepts']==0 and facts['zooms']==[]
    assert not (tmp_path/'brief/zoom_1.png').exists()


def test_zero_zooms_and_missing_comparison(tmp_path,tables):
    concepts,associations=tables
    facts=report.build(concepts,associations[associations.split=='train'],tmp_path/'brief','Example',{},zoom_count=0)
    assert facts['zooms']==[] and facts['n_comparable_hits']==0


@pytest.mark.parametrize('issue', ['missing_correction','unknown_id','duplicate','bad_p','bad_accuracy'])
def test_rejects_ambiguous_or_invalid_evidence(tables,issue):
    concepts,associations=tables
    if issue=='missing_correction':associations=associations.drop(columns='odds_ratio_p_value_bonferroni')
    if issue=='unknown_id':associations.loc[0,'concept_id']='missing'
    if issue=='duplicate':associations=pd.concat([associations,associations.iloc[:1]])
    if issue=='bad_p':associations.loc[0,'odds_ratio_p_value_bonferroni']=-1
    if issue=='bad_accuracy':concepts.loc[0,'interpretation_accuracy']=90.
    with pytest.raises(ValueError):report.validate(concepts,associations)


def test_many_outcomes_have_a_legend_entry_for_every_significant_color():
    concepts = pd.DataFrame(dict(concept_id=[f'c{i}' for i in range(7)],
        x=np.linspace(0, 1, 7), y=np.linspace(1, 0, 7), modality=['image'] * 7))
    associations = pd.DataFrame([dict(concept_id=f'c{i}', outcome=f'digit{i}',
        split='train', odds_ratio=2., odds_ratio_p_value_bonferroni=.001) for i in range(6)])
    concepts, associations = report.validate(concepts, associations)
    colored, facts, colors, display = report.prepare(concepts, associations, 'train', 'test', .05, 0)
    assert facts['n_significant_concepts'] == 6
    assert set(colors)=={f'digit{i}' for i in range(6)}
    assert len(set(colors.values()))==6
    assert set(colored['_color']).issubset(set(colors.values()) | {report.GRAY})
    assert colored.loc[colored.concept_id == 'c6', '_color'].item() == report.GRAY


def test_image_panels_use_activation_ranks_without_heldout_outcome_selection(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from PIL import Image
    import torch
    from glyph import Glyph
    ids = ['t0', 't1', 't2', 't3', 'e0', 'e1', 'e2', 'e3']
    paths = []
    for i, sample in enumerate(ids):
        path = tmp_path / f'{sample}.png'
        Image.new('RGB', (12, 12), (20 * i, 30, 80)).save(path)
        paths.append(str(path))
    metadata = tmp_path / 'metadata.csv'
    pd.DataFrame(dict(sample_id=ids, image_path=paths)).to_csv(metadata, index=False)
    obj = SimpleNamespace(cfg=SimpleNamespace(data=SimpleNamespace(metadata=str(metadata))),
        embeddings={'train': {'sample_ids': ids[:4]}, 'test': {'sample_ids': ids[4:]}},
        sae_features={'image_path': {'train': torch.tensor([[4., 1.], [3., 2.], [2., 3.], [1., 4.]]),
                                     'test': torch.tensor([[1., 4.], [4., 1.], [2., 3.], [3., 2.]])}},
        outcomes={'train': {'outcome': np.array([0, 0, 1, 1])},
                  'test': {'outcome': np.array([0, 1, 0, 1])}})
    monkeypatch.setattr(Glyph, 'load_run', lambda *args, **kwargs: obj)
    concepts = pd.DataFrame(dict(concept_id=['image_path:2:0', 'image_path:2:1'],
        modality=['image_path'] * 2, x=[.2, .8], y=[.3, .7]))
    associations = pd.DataFrame([dict(concept_id=f'image_path:2:{i}', outcome='outcome',
        split=s, odds_ratio=2., odds_ratio_ci_lower=1.2, odds_ratio_ci_upper=3.,
        odds_ratio_p_value=.001 * (i + 1), odds_ratio_p_value_bonferroni=.002 * (i + 1))
        for i in range(2) for s in ['train', 'test']])
    examples = report.collect_image_examples('unused', associations, [], 'train', 'test')
    assert [s['sample_id'] for s in examples[0]['samples']] == ['t0', 't1', 'e1', 'e3']
    obj.outcomes['test']['outcome'] = 1 - obj.outcomes['test']['outcome']
    changed = associations.copy()
    changed.loc[changed.split == 'test', 'odds_ratio_p_value_bonferroni'] = [1., 0.]
    again = report.collect_image_examples('unused', changed, [], 'train', 'test')
    assert [[s['sample_id'] for s in e['samples']] for e in examples] == [
        [s['sample_id'] for s in e['samples']] for e in again]
    facts = report.build(concepts, associations, tmp_path / 'with_images', 'Image report', {}, image_run='unused', preview_html=True)
    assert len(facts['image_examples']) == 2
    anchors=facts['image_examples'][0]['scale_samples']
    assert [sample['sample_id'] for sample in anchors]==['t3','t2','t1','t0']
    assert facts['image_examples'][0]['distribution']['n_active']==4
    assert (tmp_path / 'with_images/activation_scale_1.png').is_file()
    assert (tmp_path / 'with_images/activation_tail_1.png').is_file()
    assert '<details><summary>Expand high-activation detail' in (tmp_path / 'with_images/report.html').read_text()
    assert (tmp_path / 'with_images/image_examples.png').is_file()
    assert (tmp_path / 'with_images/concept_1.png').is_file()
    assert (tmp_path / 'with_images/concept_2.png').is_file()
    assert (tmp_path / 'with_images/report.md').read_text().index('concept_1.png') < (tmp_path / 'with_images/report.md').read_text().index('concept_2.png')
    saved = json.loads((tmp_path / 'with_images/facts.json').read_text())
    assert '_path' not in saved['image_examples'][0]['samples'][0]
    assert len(saved['image_examples'][0]['samples'][0]['sha256']) == 64
    obj.sae_features['image_path']['test'].zero_()
    no_comparison = report.collect_image_examples('unused', associations, [], 'train', 'test')
    assert all(sample['split'] == 'train' for item in no_comparison for sample in item['samples'])
    obj.sae_features['image_path']['train'][:,0] = __import__('torch').tensor([0.,1.,2.,4.])
    frame = pd.read_csv(obj.cfg.data.metadata,dtype={'sample_id':str}).set_index('sample_id')
    scale,distribution = report.activation_scale_examples(obj,frame,'image_path',0,'outcome','train')
    assert scale[0]['inactive'] and scale[0]['activation']==0
    assert all(sample['activation']>0 for sample in scale[1:])
    assert scale[-1]['activation']==4  # Never replace the maximum just to force distinct images.
    assert sum(distribution['counts'])==4 and distribution['n_zero']==1



def test_extreme_confidence_limits_do_not_round_positive_bounds_to_zero():
    text = report.effect(dict(odds_ratio=119.45, odds_ratio_ci_lower=.000012,
                              odds_ratio_ci_upper=326241555.4))
    assert '0.00' not in text
    assert 'e-05' in text and 'e+08' in text


def test_report_map_uses_prevalence_area_instead_of_log_counts(tables):
    concepts,associations=tables
    concepts['activation_rate']=[.01,.05,.25]
    concepts['active_count']=[10,50,250]
    concepts,associations=report.validate(concepts,associations)
    plotted,*_=report.prepare(concepts,associations,'train','test',.05,0)
    fig,ax=report.plt.subplots()
    report.draw_map(ax,plotted)
    sizes=np.concatenate([collection.get_sizes() for collection in ax.collections])
    np.testing.assert_allclose(sorted(sizes),[4.,20.,100.])
    report.plt.close(fig)


def test_supplied_report_correlations_obey_the_same_degree_cap(tmp_path):
    concepts=pd.DataFrame(dict(concept_id=[str(i) for i in range(12)],x=np.linspace(0,1,12),y=np.linspace(0,1,12),modality=['image']*12,activation_rate=[.1]*12))
    associations=pd.DataFrame([dict(concept_id=str(i),outcome='y',split='train',odds_ratio=1.1,odds_ratio_p_value_bonferroni=1.) for i in range(12)])
    edges=[dict(source='0',target=str(i),weight=1-i/100) for i in range(1,12)]
    facts=report.build(concepts,associations,tmp_path/'links','Links',dict(correlation_edges=edges),zoom_count=0)
    assert len(facts['metadata']['correlation_edges'])==8
    assert len(edges)==11  # The caller's input was not rewritten.


def test_compact_report_omits_the_full_combined_row():
    data=pd.DataFrame(dict(split=['test']*3,arm=['Literature','Literature + Glyph','Literature + Glyph to 10']))
    assert report.prediction_rows(dict(intervals=data),'test').arm.tolist()==['Literature','Literature + Glyph to 10']


def test_paired_comparison_brackets_use_saved_pvalues_and_real_stars():
    intervals=pd.DataFrame(dict(split=['test']*5,arm=['Literature','Literature top 10','Glyph','Glyph top 10','Literature + Glyph to 10']))
    pairs=pd.DataFrame([dict(split='test',reference='Literature',comparison='Glyph top 10',p_value=.4),dict(split='test',reference='Literature',comparison='Literature + Glyph to 10',p_value=.002)])
    rows=report.displayed_auc_comparisons(dict(intervals=intervals,comparisons=pairs,manifest={}), 'test')
    assert [row['p_value'] for row in rows]==[.4,.002]
    assert report.significance_label(.4)=='n.s.' and report.significance_label(.002)=='**'
    assert report.significance_label(.0002)=='***'


def test_all_ten_outcomes_get_individual_report_colors():
    concepts=pd.DataFrame(dict(concept_id=[str(i) for i in range(10)],x=np.arange(10),y=np.arange(10),modality=['image']*10))
    associations=pd.DataFrame([dict(concept_id=str(i),outcome=f'digit{i}',split='train',odds_ratio=2.,odds_ratio_p_value_bonferroni=.001) for i in range(10)])
    concepts,associations=report.validate(concepts,associations)
    colored,_,colors,_=report.prepare(concepts,associations,'train','test',.05,0)
    assert len(colors)==10 and colored._color.nunique()==10
    assert not any('Other' in key for key in colors)


def test_ten_digit_legend_uses_an_ordered_numeric_color_strip():
    from glyph.visualization.colors import outcome_palette
    colors=outcome_palette([f'is_digit_{i}' for i in range(10)])
    fig,ax=report.plt.subplots()
    gray=report.plt.Line2D([],[],marker='o',linestyle='',color=report.GRAY,label='No significant estimate')
    report.map_legend(fig,ax,colors,{},[gray])
    assert len(ax.child_axes)==1
    assert ax.child_axes[0].get_xlabel()=='Digit'
    np.testing.assert_array_equal(ax.child_axes[0].get_xticks(),np.arange(10))
    report.plt.close(fig)


def test_report_can_focus_two_concepts_for_one_outcome(tables,tmp_path):
    concepts,associations=tables
    metadata=dict(focus_concepts=['b','a'],focus_outcome='outcome',concept_notes={'b':'Descriptive training example, unvalidated.'})
    facts=report.build(concepts,associations,tmp_path/'focused','Focused',metadata)
    assert [row['concept_id'] for row in facts['zooms']]==['b','a']
    assert all(row['outcome']=='outcome' for row in facts['zooms'])
    assert facts['zooms'][0]['annotation']=='Descriptive training example, unvalidated.'


def test_report_labels_failed_reconstruction_and_the_actual_capped_policy():
    lines=report.metric_lines({'quality':{'image':{'selection':{'selected_top_k':80,'reached_minimum':False,'r2_targets':[.85,.9]},'reconstruction':{'val':{'r2':.7},'test':{'r2':.69}}}}})
    assert 'below 0.85 target' in lines[0]
    assert 'pools literature and Glyph' in report.capped_model_description({'combined_top_n_strategy':'pooled'})
    assert 'retains up to 10' in report.capped_model_description({})


def test_large_forest_panels_select_on_training_only_and_keep_full_counts():
    rows=[]
    for arm in ['Literature','Glyph']:
        for i in range(20):
            for split in ['train','test']:
                rows.append(dict(arm=arm,marker=f'm{i:02}',split=split,p_value=(i+1)/1000,corrected_p_value=.01,odds_ratio=2.,ci_lower=1.5,ci_upper=2.5))
    table=pd.DataFrame(rows);study={'associations':table}
    shown=report.displayed_association_rows(study,'test')
    assert shown.groupby('arm').size().eq(6).all()
    assert set(shown.marker)=={f'm{i:02}' for i in range(6)}
    altered=table.copy();altered.loc[altered.split.eq('test'),'p_value']=np.linspace(0,1,40)
    assert report.displayed_association_rows({'associations':altered},'test')[['arm','marker']].equals(shown[['arm','marker']])
    assert all(v['reproduced']==20 for v in report.matched_reproduction(study,'test').values())
