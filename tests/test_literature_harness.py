import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT=Path(__file__).parents[1]/'skills/glyph-literature/scripts/compare.py'
spec=importlib.util.spec_from_file_location('literature_harness',SCRIPT)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


@pytest.fixture
def study():
    rng=np.random.default_rng(7);n=160;x=rng.normal(size=(n,4))
    y=(x[:,0]+rng.normal(size=n)>.1).astype(int)
    cohort=pd.DataFrame(dict(sample_id=[f's{i}' for i in range(n)],patient_id=[f'p{i}' for i in range(n)],
        split=['train']*80+['val']*40+['test']*40,y=y))
    glyph=pd.DataFrame(dict(sample_id=cohort.sample_id,g1=x[:,0]+x[:,1],g2=x[:,2]))
    markers=pd.DataFrame(dict(sample_id=cohort.sample_id,m1=x[:,0],m2=x[:,3]))
    manifest=dict(markers=[dict(column=name,definition='Direct measurement',source_url='https://example.org/source') for name in ['m1','m2']],
        glyph_features=['g1','g2'],glyph_selection_rule='Frozen training-only selection')
    return cohort,glyph,markers,manifest


def test_heldout_labels_cannot_change_fitting_or_tuning(study):
    cohort,glyph,markers,manifest=study
    first=runner.fit_models(cohort,glyph,markers,manifest,alphas=np.array([.001,.1,1.]))
    changed=cohort.copy();changed.loc[changed.split=='test','y']=1-changed.loc[changed.split=='test','y']
    second=runner.fit_models(changed,glyph,markers,manifest,alphas=np.array([.001,.1,1.]))
    pd.testing.assert_frame_equal(first[2],second[2])
    for arm in first[3]:
        for name in ['coefficients','mean','sd','alpha']:np.testing.assert_array_equal(first[3][arm][name],second[3][arm][name])
        np.testing.assert_array_equal(first[0][arm],second[0][arm])


def test_bootstrap_is_paired_and_deterministic(study):
    cohort,_,_,_=study
    y=cohort.y.to_numpy();prob=.2+.6*y
    pred=cohort[['sample_id','split','y']].copy()
    for arm in ['Literature','Glyph','Literature + Glyph']:pred[arm]=prob
    intervals,comparisons,audit=runner.paired_bootstrap(cohort,pred,100)
    assert np.all(intervals.auc==1)
    assert np.all(comparisons.delta_auc==0) and np.all(comparisons.p_value==1)
    assert audit[0]['replicates']==100
    pd.testing.assert_frame_equal(comparisons,runner.paired_bootstrap(cohort,pred,100)[1])


@pytest.mark.parametrize('fault',['ids','patient_overlap','source','frozen_selection'])
def test_invalid_studies_are_rejected(study,fault):
    cohort,glyph,markers,manifest=study
    if fault=='ids':glyph=glyph.iloc[1:]
    if fault=='patient_overlap':cohort.loc[100,'patient_id']=cohort.loc[0,'patient_id']
    if fault=='source':manifest['markers'][0]['source_url']=''
    if fault=='frozen_selection':manifest['glyph_features']=['missing']
    with pytest.raises(ValueError):runner.validate_inputs(cohort,glyph,markers,manifest)


def test_individual_family_has_equal_marker_counts_and_train_scaling(study):
    result=runner.individual_associations(*study)
    assert set(result.n_tests)=={4}
    assert result.groupby(['arm','split']).size().eq(2).all()
    assert result.converged.all()
    np.testing.assert_allclose(result.corrected_p_value,np.minimum(1,result.p_value*4))



def test_optional_comparison_adjustment_uses_orthogonal_method_and_depth_two(study):
    pytest.importorskip('xgboost')
    cohort, glyph, markers, manifest = study
    cohort = cohort.assign(age=np.linspace(20, 80, len(cohort)))
    manifest = dict(manifest, confounder_adjustment=dict(n_splits=2, inner_splits=2, n_estimators=8))
    results, metadata = runner.adjusted_associations(cohort, glyph, markers, manifest, ['age'])
    assert metadata['method'] == 'fmr_orthogonal'
    assert metadata['max_depth'] == 2 and metadata['device'] == 'cpu'
    assert metadata['settings']['inner_seed_rule'] == 'seed + outer_fold_index + 1'
    assert set(results.adjustment_method) == {'fmr_orthogonal'}
    assert set(results.adjustment_max_depth) == {2}
    assert set(results.n_tests) == {4}
    assert set(results.split) == {'train', 'test'}
    assert (results.adjustment_status == 'ok').all()


def test_six_variants_rank_only_training_rows_and_cap_total(study):
    cohort,glyph,markers,manifest=study
    rng=np.random.default_rng(11)
    for i in range(15):glyph[f'extra_{i}']=rng.normal(size=len(cohort))
    manifest=dict(manifest,glyph_features=list(glyph.columns[1:]))
    first,ranking=runner.comparison_feature_sets(cohort,glyph,markers,manifest)
    assert list(first)==['Literature','Literature top 10','Glyph','Glyph top 10','Literature + Glyph','Literature + Glyph to 10']
    assert len(first['Literature'])==len(first['Literature top 10'])==2
    assert len(first['Glyph'])==17 and len(first['Glyph top 10'])==10
    assert len(first['Literature + Glyph'])==19 and len(first['Literature + Glyph to 10'])==10
    assert set(first['Literature']).issubset(first['Literature + Glyph to 10'])
    changed=cohort.copy();changed.loc[changed.split.ne('train'),'y']=1-changed.loc[changed.split.ne('train'),'y']
    altered=glyph.copy();altered.loc[changed.split.ne('train'),altered.columns[1:]]=1e6
    second,other=runner.comparison_feature_sets(changed,altered,markers,manifest)
    assert first==second
    pd.testing.assert_frame_equal(ranking,other)


def test_wide_solver_matches_ridge_newton_objective():
    rng=np.random.default_rng(13);x=rng.normal(size=(90,12));x[:,3]=0
    y=(x[:,0]+rng.normal(size=len(x))>0).astype(float);alphas=np.array([.001,.1,1.])
    reference=runner.fit_batch(x,y,alphas)
    actual=runner.fit_batch_wide(x,y,alphas)
    np.testing.assert_allclose(actual[1],reference[1],atol=2e-6)
    np.testing.assert_allclose(actual[0],reference[0],atol=2e-6)
    assert actual[2]<1e-7 and np.all(actual[0][4]==0)


def test_six_arm_bootstrap_keeps_identical_predictions_identical(study):
    cohort,glyph,markers,manifest=study
    predictions,metrics,tuning,models=runner.fit_models(*study,alphas=np.array([.01,.1]))
    assert len(models)==6 and len(metrics)==18
    np.testing.assert_array_equal(predictions['Literature'],predictions['Literature top 10'])
    intervals,comparisons,audit=runner.paired_bootstrap(cohort,predictions,100)
    assert len(intervals)==6 and len(comparisons)==15 and comparisons.primary.sum()==1
    duplicate=comparisons.query("reference == 'Literature' and comparison == 'Literature top 10'").iloc[0]
    assert duplicate.delta_auc==0 and duplicate.p_value==1


def test_combined_cap_prioritizes_the_ranked_literature_baseline(study):
    cohort,glyph,markers,manifest=study
    rng=np.random.default_rng(55)
    for i in range(12):markers[f'extra_{i}']=rng.normal(size=len(cohort))
    manifest=dict(manifest,markers=[dict(column=c,definition='Measured value',source_url='https://example.org/source') for c in markers.columns if c!='sample_id'])
    arms,_=runner.comparison_feature_sets(cohort,glyph,markers,manifest)
    assert arms['Literature + Glyph to 10']==arms['Literature top 10']


def test_validation_stopping_restores_best_auc_and_loss_checkpoint():
    rng=np.random.default_rng(98);train=rng.normal(size=(55,6));val=rng.normal(size=(40,6))
    y=(train[:,0]+rng.normal(size=55)>.2).astype(float)
    # A deliberately shifted validation association forces an early optimum.
    y_val=(-val[:,0]+rng.normal(size=40)>0).astype(float)
    coefficients,diagnostics=runner.fit_validated_grid(train,y,val,y_val,np.full((1,6),.001),patience=2)
    info=diagnostics[0];trace=pd.DataFrame(json.loads(info['validation_trace']))
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y_val,runner.predict(val,coefficients).ravel())==pytest.approx(trace.validation_auc.max())
    assert info['stopped_early'] and info['best_iteration']<info['optimizer_iterations']


def test_combined_models_use_separate_blocks_and_keep_baseline_candidate(study):
    predictions,metrics,tuning,models=runner.fit_models(*study,alphas=np.array([.01,.1]))
    baseline=metrics.query("arm == 'Literature' and split == 'val'").iloc[0].auc
    for arm in ['Literature + Glyph','Literature + Glyph to 10']:
        assert models[arm]['regularization']=='separate_literature_glyph_blocks'
        assert models[arm]['literature_alpha']==models['Literature']['alpha']
        assert metrics.query('arm == @arm and split == "val"').iloc[0].auc>=baseline
        assert 'best_iteration' in models[arm]


def test_prediction_fits_are_invariant_to_manifest_column_order(study):
    cohort,glyph,markers,manifest=study
    one=runner.fit_models(*study,alphas=np.array([.01,.1]))
    reordered=dict(manifest,markers=list(reversed(manifest['markers'])),glyph_features=list(reversed(manifest['glyph_features'])))
    two=runner.fit_models(cohort,glyph,markers,reordered,alphas=np.array([.01,.1]))
    for arm in one[3]:
        np.testing.assert_array_equal(one[3][arm]['coefficients'],two[3][arm]['coefficients'])
        np.testing.assert_array_equal(one[0][arm],two[0][arm])


def test_pooled_cap_ranks_both_sources_using_training_only(study):
    cohort,glyph,markers,manifest=study
    rng=np.random.default_rng(51)
    for i in range(12):markers[f'extra_{i}']=rng.normal(size=len(cohort))
    manifest=dict(manifest,combined_top_n_strategy='pooled',markers=[dict(column=c,definition='Measured value',source_url='https://example.org/source') for c in markers.columns if c!='sample_id'])
    arms,ranking=runner.comparison_feature_sets(cohort,glyph,markers,manifest)
    expected=[(r.source,r.column) for r in ranking.head(10).itertuples()]
    assert arms['Literature + Glyph to 10']==expected
    changed=cohort.copy();changed.loc[changed.split.ne('train'),'y']=1-changed.loc[changed.split.ne('train'),'y']
    assert runner.comparison_feature_sets(changed,glyph,markers,manifest)[0]==arms
    predictions,metrics,_,models=runner.fit_models(cohort,glyph,markers,manifest,alphas=np.array([.01]))
    assert len(models['Literature + Glyph to 10']['feature_names'])==10
    assert np.isfinite(predictions['Literature + Glyph to 10']).all()
    with pytest.raises(ValueError,match='combined_top_n_strategy'):
        runner.comparison_feature_sets(cohort,glyph,markers,dict(manifest,combined_top_n_strategy='invalid'))


def test_constant_heldout_clinical_marker_is_not_estimable(study):
    cohort,glyph,markers,manifest=study
    markers.loc[cohort.split.eq('test'),'m2']=1000.  # Nonzero after training normalization.
    result=runner.individual_associations(cohort,glyph,markers,manifest)
    row=result.query("arm == 'Literature' and marker == 'm2' and split == 'test'").iloc[0]
    assert row.fit_status=='constant_marker_in_split' and not row.converged
    assert np.isnan(row.odds_ratio) and np.isnan(row.p_value)
    assert result.query("arm == 'Literature' and marker == 'm2' and split == 'train'").iloc[0].converged
    markers.loc[cohort.split.eq('train'),'m2']=0.
    row=runner.individual_associations(cohort,glyph,markers,manifest).query("arm == 'Literature' and marker == 'm2' and split == 'train'").iloc[0]
    assert row.fit_status=='no_training_variation' and not row.converged
