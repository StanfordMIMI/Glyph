from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from glyph.extraction.labs import prepare_labs
from glyph.extraction.bundle import split_lab_name


@pytest.fixture
def bundle(tmp_path):
    stats=pd.DataFrame({'lab_id':[1,2], 'name':['Glucose (mg/dL)','Glucose (mmol/L)'],
                        'measurement_name':['Glucose','Glucose'], 'training_unit':['mg/dL','mmol/L'],
                        'is_numeric':[True,True],'retained':[True,True]})
    stats.to_json(tmp_path/'lab_metadata.json',orient='records')
    pd.DataFrame({'loinc_code':['123-4','123-4'],'lab_id':[1,2]}).to_csv(tmp_path/'lab_loinc_map.csv',index=False)
    (tmp_path/'bundle.json').write_text(json.dumps({'preprocessing_profile':'published-v1'}))
    return tmp_path


def test_balanced_unit_parsing():
    assert split_lab_name('Globulin (g/dL (calc))')==('Globulin','g/dL (calc)')
    assert split_lab_name('Albumin/Globulin Ratio ((calc))')==('Albumin/Globulin Ratio','(calc)')


def test_ambiguous_loinc_requires_unit_instead_of_overwriting(bundle):
    metadata=pd.DataFrame({'sample_id':['a'],'patient_id':[1],'date':['2024-07-01']})
    labs=pd.DataFrame({'patient_id':['1'],'date':['2024-05-01'],'loinc_code':['123-4'],'value':[5.]})
    with pytest.raises(ValueError,match='multiple model labs'):
        prepare_labs(metadata,labs,bundle)
    labs['unit']='mmol/L'
    result=prepare_labs(metadata,labs,bundle)
    assert result['a'][0]['lab_id']==2


def test_direct_lab_id_still_validates_units(bundle):
    metadata=pd.DataFrame({'sample_id':['a'],'patient_id':[1],'date':['2024-07-01']})
    labs=pd.DataFrame({'patient_id':[1],'date':['2024-05-01'],'lab_id':[1],'value':[5.],'unit':['mmol/L']})
    with pytest.raises(ValueError,match='differing from training'):
        prepare_labs(metadata,labs,bundle)
    labs=labs.drop(columns='unit')
    with pytest.raises(ValueError,match='needs measurement units'):
        prepare_labs(metadata,labs,bundle)
    assert prepare_labs(metadata,labs,bundle,require_units=False)['a']


def test_published_quarter_timestamp_and_most_recent_order(bundle):
    metadata=pd.DataFrame({'sample_id':['a'],'patient_id':[1],'date':['2024-07-01']})
    labs=pd.DataFrame({'patient_id':[1]*4,'date':['2024-02-01','2024-03-01','2024-05-01','2024-06-01'],
                       'lab_id':[1]*4,'value':[1.,2.,3.,4.],'unit':['mg/dL']*4})
    tokens,report=prepare_labs(metadata,labs,bundle,return_report=True)
    assert [x['Value'] for x in tokens['a']]==[4.,2.]
    assert tokens['a'][0]['years_prior']<tokens['a'][1]['years_prior']
    assert report['profile']=='published-v1'


def test_no_eligible_labs_are_explicit(bundle):
    metadata=pd.DataFrame({'sample_id':['a'],'patient_id':[1],'date':['2024-07-01']})
    labs=pd.DataFrame({'patient_id':[2],'date':['2024-05-01'],'lab_id':[1],'value':[5.],'unit':['mg/dL']})
    result,report=prepare_labs(metadata,labs,bundle,return_report=True)
    assert result=={'a':[]} and report['studies_without_eligible_labs']==1


def test_prepare_inputs_requires_explicit_missing_study_exclusion(bundle,tmp_path):
    from glyph.extraction.labs import prepare_lab_inputs
    metadata=pd.DataFrame({'sample_id':['a','b'],'patient_id':[1,2],'date':['2024-07-01']*2,'outcome':[0,1]})
    labs=pd.DataFrame({'patient_id':[1],'date':['2024-05-01'],'lab_id':[1],'value':[5.],'unit':['mg/dL']})
    with pytest.raises(ValueError,match='drop_empty=True'):
        prepare_lab_inputs(metadata,labs,bundle,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
    path=prepare_lab_inputs(metadata,labs,bundle,tmp_path/'prepared',drop_empty=True)
    output=pd.read_csv(path)
    assert output.sample_id.tolist()==['a']
    assert Path(output.labs_path.iloc[0]).is_file()
    report=json.loads((path.parent/'preparation_report.json').read_text())
    assert report['written_studies']==1 and report['studies_without_eligible_labs']==1


def test_published_preparation_keeps_raw_numeric_strings(bundle):
    metadata=pd.DataFrame({'sample_id':['a'],'patient_id':[1],'date':['2024-07-01']})
    labs=pd.DataFrame({'patient_id':[1,1],'date':['2024-03-01','2024-05-01'],
                       'lab_id':[1,1],'value':['-2.5','1e3'],'unit':['mg/dL','mg/dL']})
    tokens=prepare_labs(metadata,labs,bundle)
    assert [x['Value'] for x in tokens['a']]==['1e3','-2.5']


def test_missing_patient_id_cannot_pool_unidentified_people(bundle):
    metadata=pd.DataFrame({'sample_id':['a'],'patient_id':[None],'date':['2024-07-01']})
    labs=pd.DataFrame({'patient_id':[None],'date':['2024-05-01'],'lab_id':[1],'value':[5.],'unit':['mg/dL']})
    with pytest.raises(ValueError,match='nonmissing patient IDs'):
        prepare_labs(metadata,labs,bundle)


def test_published_encoder_is_stable_when_torch_fastpath_changes():
    import torch
    from glyph.extraction.labs import LabsEmbedder
    stats=pd.DataFrame({'lab_id':[1],'is_numeric':[True],'retained':[True],
                        'mean':[0.],'std':[1.],'categories':[[]]})
    model=LabsEmbedder(stats,pad_to=128).eval()
    # A legacy unparsed string creates an interior masked position.
    inputs=[[{'lab_id':1,'Value':1.,'years_prior':.1},
             {'lab_id':1,'Value':'-2.5','years_prior':.2},
             {'lab_id':1,'Value':3.,'years_prior':.3}],
            [{'lab_id':1,'Value':2.,'years_prior':.1}]]
    before=torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(True)
        with torch.inference_mode(): expected=model(inputs)
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.inference_mode(): actual=model(inputs)
        torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-6)
    finally:
        torch.backends.mha.set_fastpath_enabled(before)


def test_explicit_id_can_disambiguate_only_selected_measurements(bundle):
    metadata=pd.DataFrame({'sample_id':['a'],'patient_id':[1],'date':['2024-07-01']})
    labs=pd.DataFrame({'patient_id':[1,1],'date':['2024-03-01','2024-05-01'],
                       'loinc_code':['123-4','123-4'],'lab_id':[1,None],
                       'unit':['mg/dL','mmol/L'],'value':['5.','6.']})
    tokens=prepare_labs(metadata,labs,bundle)
    assert [v['lab_id'] for v in tokens['a']]==[2,1]


def test_local_mapping_is_applied_during_preparation(bundle):
    mapping=pd.read_csv(bundle/'lab_loinc_map.csv')
    mapping['selected_code']=['STANFORD_LAB//glucose','123-4']
    mapping['selected_code_system']=['STANFORD_LAB','LOINC']
    mapping.loc[0,'loinc_code']=None
    mapping.to_csv(bundle/'lab_loinc_map.csv',index=False)
    meta=pd.DataFrame({'sample_id':['a'],'patient_id':[1],'date':['2024-07-01']})
    labs=pd.DataFrame({'patient_id':[1],'date':['2024-05-01'],'code':['STANFORD_LAB//glucose'], 'value':['5.'], 'unit':['mg/dL']})
    assert prepare_labs(meta,labs,bundle)['a'][0]['lab_id']==1
