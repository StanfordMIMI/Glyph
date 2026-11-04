from types import SimpleNamespace

import pytest
from glyph.interpretation.autointerp import AutoInterp
from glyph.interpretation.settings import ready


def parser():return AutoInterp.__new__(AutoInterp)


@pytest.mark.parametrize('response',['1. 1','1. 1\n1. 0','1. 1\n3. 0','Maybe 1 or 0.','[1,true]'])
def test_incomplete_predictions_are_not_scored_as_real_accuracy(response):
    result=parser()._parse_discriminative_predictions(response,2,[1,0])
    assert result['accuracy'] is None and result['status']=='invalid_or_incomplete_predictions'
    assert result['predictions']==[]


def test_numbered_and_json_responses_have_real_balanced_scores():
    for response in ['* PREDICTIONS:\n1: 1\n2: 0\n3: 0\n4: 1','[1,0,0,1]']:
        result=parser()._parse_discriminative_predictions(response,4,[1,0,1,0])
        assert result['accuracy']==.5 and result['balanced_accuracy']==.5
        assert result['correct']==2


def test_environment_credentials_enable_optional_calls_without_mutating_config(monkeypatch):
    monkeypatch.delenv('GOOGLE_API_KEY',raising=False);monkeypatch.delenv('GEMINI_API_KEY',raising=False)
    llm=SimpleNamespace(provider='gemini_api',gemini_api_key=None)
    cfg=SimpleNamespace(autointerp=SimpleNamespace(enabled=True,llm=llm))
    assert not ready(cfg)
    monkeypatch.setenv('GEMINI_API_KEY','dummy-env-key');assert ready(cfg)
    assert llm.gemini_api_key is None
    cfg.autointerp.enabled=False;assert not ready(cfg)


def test_same_feature_is_interpreted_once_and_test_groups_are_balanced(tmp_path):
    import numpy as np,pandas as pd,torch
    from omegaconf import OmegaConf
    obj=parser();obj.top_n=1;obj.bottom_n=0;obj.top_pct=1.;obj.n_fit=5;obj.n_test=4;obj.seed=3;obj.model_name='synthetic-model'
    train_ids=[f'train{i}' for i in range(20)];test_ids=[f'test{i}' for i in range(12)]
    metadata=pd.DataFrame(dict(sample_id=train_ids+test_ids,description=['A synthetic numeric observation']*32));path=tmp_path/'metadata.csv';metadata.to_csv(path,index=False)
    obj.cfg=OmegaConf.create({'data':{'metadata':str(path)},'autointerp':{'autointerp_text_map':{'x':'description'},
        'full_reports_prompt':'Reference {reference_patients}\nHigh {patient_data}',
        'discriminative_autointerp_score_prompt':'Interpretation {interpretation}\n{patient_data}'}})
    features={'train':torch.tensor([[float(i+1)] if i<5 else [0.] for i in range(20)]),
              'test':torch.tensor([[float(i+1)] if i<4 else [0.] for i in range(12)])}
    rows=pd.DataFrame([dict(outcome=outcome,modality='x',top_k=1,split='train',feature_idx=0,feature_name='feature_0',train_odds_ratio_rank=1,odds_ratio=2.) for outcome in ['a','b']])
    calls=[]
    def reply(prompt):
        calls.append(prompt)
        return '* This cohort is characterized by higher numeric values.' if len(calls)==1 else '* PREDICTIONS:\n'+'\n'.join(f'{i+1}. {i%2}' for i in range(8))
    obj._call_gemini=reply
    result=obj.run_for_modality('x',features,{'train':train_ids,'test':test_ids},rows,verbose=False)
    assert len(calls)==2 and len(result)==2
    assert result.test_n_total.tolist()==[8,8]
    assert all(len(row.test_positive_ids)==len(row.test_negative_ids)==4 for row in result.itertuples())
    assert result.test_accuracy.notna().all()
    assert 'test_labels' not in calls[1]
