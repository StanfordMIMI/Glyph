import numpy as np
import pandas as pd
from examples.pad_ufes import select_patients, clinical_measurements, baseline_manifest, SYMPTOMS


def test_patient_selection_is_unique_and_outcome_blind():
    rows=[]
    for p in range(60):
        for image in ['a','b']:
            rows.append(dict(patient_id=f'p{p:03}',lesion_id=p,img_id=f'p{p:03}_{image}.png',diagnostic='BCC' if p%2 else 'ACK',age=60,region='ARM',**{c:'TRUE' for c in SYMPTOMS}))
    source=pd.DataFrame(rows);selected=select_patients(source)
    assert len(selected)==60 and selected.patient_id.is_unique
    assert selected.img_id.str.endswith('_a.png').all()
    changed=source.copy();changed.loc[changed.img_id.str.endswith('_b.png'),'diagnostic']='MEL'
    assert set(select_patients(changed).img_id)==set(selected.img_id)
    assert set(selected.split)=={'train','val','test'}


def test_unknown_symptoms_remain_missing_and_labels_are_excluded():
    frame=pd.DataFrame(dict(sample_id=['a','b'],age=[30,70],region=['ARM','UNKNOWN'],**{c:['False','UNK'] for c in SYMPTOMS}))
    result=clinical_measurements(frame)
    assert result.loc[0,'itch']==0 and np.isnan(result.loc[1,'itch'])
    assert result.loc[0,'site_arm']==1 and np.isnan(result.loc[1,'site_arm'])
    markers=[m['column'] for m in baseline_manifest()['markers']]
    assert set(markers)==set(result)-{'sample_id'} and len(markers)==22
    assert not set(markers)&{'diagnostic','biopsed','skin_cancer','patient_id'}


def test_readable_names_compose_without_changing_outcome_keys(tmp_path, monkeypatch):
    import examples.pad_ufes as demo
    from glyph.config import load_config
    monkeypatch.setattr(demo, 'Glyph', lambda name, config_overrides: load_config(name, config_overrides))
    cfg = demo.make_glyph(tmp_path / 'metadata.csv', tmp_path / 'runs')
    assert list(cfg.data.outcome_cols) == demo.OUTCOMES
    assert set(cfg.visualization.outcome_names) == set(demo.OUTCOMES)
    assert cfg.visualization.outcome_names.is_nev == 'Melanocytic nevus (mole)'
