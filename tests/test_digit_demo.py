import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from examples.dinov3_digits import digit4_focus


def test_digit4_focus_selects_complementary_separated_training_concepts(tmp_path, monkeypatch):
    train=torch.zeros(40,3);train[:12,0]=torch.linspace(1,2,12);train[12:24,1]=torch.linspace(1,2,12);train[:24,2]=1
    run=tmp_path/'run';run.mkdir();html=run/'concept_map_image_path_k2_train.html'
    nodes=[dict(feature_idx=i,x=x,y=y) for i,(x,y) in enumerate([(.1,.1),(.7,.7),(.45,.45)])]
    html.write_text('const DATA='+json.dumps(dict(nodes=nodes))+',NS=')
    rows=pd.DataFrame([dict(split='train',feature_idx=i,outcome='is_digit_4',odds_ratio=2.,odds_ratio_p_value_bonferroni=.001) for i in range(3)])
    obj=SimpleNamespace(run_dir=run,phewas_results=rows,sae_features={'image_path':{'train':train}},outcomes={'train':{'is_digit_4':np.r_[np.ones(24),np.zeros(16)]}},sae_models={'image_path':SimpleNamespace(selected_top_k=2)})
    result=digit4_focus(obj)
    assert set(result['focus_concepts'])=={'image_path:2:0','image_path:2:1'}
    assert result['digit4_selection']['within_digit4_correlation']<0
    import examples.dinov3_digits as demo
    monkeypatch.setattr(demo,'__file__',str(tmp_path/'examples/dinov3_digits.py'))
    profile=tmp_path/'demos/dinov3_digits/report_profile.json';profile.parent.mkdir(parents=True)
    profile.write_text(json.dumps(dict(source_run='run',focus_concepts=['image_path:2:0','image_path:2:2'],selection_note='Reviewed nearby higher-prevalence examples')))
    curated=digit4_focus(obj)
    assert curated['focus_concepts']==['image_path:2:0','image_path:2:2']
    assert curated['digit4_selection']['training_active_counts']==[12,24]
    profile.unlink()
    nodes[1].update(x=.11,y=.11);html.write_text('const DATA='+json.dumps(dict(nodes=nodes))+',NS=')
    assert digit4_focus(obj)=={}  # Do not substitute an adjacent duplicate.
