import base64
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from glyph.visualization.image_examples import collect_image_examples, image_payload


def test_examples_use_activation_ranks_and_share_one_thumbnail_per_image(tmp_path):
    paths=[]
    for i in range(4):
        p=tmp_path/f'{i}.png';Image.new('RGB',(500,250),(i*60,30,50)).save(p);paths.append(str(p))
    metadata=pd.DataFrame(dict(sample_id=['a','b','c','d'],image_path=paths,outcome=[0,1,0,1]))
    values=np.array([[1.,9.],[6.,7.],[4.,2.],[0.,0.]])
    result=collect_image_examples(values,['a','b','c','d'],metadata,'image_path',split='train')
    assert len(result['image_bank'])==3
    names=lambda index:[result['image_bank'][key]['sample_id'] for key,value in result['concept_images'][str(index)]]
    assert names(0)==['b','c'] and names(1)==['a','b']
    for record in result['image_bank'].values():
        assert max(record['width'],record['height'])==256
        with Image.open(BytesIO(base64.b64decode(record['src'].split(',')[1]))) as im:assert im.size==(256,128)
    flipped=metadata.copy();flipped.outcome=1-flipped.outcome
    assert collect_image_examples(values,['a','b','c','d'],flipped,'image_path')==result
    assert image_payload(result)==result


def test_sparse_examples_missing_paths_and_nonimages_are_explicit(tmp_path):
    path=tmp_path/'digit.png';Image.new('RGB',(8,8),'white').save(path)
    metadata=pd.DataFrame(dict(sample_id=['a','b'],image_path=[str(tmp_path/'missing.png'),str(path)]))
    result=collect_image_examples(np.array([[8.,0.],[1.,0.]]),['a','b'],metadata,'image_path')
    assert len(result['concept_images']['0'])==1 and '1' not in result['concept_images']
    record=next(iter(result['image_bank'].values()));assert record['pixelated'] and record['src'].startswith('data:image/png;')
    assert not collect_image_examples(np.ones((2,2)),['a','b'],metadata,'image_path',count=0)['image_bank']
    metadata['labs_path']=['not-a-lab.json']*2
    assert not collect_image_examples(np.ones((2,2)),['a','b'],metadata,'labs_path')['image_bank']
    with pytest.raises(ValueError):image_payload(dict(image_bank={'a':dict(src='https://example.com/image.png')},concept_images={}))


def test_main_visualize_uses_sample_ids_and_supports_omitting_images(tmp_path):
    import json,re,torch
    from omegaconf import OmegaConf
    from glyph import Glyph
    paths=[]
    for label in ['a','b']:
        p=tmp_path/f'{label}.png';Image.new('RGB',(8,8),'red').save(p);paths.append(str(p))
    csv=tmp_path/'metadata.csv';pd.DataFrame(dict(case_id=['a','b'],image_path=paths)).to_csv(csv,index=True)
    obj=Glyph.__new__(Glyph);obj.cfg=OmegaConf.create(dict(seed=42,data=dict(metadata=str(csv),output_dir=str(tmp_path),outcome_cols=[],embedding_map={'image_path':'precomputed'}),visualization=dict(interactive=True,static_jpg=False,correlation_percentile=.975,bonferroni_alpha=.05)))
    obj.sae_features={'image_path':{'train':torch.tensor([[1.],[3.]])}};obj.sae_alive_features={('image_path',1):torch.tensor([True])}
    obj.embeddings={'train':{'sample_ids':['b','a']}};obj.phewas_results=None;obj.autointerp_results=None
    path=obj.visualize()[0];payload=json.loads(re.search(r'const DATA=(.*?),NS=',path.read_text()).group(1))
    ids=[payload['image_bank'][key]['sample_id'] for key,activation in payload['concept_images']['0']]
    assert ids==['a','b']
    path=obj.visualize(image_examples=0)[0];payload=json.loads(re.search(r'const DATA=(.*?),NS=',path.read_text()).group(1))
    assert not payload['image_bank']
