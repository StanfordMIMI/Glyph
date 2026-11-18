import pytest
from glyph.visualization.regions import pack_regions

NODES = [dict(i=0,concept_id='a',name='feature_3',x=.2,y=.3),
         dict(i=1,concept_id='b',name='feature_12',x=.5,y=.3)]

def test_single_and_collinear_regions_preserve_members_and_coordinates():
    result=pack_regions([dict(label='Single',concepts=['a','a']),
                         dict(label='Pair',concepts=['a','b'])],NODES)
    assert result[0]['members']==[0]
    assert result[1]['members']==[0,1]
    assert len(result[0]['points'])>=4 and len(result[1]['points'])>=4
    assert NODES[0]['x']==.2 and NODES[1]['x']==.5

def test_regions_accept_unique_names_and_keep_text_literal():
    result=pack_regions([dict(label='<script>example</script>',concepts=['feature_3'])],NODES)
    assert result[0]['members']==[0]
    assert result[0]['label']=='<script>example</script>'

@pytest.mark.parametrize('region',[dict(label='Missing',concepts=['missing']),
                                    dict(label='',concepts=['a']),dict(label='Empty',concepts=[])])
def test_invalid_regions_are_rejected(region):
    with pytest.raises(ValueError):pack_regions([region],NODES)

def test_no_regions_means_no_invented_overlay():
    assert pack_regions(None,NODES)==[]
