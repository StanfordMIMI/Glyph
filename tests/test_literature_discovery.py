import csv
import importlib.util
import io
import json
from pathlib import Path
import urllib.error
import urllib.parse

import pytest

ROOT=Path(__file__).parents[1]/'skills/glyph-literature/scripts'
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
discovery=load('discover');workflow=load('workflow')


@pytest.fixture
def protocol():
    return dict(schema_version=1,scope='Synthetic retrieval check',outcome='test',queries={'pubmed':['sample query'],'paperclip':['supplement query']},max_records_per_query=2)


class FakeSearch:
    def __init__(self):self.requests=[]
    def __call__(self,request,timeout):
        self.requests.append(request)
        if request.full_url.endswith('esearch.fcgi'):
            data={'esearchresult':{'count':'5','idlist':['123']}};return io.BytesIO(json.dumps(data).encode())
        if request.full_url.endswith('efetch.fcgi'):
            return io.BytesIO(b'<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article><ArticleTitle>Public synthetic paper</ArticleTitle><Abstract><AbstractText>Example abstract.</AbstractText></Abstract><Journal><Title>Example Journal</Title></Journal></Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/example</ArticleId></ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>')
        return io.BytesIO(json.dumps({'papers':[dict(id='internal-id',object='paper',pmid='123',doi='https://doi.org/10.1/example',title='Same synthetic paper',url='https://example.org/source'),dict(id='other',object='paper',doi='10.1/other',title='Supplement',url='https://example.org/other')],'usage':{'searches':1}}).encode())


def test_resumption_deduplication_denominators_and_secret_handling(tmp_path,monkeypatch,protocol):
    key='dummy-secret-never-save';monkeypatch.setenv('PAPERCLIP_API_KEY',key);monkeypatch.setenv('NCBI_API_KEY','dummy-ncbi-secret')
    fake=FakeSearch();transport=discovery.Retrieval(tmp_path/'cache',opener=fake)
    result=discovery.discover(protocol,tmp_path,retrieval=transport)
    assert result['pubmed_unique_retrieved']==1 and result['paperclip_only_unique']==1
    assert result['combined_unique']==2
    assert result['pubmed_hit_counts_by_query'][0]['total_hits']==5
    assert result['pubmed_hit_counts_by_query'][0]['truncated']
    assert len(fake.requests)==3
    search=next(request for request in fake.requests if request.full_url.endswith('/search'))
    assert search.get_header('X-api-key')==key
    assert json.loads(search.data)['sources']==['abstracts']
    for path in tmp_path.rglob('*'):
        if path.is_file():assert key not in path.read_text() and 'dummy-ncbi-secret' not in path.read_text()
    def never(*args,**kwargs):pytest.fail('Offline replay made a request')
    replay=discovery.discover(protocol,tmp_path,offline=True,retrieval=discovery.Retrieval(tmp_path/'cache',True,never))
    assert replay['combined_unique']==2
    assert replay['cache_hits']==3


def test_optional_key_is_skipped_and_screening_decisions_survive_resume(tmp_path,monkeypatch,protocol):
    monkeypatch.delenv('PAPERCLIP_API_KEY',raising=False)
    discovery.discover(protocol,tmp_path,retrieval=discovery.Retrieval(tmp_path/'cache',opener=FakeSearch()))
    rows=list(csv.DictReader((tmp_path/'screening.csv').open()));rows[0]['title_decision']='exclude'
    with (tmp_path/'screening.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    discovery.discover(protocol,tmp_path,offline=True)
    assert list(csv.DictReader((tmp_path/'screening.csv').open()))[0]['title_decision']=='exclude'
    changed=dict(protocol,scope='different')
    with pytest.raises(ValueError,match='Protocol changed'):discovery.discover(changed,tmp_path)


def test_auth_failure_does_not_expose_credentials(tmp_path):
    secret='private-dummy-key'
    def fail(request,timeout):raise urllib.error.HTTPError(request.full_url,401,secret,{},None)
    with pytest.raises(RuntimeError) as error:discovery.Retrieval(tmp_path,opener=fail).request('paperclip','search',{'query':'test'},key=secret)
    assert secret not in str(error.value)
    assert not list(tmp_path.glob('*.json'))


def test_identifier_bridge_merges_all_providers():
    rows=[dict(provider='pubmed',pmid='1',title='A'),dict(provider='paperclip',doi='10.1/a',title='B'),dict(provider='paperclip',pmid='1',doi='10.1/a',title='A')]
    assert len(discovery.deduplicate(rows))==1


def test_marker_formula_missingness_and_outcome_protection():
    import numpy as np,pandas as pd
    frame=pd.DataFrame({'sample_id':['1','2','3'],'a':[2.,0.,np.nan],'b':[1.,0.,2.]})
    cohort=pd.DataFrame({'sample_id':['3','1','2']})
    expr={'op':'ratio','args':[{'op':'identity','column':'a'},{'op':'identity','column':'b'}]}
    manifest={'markers':[dict(column='ratio',formula=expr,units='unitless',source_url='https://example.org')]}
    result,audit=workflow.materialize(cohort,frame,manifest)
    assert np.isnan(result.ratio.iloc[0]) and result.ratio.iloc[1]==2 and np.isnan(result.ratio.iloc[2])
    assert audit[0]['missing']==2
    manifest['markers'][0]['formula']={'op':'identity','column':'y'}
    with pytest.raises(ValueError,match='outcome'):workflow.materialize(cohort,frame.assign(y=1),manifest)
    with pytest.raises(ValueError,match='arbitrary code'):workflow.evaluate('__import__("os")',frame)


def make_packet(folder):
    folder.mkdir();record='pmid:synthetic'
    (folder/'papers.json').write_text(json.dumps([dict(record_id=record,title='Synthetic test source')]))
    (folder/'queries.json').write_text('[]');(folder/'protocol.json').write_text('{}');(folder/'funnel.json').write_text(json.dumps(dict(combined_unique=1)))
    fields=['record_id','title_decision','abstract_decision','feasible','verified_source_url']
    with (folder/'screening.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerow(dict(record_id=record,title_decision='include',abstract_decision='include',feasible='yes',verified_source_url='https://example.org/synthetic'))
    return dict(title='Synthetic software check',markers=[dict(id='m1',column='m1',record_id=record,definition='Synthetic identity feature',units='arbitrary',formula={'op':'identity','column':'x'},source_url='https://example.org/synthetic')],glyph_features=['g1'],glyph_selection_rule='Predetermined synthetic feature')


def test_freeze_requires_completed_source_verified_review(tmp_path):
    packet=tmp_path/'search';manifest=make_packet(packet);source=tmp_path/'candidate.json';source.write_text(json.dumps(manifest))
    result=workflow.freeze(packet,source,tmp_path/'frozen.json')
    assert result['source_search']['selected_unique_papers']==1
    assert result['review_provenance']['screening.csv']==workflow.sha(packet/'screening.csv')
    with pytest.raises(ValueError,match='already exists'):workflow.freeze(packet,source,tmp_path/'frozen.json')
    manifest['markers'][0]['source_url']='https://different.example/'
    with pytest.raises(ValueError,match='match the source'):workflow.review(packet,manifest)


def test_real_numerical_study_resumes_without_refitting_and_rejects_stale_inputs(tmp_path):
    import numpy as np,pandas as pd
    packet=tmp_path/'search';manifest=make_packet(packet);candidate=tmp_path/'candidate.json';candidate.write_text(json.dumps(manifest))
    frozen=tmp_path/'frozen.json';workflow.freeze(packet,candidate,frozen)
    rng=np.random.default_rng(9);x=rng.normal(size=120);y=(x+rng.normal(size=120)>.1).astype(int)
    cohort=pd.DataFrame(dict(sample_id=[f'fake{i}' for i in range(120)],split=['train']*60+['val']*30+['test']*30,y=y))
    measurements=pd.DataFrame(dict(sample_id=cohort.sample_id,x=x))
    glyph=pd.DataFrame(dict(sample_id=cohort.sample_id,g1=x+rng.normal(size=120)*.2))
    files=[]
    for name,frame in [('cohort',cohort),('measurements',measurements),('glyph',glyph)]:
        path=tmp_path/(name+'.csv');frame.to_csv(path,index=False);files.append(path)
    state=workflow.run_study(*files,frozen,tmp_path/'study',bootstrap=100,search=packet)
    assert state['completed']==['comparison']
    prediction=tmp_path/'study/comparison/predictions.csv';mtime=prediction.stat().st_mtime_ns
    workflow.run_study(*files,frozen,tmp_path/'study',bootstrap=100,search=packet)
    assert prediction.stat().st_mtime_ns==mtime
    measurements['x']=measurements.x+1;measurements.to_csv(files[1],index=False)
    with pytest.raises(ValueError,match='inputs or code changed'):workflow.run_study(*files,frozen,tmp_path/'study',bootstrap=100,search=packet)


def test_pubmed_doi_comes_from_the_paper_not_its_references():
    xml = b'''<PubmedArticleSet><PubmedArticle>
    <MedlineCitation><PMID>123</PMID><Article><ArticleTitle>Primary paper</ArticleTitle></Article></MedlineCitation>
    <PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/primary</ArticleId></ArticleIdList>
      <ReferenceList><Reference><ArticleIdList><ArticleId IdType="doi">10.1/reference</ArticleId></ArticleIdList></Reference></ReferenceList>
    </PubmedData></PubmedArticle></PubmedArticleSet>'''
    result = discovery.pubmed_records(xml)[0]
    assert result['pmid'] == '123' and result['doi'] == '10.1/primary'
    # A reference DOI must never fill an unknown primary-paper DOI.
    xml = xml.replace(b'<ArticleId IdType="doi">10.1/primary</ArticleId>', b'')
    assert discovery.pubmed_records(xml)[0]['doi'] == ''
