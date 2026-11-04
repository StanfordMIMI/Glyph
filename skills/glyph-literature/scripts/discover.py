"""Retrieve a resumable, source-linked literature packet for agent screening."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

PUBMED='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
PAPERCLIP='https://paperclip.gxl.ai/api/v1/'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'))


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


class Retrieval:
    def __init__(self,cache,offline=False,opener=None):
        self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.offline=offline;self.opener=opener or urllib.request.urlopen
        self.live_requests=0;self.cache_hits=0;self.last_pubmed=0

    def request(self,provider,operation,payload,key=None,xml=False):
        # Credentials never form part of a cache key, URL, saved response envelope or exception.
        identifier=hashlib.sha256(canonical([provider,operation,payload,xml]).encode()).hexdigest()
        path=self.cache/(identifier+'.json')
        if path.exists():self.cache_hits+=1;return json.loads(path.read_text())
        if self.offline:raise ValueError(f'Offline cache miss for {provider} {operation}')
        headers={'User-Agent':'Glyph-literature/0.1','Accept':'application/json'}
        if provider=='pubmed':
            url=PUBMED+operation;body=dict(payload,tool='Glyph')
            if key:body['api_key']=key
            data=urllib.parse.urlencode(body).encode();headers['Content-Type']='application/x-www-form-urlencoded'
        elif provider=='paperclip':
            if not key:raise ValueError('Set PAPERCLIP_API_KEY for Paperclip search')
            url=PAPERCLIP+operation;data=canonical(payload).encode()
            headers.update({'Content-Type':'application/json','X-API-Key':key})
        else:raise ValueError('Unknown literature provider')
        for attempt in range(3):
            if provider=='pubmed':
                time.sleep(max(0,.35-(time.monotonic()-self.last_pubmed)))
                self.last_pubmed=time.monotonic()
            request=urllib.request.Request(url,data=data,headers=headers,method='POST')
            try:
                self.live_requests+=1
                with self.opener(request,timeout=25) as response:raw=response.read().decode('utf-8')
                result={'data':raw if xml else json.loads(raw),'retrieved_at':datetime.now(timezone.utc).isoformat(),
                        'sha256':hashlib.sha256(raw.encode()).hexdigest(),'provider':provider}
                if isinstance(result['data'],dict) and ('error' in result['data'] or 'ERROR' in result['data']):
                    raise ValueError(f'{provider} returned an API error; response was not cached')
                write_json(path,result);return result
            except urllib.error.HTTPError as exc:
                if exc.code in [429,500,502,503,504] and attempt<2:time.sleep(min(5,2**attempt));continue
                raise RuntimeError(f'{provider} request failed (HTTP {exc.code}); credentials and response body omitted') from None
            except (urllib.error.URLError,TimeoutError):
                if attempt<2:time.sleep(2**attempt);continue
                raise RuntimeError(f'{provider} network request failed after three attempts') from None


def text(node,path):
    found=node.find(path)
    return ''.join(found.itertext()).strip() if found is not None else ''


def pubmed_records(xml):
    root=ET.fromstring(xml);records=[]
    for node in root.findall('.//PubmedArticle'):
        identifiers={element.get('IdType'):element.text for element in node.findall('./PubmedData/ArticleIdList/ArticleId')}
        pmid=text(node,'.//MedlineCitation/PMID')
        records.append(dict(pmid=pmid,doi=identifiers.get('doi') or '',title=text(node,'.//ArticleTitle'),
            abstract='\n'.join(''.join(element.itertext()).strip() for element in node.findall('.//Abstract/AbstractText')),
            journal=text(node,'.//Journal/Title'),year=text(node,'.//PubDate/Year') or text(node,'.//PubDate/MedlineDate')[:4],
            source_url=f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',provider='pubmed'))
    return records


def normalize_doi(value):
    return re.sub(r'^(https?://(dx\.)?doi.org/|doi:\s*)','',str(value or '').strip().lower())


def deduplicate(records):
    # Union all aliases, so a DOI-only supplement can bridge records also carrying a PMID.
    output=[]
    for raw in records:
        row=dict(raw);row['doi']=normalize_doi(row.get('doi'));row['pmid']=str(row.get('pmid') or '')
        aliases={prefix+value for prefix,value in [('pmid:',row['pmid']),('doi:',row['doi'])] if value}
        if not aliases:aliases={'title:'+re.sub(r'\W+','',row.get('title','').lower())}
        matches=[i for i,item in enumerate(output) if aliases & set(item['_aliases'])]
        sources=[dict(provider=row['provider'],query=row.get('query',''),source_url=row.get('source_url',''))]
        for i in reversed(matches):
            old=output.pop(i);aliases.update(old['_aliases']);sources.extend(old['sources'])
            for name in ['pmid','doi','abstract','journal','year','title','source_url']:
                if not row.get(name):row[name]=old.get(name,'')
        row['_aliases']=sorted(aliases);row['sources']=sources;output.append(row)
    for row in output:
        row['record_id']=('pmid:'+row['pmid']) if row['pmid'] else ('doi:'+row['doi']) if row['doi'] else row['_aliases'][0]
        row.pop('_aliases',None)
    return sorted(output,key=lambda row:row['record_id'])


def validate_protocol(protocol):
    allowed={'schema_version','scope','outcome','queries','max_records_per_query','paperclip_sources','paperclip_key_env','require_paperclip','notes','impact'}
    if set(protocol)-allowed:raise ValueError('Unsupported protocol fields; store credentials in environment variables only')
    if protocol.get('schema_version')!=1 or not protocol.get('scope') or not protocol.get('outcome'):raise ValueError('Protocol needs schema_version=1, scope and outcome')
    queries=protocol.get('queries',{})
    if set(queries)-{'pubmed','paperclip'} or not any(queries.values()):raise ValueError('Supply pubmed and/or paperclip queries')
    if any(not isinstance(values,list) or any(not isinstance(v,str) or not v.strip() for v in values) for values in queries.values()):raise ValueError('Queries must be nonempty strings in lists')
    limit=protocol.get('max_records_per_query',100)
    if not isinstance(limit,int) or not 1<=limit<=9999:raise ValueError('max_records_per_query must be 1–9999; split larger PubMed searches explicitly')
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',protocol.get('paperclip_key_env','PAPERCLIP_API_KEY')):raise ValueError('Invalid credential environment variable name')
    return limit


def discover(protocol,output,offline=False,retrieval=None):
    limit=validate_protocol(protocol);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    protocol_path=output/'protocol.json'
    if protocol_path.exists() and canonical(json.loads(protocol_path.read_text()))!=canonical(protocol):raise ValueError('Protocol changed; choose a new search directory')
    write_json(protocol_path,protocol)
    transport=retrieval or Retrieval(output/'cache',offline)
    key=os.environ.get(protocol.get('paperclip_key_env','PAPERCLIP_API_KEY'))
    if protocol.get('require_paperclip') and not key and not offline:raise ValueError('Required Paperclip key is not configured')
    all_records=[];query_log=[]
    for provider,queries in protocol['queries'].items():
        for query in queries:
            if provider=='paperclip':
                prior=json.loads((output/'queries.json').read_text()) if (output/'queries.json').exists() else []
                skipped=any(row['provider']==provider and row['query']==query and row['status']=='skipped_missing_key' for row in prior)
                if not key and (not offline or skipped):
                    query_log.append(dict(provider=provider,query=query,status='skipped_missing_key'));continue
                payload=dict(query=query,num_results=min(limit,200),sources=protocol.get('paperclip_sources',['abstracts']),expand=False,contents={'abstract':True,'text':False})
                response=transport.request(provider,'search',payload,key=key)
                data=response['data'];hits=data.get('papers',data.get('results'))
                if not isinstance(hits,list):raise ValueError('Unexpected Paperclip search response; expected papers/results')
                for hit in hits:
                    if hit.get('object','paper')!='paper':continue
                    all_records.append(dict(provider=provider,query=query,pmid=str(hit.get('pmid') or ''),doi=hit.get('doi') or '',
                        title=hit.get('title',''),abstract=hit.get('abstract') or hit.get('snippet',''),journal=hit.get('journal') or '',
                        year=hit.get('year') or '',source_url=hit.get('url') or '',paperclip_id=hit.get('id','')))
                query_log.append(dict(provider=provider,query=query,status='complete',retrieved=len(hits),
                    coverage='Ranked, capped supplementary results; not an exhaustive denominator',retrieved_at=response['retrieved_at'],usage=data.get('usage',{})))
            else:
                response=transport.request(provider,'esearch.fcgi',dict(db='pubmed',term=query,retmode='json',retmax=limit,sort='pub_date'),key=os.environ.get('NCBI_API_KEY'))
                result=response['data']['esearchresult'];ids=result['idlist'];count=int(result['count'])
                for start in range(0,len(ids),100):
                    fetched=transport.request(provider,'efetch.fcgi',dict(db='pubmed',id=','.join(ids[start:start+100]),retmode='xml'),key=os.environ.get('NCBI_API_KEY'),xml=True)
                    all_records.extend(dict(row,query=query) for row in pubmed_records(fetched['data']))
                query_log.append(dict(provider=provider,query=query,status='complete',total_hits=count,retrieved=len(ids),truncated=len(ids)<count,retrieved_at=response['retrieved_at']))
    records=deduplicate(all_records)
    write_json(output/'papers.json',records);write_json(output/'queries.json',query_log)
    funnel=dict(pubmed_hit_counts_by_query=[row for row in query_log if row['provider']=='pubmed'],
        pubmed_unique_retrieved=sum(any(s['provider']=='pubmed' for s in row['sources']) for row in records),
        paperclip_only_unique=sum(all(s['provider']!='pubmed' for s in row['sources']) for row in records),
        combined_unique=len(records),screening_status='pending_agent_review',live_requests=transport.live_requests,cache_hits=transport.cache_hits)
    write_json(output/'funnel.json',funnel)
    ledger=output/'screening.csv'
    fields=['record_id','title','title_decision','abstract_decision','feasible','reason','citation_count','citation_provider','citation_date','verified_source_url']
    existing=list(csv.DictReader(ledger.open())) if ledger.exists() else []
    known={row['record_id'] for row in existing}
    if known-{row['record_id'] for row in records}:raise ValueError('Retrieved identifiers changed; create a new search snapshot before rescreening')
    new=[dict(record_id=row['record_id'],title=row['title']) for row in records if row['record_id'] not in known]
    if new or not ledger.exists():
        with ledger.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(existing+new)
    return funnel


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--offline',action='store_true');args=parser.parse_args(argv)
    print(json.dumps(discover(json.loads(args.protocol.read_text()),args.output,args.offline),indent=2))


if __name__=='__main__':main()
