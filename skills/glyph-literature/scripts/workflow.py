"""Freeze reviewed markers, reconstruct them, and run local comparative studies."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent


def load_script(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def review(search,manifest):
    search=Path(search);papers={row['record_id']:row for row in json.loads((search/'papers.json').read_text())}
    ledger=pd.read_csv(search/'screening.csv',keep_default_na=False)
    required={'record_id','title_decision','abstract_decision','feasible','verified_source_url'}
    if required-set(ledger) or ledger.record_id.duplicated().any() or set(ledger.record_id)!=set(papers):raise ValueError('Screening ledger must cover every unique retrieved record exactly once')
    protocol=json.loads((search/'protocol.json').read_text());policy=protocol.get('impact',{})
    journals={str(j).strip().lower() for j in policy.get('leading_journals',[])}
    minimum=policy.get('min_citations')
    impact_passing=set()
    included=set()
    for row in ledger.itertuples():
        journal=str(papers[row.record_id].get('journal','')).strip().lower()
        citation_text=getattr(row,'citation_count','')
        try:citations=int(citation_text) if str(citation_text) else None
        except ValueError:raise ValueError('Citation counts must be integers or empty')
        citation_pass=minimum is not None and citations is not None and citations>=minimum
        impact_pass=(not journals and minimum is None) or journal in journals or citation_pass
        if citation_pass and (not getattr(row,'citation_provider','') or not getattr(row,'citation_date','')):raise ValueError('Citation-based impact eligibility needs provider and retrieval date')
        if impact_pass:impact_passing.add(row.record_id)
        if row.title_decision=='include' and not impact_pass:raise ValueError('Included title does not satisfy the configured impact criterion')
        if row.title_decision not in ['include','exclude']:raise ValueError('Finish title screening before freezing the study')
        if row.abstract_decision not in ['','include','exclude'] or row.feasible not in ['','yes','no']:raise ValueError('Invalid abstract or feasibility decision')
        if row.title_decision=='include' and row.abstract_decision not in ['include','exclude']:raise ValueError('Finish abstract screening for included titles')
        if row.abstract_decision=='include':
            if row.title_decision!='include' or row.feasible not in ['yes','no']:raise ValueError('Record local feasibility after abstract inclusion')
            if row.feasible=='yes':
                if not row.verified_source_url.startswith(('https://','http://')):raise ValueError('Verify the original source before including a locally testable paper')
                included.add(row.record_id)
    marker_ids=set();selected=set()
    for marker in manifest.get('markers',[]):
        identifier=marker.get('id',marker.get('column'))
        if not identifier or identifier in marker_ids:raise ValueError('Marker IDs must be unique')
        marker_ids.add(identifier)
        record=marker.get('record_id')
        if record not in included:raise ValueError('Each marker must cite an included, feasible, source-verified paper')
        if not marker.get('definition') or not marker.get('formula') or not marker.get('units'):raise ValueError('Markers need definitions, explicit formulas and units')
        source=ledger.set_index('record_id').loc[record,'verified_source_url']
        if marker.get('source_url')!=source:raise ValueError('Marker citation must match the source verified in the ledger')
        selected.add(record)
    if not marker_ids:raise ValueError('No locally testable markers selected')
    funnel=json.loads((search/'funnel.json').read_text())
    funnel.update(screening_status='reviewed',impact_passing=len(impact_passing),titles_screened=len(ledger),titles_included=int((ledger.title_decision=='include').sum()),
        abstracts_included=int((ledger.abstract_decision=='include').sum()),locally_testable_papers=len(included),selected_unique_papers=len(selected),markers=len(marker_ids))
    return funnel


def freeze(search,manifest_path,output):
    manifest=json.loads(Path(manifest_path).read_text());funnel=review(search,manifest)
    manifest['source_search']=funnel
    manifest['review_provenance']={name:sha(Path(search)/name) for name in ['papers.json','queries.json','protocol.json','screening.csv']}
    output=Path(output)
    if output.exists():raise ValueError('Frozen manifest already exists; choose a new version')
    save(output,manifest);save(output.with_suffix('.sha256.json'),{'sha256':sha(output)})
    return manifest


def columns(expression):
    if isinstance(expression,str):return set()
    result={expression['column']} if 'column' in expression else set(expression.get('columns',[]))
    for item in expression.get('args',[]):result|=columns(item)
    return result


def evaluate(expression,frame,default_column=None):
    if isinstance(expression,str):
        if expression!='identity':raise ValueError('Use a structured formula; arbitrary code is not evaluated')
        expression={'op':'identity','column':default_column}
    if not isinstance(expression,dict):raise ValueError('Formula must be a declarative object')
    op=expression.get('op')
    if op in ['identity','scale','threshold']:
        if expression.get('column') not in frame:raise ValueError('Required measurement column is absent')
        x=pd.to_numeric(frame[expression['column']],errors='coerce').to_numpy(dtype=float,copy=True)
        x[~np.isfinite(x)]=np.nan
        if op=='scale':return x*float(expression['factor'])+float(expression.get('offset',0))
        if op=='identity':return x
        functions={'>':np.greater,'>=':np.greater_equal,'<':np.less,'<=':np.less_equal,'==':np.equal}
        if expression.get('comparison') not in functions:raise ValueError('Unsupported threshold comparison')
        return np.where(np.isnan(x),np.nan,functions[expression['comparison']](x,float(expression['value'])).astype(float))
    if op in ['ratio','difference','sum','product','all','any']:
        args=expression.get('args',[])
        if not args or (op in ['ratio','difference'] and len(args)!=2):raise ValueError('Wrong number of formula operands')
        values=np.array([evaluate(item,frame) for item in args])
        with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
            if op=='ratio':result=values[0]/values[1]
            elif op=='difference':result=values[0]-values[1]
            elif op=='sum':result=values.sum(axis=0)
            elif op=='product':result=values.prod(axis=0)
            else:
                known=np.isin(values,[0,1])|np.isnan(values)
                if not known.all():raise ValueError('Boolean formulas require 0/1 operands')
                if op=='all':result=np.where((values==0).any(axis=0),0,np.where(np.isnan(values).any(axis=0),np.nan,1))
                else:result=np.where((values==1).any(axis=0),1,np.where(np.isnan(values).any(axis=0),np.nan,0))
        result[~np.isfinite(result)]=np.nan;return result
    raise ValueError(f'Unsupported formula operation: {op}')


def materialize(cohort,measurements,manifest):
    for table in [cohort,measurements]:
        if 'sample_id' not in table or table.sample_id.isna().any() or table.sample_id.duplicated().any():raise ValueError('Inputs require unique nonmissing sample_id values')
    if set(cohort.sample_id)-set(measurements.sample_id):raise ValueError('Measurements must cover the exact eligible cohort')
    values=measurements.set_index('sample_id').loc[cohort.sample_id].reset_index()
    forbidden={'sample_id','patient_id','split','y',manifest.get('outcome_column','y')}
    output=pd.DataFrame({'sample_id':cohort.sample_id});audit=[]
    for marker in manifest['markers']:
        expression=marker['formula'];used=columns(expression) if isinstance(expression,dict) else {marker.get('source_column',marker['column'])}
        if used&forbidden:raise ValueError('Marker reconstruction must not use identifiers, split labels or the outcome')
        missing=used-set(values)
        if missing:raise ValueError(f'Missing measurements for marker {marker["column"]}: {sorted(missing)}')
        output[marker['column']]=evaluate(expression,values,marker.get('source_column',marker['column']))
        audit.append(dict(marker=marker['column'],n=len(output),missing=int(output[marker['column']].isna().sum()),
            proxy=bool(marker.get('proxy',False)),units=marker['units'],source_url=marker['source_url']))
    return output,audit


def run_study(cohort_path,measurements_path,glyph_path,manifest_path,output,*,bootstrap=2000,search=None,run=None,report=False,glyph_associations=None,interpretation_config=None,confounder_cols=None,adjustment_device='cpu',image_examples=False):
    output=Path(output)
    inputs={'cohort':sha(cohort_path),'measurements':sha(measurements_path),'glyph':sha(glyph_path),'manifest':sha(manifest_path),
            'bootstrap':bootstrap,'report':report,'image_examples':image_examples,'confounder_cols':confounder_cols,'adjustment_device':adjustment_device,'workflow':sha(__file__),'comparison':sha(HERE/'compare.py')}
    if image_examples and not report:raise ValueError('Image examples require --report')
    if report:inputs['report_script']=sha(HERE.parents[1]/'glyph-report/scripts/build_report.py')
    if interpretation_config:inputs['interpretation_script']=sha(HERE.parents[1]/'glyph-discovery/scripts/interpret.py')
    if adjustment_device!='cpu' and not confounder_cols:raise ValueError('Confounder columns are required for an adjustment device option')
    if glyph_associations:inputs['glyph_associations']=sha(glyph_associations)
    if interpretation_config:inputs['interpretation_config']=sha(interpretation_config)
    if run:
        run=Path(run)
        inputs['run_files']={path.name:sha(path) for path in run.iterdir() if path.is_file() and path.name in ['fitted_saes.pt','sae_features.pkl','hypotheses.pkl','autointerp_results.pkl','config.yaml']}
    if search:inputs['search_files']={name:sha(Path(search)/name) for name in ['papers.json','queries.json','protocol.json','screening.csv']}
    output.mkdir(parents=True,exist_ok=True);statepath=output/'state.json'
    state=json.loads(statepath.read_text()) if statepath.exists() else {'inputs':inputs,'completed':[]}
    if state['inputs']!=inputs:raise ValueError('Study inputs or code changed; use a new output directory')
    save(statepath,state)
    manifest=json.loads(Path(manifest_path).read_text())
    if search:
        current=review(search,manifest)
        provenance=manifest.get('review_provenance',{})
        if any(provenance.get(name)!=sha(Path(search)/name) for name in ['papers.json','queries.json','protocol.json','screening.csv']):raise ValueError('Search review changed after the manifest was frozen')
    elif manifest.get('review_provenance'):raise ValueError('Supply the reviewed search packet for a search-backed frozen manifest')
    cohort=pd.read_csv(cohort_path,dtype={'sample_id':str,'patient_id':str});measurements=pd.read_csv(measurements_path,dtype={'sample_id':str})
    markers,audit=materialize(cohort,measurements,manifest);markers.to_csv(output/'markers.csv',index=False);save(output/'marker_reconstruction.json',audit)
    if 'comparison' not in state['completed']:
        target=output/'comparison'
        if target.exists():
            attempts=output/'attempts';attempts.mkdir(exist_ok=True);shutil.move(str(target),str(attempts/('comparison-'+uuid.uuid4().hex[:8])))
        command=[sys.executable,'-B',str(HERE/'compare.py'),'--cohort',str(cohort_path),'--glyph',str(glyph_path),
            '--markers',str(output/'markers.csv'),'--manifest',str(manifest_path),'--output',str(target),'--bootstrap',str(bootstrap)]
        if glyph_associations:command+=['--glyph-associations',str(glyph_associations)]
        if confounder_cols:command+=['--adjust-confounders','--confounder-cols',*confounder_cols,'--adjustment-device',adjustment_device]
        with (output/'comparison.log').open('w') as log:subprocess.run(command,stdout=log,stderr=log,check=True)
        if search:save(target/'search_funnel.json',current)
        state['completed'].append('comparison');save(statepath,state)
    if interpretation_config and 'interpretation' not in state['completed']:
        if not run:raise ValueError('Optional interpretation requires a fitted --run')
        pending=output/'interpretations'
        if pending.exists():
            attempts=output/'attempts';attempts.mkdir(exist_ok=True);shutil.move(str(pending),str(attempts/('interpretation-'+uuid.uuid4().hex[:8])))
        script=HERE.parents[1]/'glyph-discovery/scripts/interpret.py'
        with (output/'interpretation.log').open('w') as log:subprocess.run([sys.executable,'-B',str(script),'--run',str(run),'--config',str(interpretation_config),'--output',str(output/'interpretations')],stdout=log,stderr=log,check=True)
        state['completed'].append('interpretation');save(statepath,state)
    if report and 'report' not in state['completed']:
        if not run:raise ValueError('The full-space report requires a fitted --run')
        script=HERE.parents[1]/'glyph-report/scripts/build_report.py';target=output/'report'
        if target.exists():
            attempts=output/'attempts';attempts.mkdir(exist_ok=True);shutil.move(str(target),str(attempts/('report-'+uuid.uuid4().hex[:8])))
        metadata={'outcome_labels':manifest.get('outcome_labels',{}),'scope':manifest.get('scope',manifest.get('title','Reviewed literature study')),
            'notes':' '.join(manifest.get('caveats',[])),'interpretation_status':'Interpretation requested separately.' if interpretation_config else 'Interpretation not requested; cached interpretations are shown when available.'}
        save(output/'report_metadata.json',metadata)
        comparison_split=manifest.get('report_comparison_split') or next(split for split in cohort.split.unique() if split not in ['train','val'])
        command=[sys.executable,'-B',str(script),'--comparison-split',comparison_split,'--run',str(run),'--literature',str(output/'comparison'),
            '--metadata',str(output/'report_metadata.json'),'--output',str(target),'--title',manifest.get('title','Glyph literature study')]
        if image_examples:command+=['--image-examples']
        if interpretation_config:command+=['--interpretations',str(output/'interpretations/autointerp_results.pkl')]
        with (output/'report.log').open('w') as log:subprocess.run(command,stdout=log,stderr=log,check=True)
        state['completed'].append('report');save(statepath,state)
    return state


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    frozen=commands.add_parser('freeze')
    for name in ['search','manifest','output']:frozen.add_argument('--'+name,type=Path,required=True)
    study=commands.add_parser('study')
    for name in ['cohort','measurements','glyph','manifest','output']:study.add_argument('--'+name,type=Path,required=True)
    for name in ['search','run','glyph-associations','interpretation-config']:study.add_argument('--'+name,type=Path)
    study.add_argument('--confounder-cols',nargs='+');study.add_argument('--adjustment-device',choices=['cpu','cuda'],default='cpu')
    study.add_argument('--image-examples',action='store_true')
    study.add_argument('--report',action='store_true');study.add_argument('--bootstrap',type=int,default=2000)
    args=parser.parse_args(argv)
    if args.command=='freeze':result=freeze(args.search,args.manifest,args.output)
    else:result=run_study(args.cohort,args.measurements,args.glyph,args.manifest,args.output,bootstrap=args.bootstrap,search=args.search,run=args.run,report=args.report,glyph_associations=args.glyph_associations,interpretation_config=args.interpretation_config,confounder_cols=args.confounder_cols,adjustment_device=args.adjustment_device,image_examples=args.image_examples)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
