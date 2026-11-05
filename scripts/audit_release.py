"""Read-only public asset and manuscript-link audit. Never uploads or exports cohort data."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.request


def get(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Glyph-release-audit'}),timeout=25) as response:
            return {'status':response.status,'body':response.read()}
    except urllib.error.HTTPError as exc:return {'status':exc.code,'body':b''}
    except (urllib.error.URLError,TimeoutError):return {'status':None,'body':b''}


def audit(paper=None):
    from glyph.zoo.registry import ZOO
    from glyph.extraction.labs import LABS_ENCODER_REPO,LABS_ENCODER_REVISION
    records=[]
    assets=[('SAE concepts','RobbieHolland/HypothesisExtractor',next(iter(ZOO.values())).revision),
            ('Labs foundation encoder',LABS_ENCODER_REPO,LABS_ENCODER_REVISION)]
    for kind,repo,revision in assets:
        response=get(f'https://huggingface.co/api/models/{repo}/revision/{revision}')
        row=dict(kind=kind,repo=repo,revision=revision,anonymous_http_status=response['status'],accessible=response['status']==200)
        if row['accessible']:
            data=json.loads(response['body']);row.update(actual_revision=data.get('sha'),files=[item['rfilename'] for item in data.get('siblings',[])],
                license=data.get('cardData',{}).get('license'),gated=data.get('gated'),private=data.get('private'))
        records.append(row)
    github=[]
    for repo in ['StanfordMIMI/Glyph','RobbieHolland/Glyph-research']:
        response=get('https://api.github.com/repos/'+repo)
        github.append(dict(repo=repo,anonymous_http_status=response['status'],
            publicly_accessible=response['status']==200,
            note='An anonymous 404 cannot distinguish a private repository from an absent repository.'))
    paper_findings=None
    if paper:
        path=Path(paper);text=subprocess.run(['pdftotext','-layout',str(path),'-'],capture_output=True,text=True,check=True).stdout
        links=subprocess.run(['pdfinfo','-url',str(path)],capture_output=True,text=True,check=True).stdout
        combined=text+'\n'+links
        paper_findings={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'filename':path.name,
            'mentions_github_package':bool(re.search(r'github\.com/(?:StanfordMIMI|RobbieHolland)/Glyph(?![-\w])',text)),
            'mentions_hf_labs_encoder':LABS_ENCODER_REPO in combined,
            'mentions_hf_sae_repo':'RobbieHolland/HypothesisExtractor' in combined,
            'mentions_activation_download':bool(re.search(r'(activation|sparse).{0,100}(huggingface|zenodo|download)',text,re.I)),
            'embedded_urls':re.findall(r'(?:https?://)[^\s]+',links),'note':'Visible text and embedded hyperlink targets checked; prose availability statements still need author review.'}
    return dict(checked_at=datetime.now(timezone.utc).isoformat(),anonymous_model_assets=records,github_repositories=github,paper=paper_findings,
        known_release_gaps=[
            {'item':'Historical underlying Merlin and Qwen encoder revisions','status':'unresolved','reason':'Published SAE registry does not identify historical upstream FM commits. Do not substitute current upstream HEAD.'},
            {'item':'Pretrained SAE decoders','status':'not_in_current_published_bundles','reason':'Encoder-only SAE assets support concept inference, not published-dictionary reconstruction R².'},
            {'item':'Public per-study activation release','status':'pending_local_preparation','reason':'Needs explicit public sample-ID mapping, release scope and provenance. No subject-level data are exported by this audit.'},
            {'item':'Source/package publication and activation data','status':'check_current_release_state','reason':'Package and repository publication are tracked separately from the pending research activation release; this read-only audit does not publish either.'}],
        registry=[dict(name=entry.name,encoder=entry.encoder,dimensions=entry.input_dim,concepts=entry.n_concepts,k=entry.top_k) for entry in ZOO.values()])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--paper',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=audit(args.paper);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='anonymous_model_assets'},indent=2))


if __name__=='__main__':main()
