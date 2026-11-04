"""Build a Pages-ready site from the explicitly public, curated demo outputs.

Only Python's standard library is required. No model loading or refitting occurs.
"""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

LABELS = {
    'dinov3_digits': ('Handwritten digits', '1,000 public digit images; ten outcome labels.'),
    'sbu_cxr': ('Chest X-ray mortality', '948 public patients; later in-hospital death and a clinical baseline.'),
    'pad_ufes': ('Smartphone skin-lesion diagnosis', '1,373 public patients; lesion diagnoses and eight clinical baseline fields.'),
}


def build_site(repo, output):
    repo,output=Path(repo),Path(output)
    manifest=json.loads((repo/'demos/manifest.json').read_text())
    if manifest.get('public_or_synthetic_only') is not True:
        raise ValueError('The curated demo manifest must explicitly identify public/synthetic inputs')
    output.mkdir(parents=True,exist_ok=False);cards=[];sizes=[]
    for name in manifest['current_demos']:
        if not re.fullmatch(r'[a-z0-9_]+',name):raise ValueError('Invalid demo directory')
        source=repo/'demos'/name;target=output/name;target.mkdir()
        explorer=(source/'explorer.html').read_text();match=re.search(r'const DATA=(.*?),NS=',explorer)
        if not match:raise ValueError(f'Explorer payload missing for {name}')
        data=json.loads(match.group(1));assets=target/'images'
        for key,record in data.get('image_bank',{}).items():
            if not re.fullmatch(r'[a-f0-9]+',key):raise ValueError('Invalid thumbnail ID')
            encoded=record['src'];mime=encoded.split(';')[0]
            extension={'data:image/jpeg':'jpg','data:image/png':'png'}.get(mime)
            if extension is None:raise ValueError('Unexpected thumbnail format')
            raw=base64.b64decode(encoded.split(',',1)[1],validate=True);assets.mkdir(exist_ok=True)
            relative=f'images/{key}.{extension}';(target/relative).write_bytes(raw);record['src']=relative
        packed=json.dumps(data,separators=(',',':'),allow_nan=False).replace('</','<\\/')
        rendered=explorer[:match.start(1)]+packed+explorer[match.end(1):]
        (target/'explorer.html').write_text(rendered)
        # Explicit allowlist: no source, raw cohorts, checkpoints, tokens or Git history.
        for filename in ['report.pdf','demo.ipynb','inputs.png','provenance.json']:
            path=source/filename
            if path.is_file() and not path.is_symlink():shutil.copy2(path,target/filename)
        report=source/'report_files/report.html'
        if report.is_file() and not report.is_symlink():shutil.copy2(report,target/'report.html')
        title,description=LABELS.get(name,(name.replace('_',' ').title(),'Public Glyph demo.'))
        preview=f'<img src="{name}/inputs.png" alt="{html.escape(title)} input examples" loading="lazy">' if (target/'inputs.png').exists() else ''
        cards.append(f'<article>{preview}<div class="body"><h2>{html.escape(title)}</h2><p>{html.escape(description)}</p><a class="primary" href="{name}/explorer.html">Open interactive explorer ↗</a><nav><a href="{name}/report.pdf">PDF report</a><a href="{name}/report.html">Image report</a><a href="{name}/demo.ipynb" download>Notebook ↓</a></nav></div></article>')
        sizes.append(dict(demo=name,standalone_bytes=len(explorer.encode()),pages_html_bytes=len(rendered.encode()),thumbnail_bytes=sum(p.stat().st_size for p in assets.glob('*')) if assets.exists() else 0))
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Glyph · interactive demos</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f8fb;color:#152536;font:16px/1.6 system-ui,-apple-system,sans-serif}main{max-width:1180px;margin:auto;padding:60px 24px}header{margin-bottom:36px}header b{letter-spacing:.14em;color:#0d9488;font-size:13px}h1{font-size:clamp(32px,5vw,52px);line-height:1.12;margin:12px 0 18px}header p{max-width:760px;color:#526477}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:22px}article{background:white;border:1px solid #dfE7ef;border-radius:18px;overflow:hidden;box-shadow:0 7px 22px #15253606}article>img{width:100%;height:200px;object-fit:contain;background:#fff;padding:12px}.body{padding:24px}h2{font-size:22px;line-height:1.3;margin:0 0 12px}.body p{color:#526477;min-height:76px}.primary{display:block;background:#0d9488;color:white;padding:11px 14px;border-radius:9px;text-decoration:none;font-weight:650;font-size:14px}a:hover{text-decoration:underline}nav{display:flex;flex-wrap:wrap;gap:14px;margin-top:18px}nav a,footer a{color:#0f766e;font-size:13px}footer{margin-top:40px;color:#667789;font-size:13px}a:focus-visible{outline:3px solid #7c3aed;outline-offset:3px}</style></head><body><main><header><b>GLYPH / PUBLIC DEMOS</b><h1>Explore the concept space.</h1><p>Real DINOv3 embeddings, learned sparse concepts and held-out outcome comparisons. Click a concept to see its associations and two high-activation image examples.</p></header><section class="cards">'''+''.join(cards)+'''</section><footer><p>The explorers load only the selected concept’s thumbnails. The downloadable standalone explorers in the repository also work offline.</p><p>Images: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html">UCI optical digits</a>; <a href="https://www.cancerimagingarchive.net/collection/covid-19-ny-sbu/">TCIA COVID-19-NY-SBU</a> and <a href="https://data.mendeley.com/datasets/zr7vgbcyr2/1">PAD-UFES-20</a> (<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>). Images are resized for display. These are research demonstrations; interpretation labels were not automatically generated.</p></footer></main></body></html>'''
    (output/'index.html').write_text(page);(output/'.nojekyll').touch()
    hashes={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'site_manifest.json').write_text(json.dumps(dict(files=hashes,sizes=sizes),indent=2)+'\n')
    return sizes


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(build_site(args.repo,args.output),indent=2))
