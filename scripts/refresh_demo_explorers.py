"""Refresh curated demo HTML from saved payloads; never refit or move concepts."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def refresh(repo):
    repo=Path(repo)
    tree=ast.parse((repo/'glyph/visualization/interactive_concept_network.py').read_text())
    template=next(ast.literal_eval(node.value) for node in tree.body
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_HTML' for t in node.targets))
    manifest_path=repo/'demos/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('public_or_synthetic_only') is not True:
        raise ValueError('Expected explicitly public/synthetic curated demos')
    changed=[]
    for name in manifest['current_demos']:
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in name):
            raise ValueError('Invalid demo name')
        relative=f'{name}/explorer.html';path=repo/'demos'/relative
        old=path.read_text();start=old.index('const DATA=')+len('const DATA=')
        data,_=json.JSONDecoder().raw_decode(old[start:])
        packed=json.dumps(data,separators=(',',':'),ensure_ascii=False,allow_nan=False).replace('</','<\\/')
        rendered=template.replace('__GLYPH_DATA__',packed)
        # Verify a presentation refresh did not alter layout, stats or examples.
        check,_=json.JSONDecoder().raw_decode(rendered[rendered.index('const DATA=')+len('const DATA='):])
        if check!=data:raise ValueError(f'Demo payload changed: {name}')
        path.write_text(rendered)
        manifest['files'][relative]=hashlib.sha256(path.read_bytes()).hexdigest()
        changed.append(relative)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    return changed


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    print('\n'.join(refresh(parser.parse_args().repo)))
