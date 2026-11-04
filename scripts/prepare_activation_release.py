"""Stage a local activation/model release using an explicit public-ID mapping. No uploads."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,save_npz


def prepare(run,id_map,output):
    from glyph import Glyph
    obj=Glyph.load_run(run,device='cpu')
    mapping=pd.read_csv(id_map,dtype=str,keep_default_na=False)
    if set(mapping)!= {'sample_id','public_sample_id'}:raise ValueError('Mapping must contain exactly sample_id and public_sample_id')
    if any(mapping[col].eq('').any() or mapping[col].duplicated().any() for col in mapping):raise ValueError('Both private and public IDs must be complete and unique')
    all_ids=[str(sample) for data in obj.embeddings.values() for sample in data['sample_ids']]
    if len(all_ids)!=len(set(all_ids)) or set(all_ids)!=set(mapping.sample_id):raise ValueError('Mapping must cover every run sample exactly once')
    public=mapping.set_index('sample_id').public_sample_id.to_dict()
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    records=[]
    for split,data in obj.embeddings.items():records.extend(dict(public_sample_id=public[str(sample)],split=split) for sample in data['sample_ids'])
    pd.DataFrame(records).to_csv(output/'samples.csv',index=False)
    entries=[]
    for modality,sae in obj.sae_models.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+',modality):raise ValueError('Release modality names must be safe filenames')
        matrix=np.concatenate([obj.sae_features[modality][split].cpu().numpy() for split in obj.embeddings],axis=0)
        if not np.isfinite(matrix).all():raise ValueError('Activations must be finite')
        save_npz(output/f'activations_{modality}.npz',csr_matrix(matrix))
        weights={name:value.detach().cpu().numpy() for name,value in sae.models[sae.selected_top_k].state_dict().items()}
        np.savez_compressed(output/f'sae_{modality}.npz',**weights)
        norm={name:(value.detach().cpu().tolist() if hasattr(value,'detach') else value) for name,value in obj.sae_normalization[modality].items()}
        (output/f'normalization_{modality}.json').write_text(json.dumps(norm,indent=2,allow_nan=False)+'\n')
        alive=np.asarray(obj.sae_alive_features[(modality,sae.selected_top_k)],dtype=bool)
        pd.DataFrame({'feature_idx':np.arange(matrix.shape[1]),'alive_on_train':alive}).to_csv(output/f'concepts_{modality}.csv',index=False)
        entries.append(dict(modality=modality,rows=matrix.shape[0],concepts=matrix.shape[1],top_k=sae.selected_top_k,
            has_decoder=all(name in weights for name in ['W_dec','b_dec']),dictionary=sae.selection_report))
    manifest=dict(schema_version=1,status='local_staging_only_not_published',row_order='samples.csv order for every activation matrix',
        activation_format='scipy CSR .npz; original concept indices retained',models=entries,
        public_id_mapping_sha256=hashlib.sha256(Path(id_map).read_bytes()).hexdigest(),
        source_run_manifest_sha256=hashlib.sha256((Path(run)/'run.json').read_bytes()).hexdigest(),
        omitted=['private sample IDs','raw inputs','outcome labels','clinical metadata','prompts','sample-level interpretation examples'],
        files={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir()) if path.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['run','public-id-map','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();result=prepare(args.run,args.public_id_map,args.output);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
