"""Private lab cache parity check. Input records remain in memory; output is aggregate JSON."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from glyph.extraction.labs import LabsEncoder, prepare_labs
from glyph.zoo import load_pretrained


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--measurements',required=True,help='Trusted research quarterly measurement pickle')
    parser.add_argument('--embedding-cache',required=True,help='Trusted research lab embedding cache')
    parser.add_argument('--bundle-dir',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--sae-cache-dir',default='cache/zoo')
    parser.add_argument('--n-studies',type=int,default=5)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    cached=torch.load(args.embedding_cache,map_location='cpu',weights_only=False)
    cache_ids=[str(i) for i in cached.get('sample_ids',cached.get('anon_accession'))]
    index={key:n for n,key in enumerate(cache_ids)}
    source=pd.read_pickle(args.measurements)
    if 'labs' in source:
        counts=source.anon_accession.astype(str).value_counts()
        eligible=[sid for sid in cache_ids if counts.get(sid,0)==1]
        selected=source[source.anon_accession.astype(str).isin(eligible[:args.n_studies])]
        references={str(row.anon_accession): row.labs for row in selected.itertuples()}
        raw_rows=[]; meta=[]; reference=[]
        for sid in eligible[:args.n_studies]:
            values=references[sid]
            meta.append({'sample_id':sid,'patient_id':sid,'date':values[0]['Date']})
            raw_rows.extend([{**v,'patient_id':sid} for v in values])
            reference.append([{'lab_id':int(v['lab_id']),'Value':v['Value'],
                               'years_prior':float(v['years_prior'])} for v in values])
        rows=pd.DataFrame(raw_rows)
        metadata=pd.DataFrame(meta)
        del source
    else:
        rows=source[source.anon_accession.astype(str).isin(cache_ids[:args.n_studies])].copy()
        rows['patient_id']=rows.anon_accession.astype(str)
        metadata=rows.groupby('patient_id',sort=False)['Date'].first().reset_index().rename(columns={'Date':'date'})
        metadata['sample_id']=metadata.patient_id
        reference=[]
        for sid in metadata.sample_id:
            group=rows[rows.patient_id==sid].sort_values(['years_prior','lab_id'],kind='stable')
            reference.append([{'lab_id':int(r.lab_id),'Value':r.Value,'years_prior':float(r.years_prior)}
                              for r in group.itertuples()])
    tokens,prep_report=prepare_labs(metadata,rows,args.bundle_dir,return_report=True)
    prepared=[tokens[sid] for sid in metadata.sample_id]
    encoder=LabsEncoder(bundle_dir=args.bundle_dir)
    actual=encoder.embed_batch(prepared)
    reference_vectors=encoder.embed_batch(reference)
    torch.backends.mha.set_fastpath_enabled(False)
    unfused_vectors=encoder.embed_batch(reference)
    torch.backends.mha.set_fastpath_enabled(True)
    vectors=torch.as_tensor(cached.get('vectors',cached.get('output')))[[index[sid] for sid in metadata.sample_id]].float()
    if hasattr(vectors,'as_tensor'):vectors=vectors.as_tensor()
    sae=load_pretrained('labs-loinc',cache_dir=args.sae_cache_dir)
    z_actual,z_expected=sae.encode(actual),sae.encode(vectors)
    with torch.no_grad():
        dense_actual=torch.relu(sae.normalize(actual) @ sae.W_enc + sae.b_enc)
        dense_expected=torch.relu(sae.normalize(vectors) @ sae.W_enc + sae.b_enc)
    boundaries=[]
    for i in range(len(prepared)):
        if torch.equal(z_actual[i]>0,z_expected[i]>0):
            continue
        ordered=dense_expected[i].sort(descending=True).values
        error=float((dense_actual[i]-dense_expected[i]).abs().max())
        gap=float(ordered[39]-ordered[40])
        changed=torch.where((z_actual[i]>0)!=(z_expected[i]>0))[0]
        ranks=dense_expected[i].argsort(descending=True).argsort()+1
        changed_ranks=ranks[changed].tolist()
        boundaries.append({'reference_ranks_of_changed_concepts':changed_ranks,
                           'changes_within_boundary_band':all(38 <= rank <= 42 for rank in changed_ranks),
                           'reference_rank40_rank41_gap':gap,'max_dense_score_error':error,
                           'gap_within_numerical_perturbation':gap<=2*error,
                           'changed_concept_count':int(((z_actual[i]>0)!=(z_expected[i]>0)).sum())})
    report={'n_studies':len(prepared),'prepared_tokens_equal_research_tokens':prepared==reference,
            'token_lengths':[len(x) for x in prepared],
            'embedding_max_abs_error':float((actual-vectors).abs().max()),
            'reference_tokens_vs_cached_relative_l2':float((reference_vectors-vectors).norm()/vectors.norm()),
            'unfused_reference_vs_cached_relative_l2':float((unfused_vectors-vectors).norm()/vectors.norm()),
            'embedding_relative_l2_error':float((actual-vectors).norm()/vectors.norm()),
            'concept_relative_l2_error':float((z_actual-z_expected).norm()/z_expected.norm()),
            'active_concept_sets_equal_per_study':(z_actual.gt(0)==z_expected.gt(0)).all(dim=1).tolist(),
            'active_concept_overlap_per_study':(z_actual.gt(0)&z_expected.gt(0)).sum(dim=1).tolist(),
            'preparation_report':prep_report,'changed_boundaries':boundaries}
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    if args.verify:
        assert report['prepared_tokens_equal_research_tokens']
        assert report['embedding_relative_l2_error'] < .001
        assert all(item['gap_within_numerical_perturbation'] and item['changes_within_boundary_band'] for item in boundaries)



if __name__=='__main__':main()
