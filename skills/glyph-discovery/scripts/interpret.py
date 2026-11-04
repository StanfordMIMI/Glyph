"""Interpret a trusted saved Glyph run using explicitly configured text and credentials."""
import argparse
import json
from pathlib import Path


def interpret(run,config,output):
    from glyph import Glyph
    from glyph.interpretation import AutoInterp
    from glyph.interpretation.settings import ready
    allowed={'provider','model','api_key_env','vertex_project','vertex_location','text_map','metadata',
             'max_features','top_n','bottom_n','n_fit','n_test','top_pct','max_tokens','temperature'}
    if set(config)-allowed:raise ValueError('Unknown interpretation fields; use api_key_env rather than storing API keys')
    if not config.get('text_map'):raise ValueError('Supply text_map from each modality to a metadata text column')
    output=Path(output)
    if output.exists():raise ValueError('Interpretation output must be a new directory')
    obj=Glyph.load_run(run,device='cpu');obj.cfg.autointerp.enabled=True
    llm=obj.cfg.autointerp.llm
    from omegaconf import OmegaConf
    patch={key:config[key] for key in ['provider','model','vertex_project','vertex_location','max_tokens','temperature'] if key in config}
    patch.update(gemini_api_key=None,gemini_api_key_env=config.get('api_key_env','GEMINI_API_KEY'),vertex_location=config.get('vertex_location','us-central1'))
    obj.cfg.autointerp.llm=OmegaConf.merge(llm,patch)
    obj.cfg.autointerp.autointerp_text_map=config['text_map']
    if config.get('metadata'):obj.cfg.data.metadata=config['metadata']
    for field,target in [('top_n','top_n_hypotheses'),('bottom_n','bottom_n_hypotheses'),('n_fit','n_fit_samples'),('n_test','n_test_samples'),('top_pct','top_pct_activating')]:
        if field in config:obj.cfg.autointerp[target]=config[field]
    if not ready(obj.cfg):raise ValueError('Interpretation credentials are not configured; set the named environment variable or Vertex project/ADC')
    if obj.phewas_results is None or obj.phewas_results.empty:raise ValueError('Interpretation needs saved association results')
    maximum=config.get('max_features',20)
    if not isinstance(maximum,int) or maximum<1:raise ValueError('max_features must be a positive integer')
    if obj.cfg.autointerp.n_fit_samples<3 or obj.cfg.autointerp.n_test_samples<1 or not 0<obj.cfg.autointerp.top_pct_activating<=1:raise ValueError('Invalid interpretation sample counts or percentile')
    training=obj.phewas_results[obj.phewas_results.split=='train'].copy()
    order='odds_ratio_p_value_bonferroni' if 'odds_ratio_p_value_bonferroni' in training else 'train_odds_ratio_rank'
    selected=training.sort_values(order).drop_duplicates(['modality','top_k','feature_idx']).head(maximum)
    keys=set(zip(selected.modality,selected.top_k,selected.feature_idx))
    obj.phewas_results=obj.phewas_results[[key in keys for key in zip(obj.phewas_results.modality,obj.phewas_results.top_k,obj.phewas_results.feature_idx)]]
    interpreter=AutoInterp(obj.cfg)
    output.mkdir(parents=True,exist_ok=False)
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    result=interpreter.run_all(sae_features=obj.sae_features,sample_ids={split:data['sample_ids'] for split,data in obj.embeddings.items()},hypotheses=obj.phewas_results,output_dir=str(output))
    if result.empty:raise ValueError('No interpretable features had enough activating and reference samples; see the log')
    result.drop(columns=[c for c in result if c.endswith('_ids') or 'prompt' in c or 'response' in c or c in ['test_labels','test_predictions']],errors='ignore').to_csv(output/'summary.csv',index=False)
    return {'interpretations':len(result),'scored':int(result.test_accuracy.notna().sum()),'results':str(output/'autointerp_results.pkl')}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['run','config','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv);print(json.dumps(interpret(args.run,json.loads(args.config.read_text()),args.output),indent=2))


if __name__=='__main__':main()
