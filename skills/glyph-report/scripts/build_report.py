"""Build a compact visual brief from saved Glyph results; no new model fitting."""
import argparse
import base64
import hashlib
import html
import json
from itertools import chain
from pathlib import Path
import re
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle,ConnectionPatch,FancyBboxPatch,FancyArrowPatch
from matplotlib.path import Path as PlotPath
import numpy as np
import pandas as pd

from glyph.visualization.node_sizes import prevalence_node_areas
from glyph.visualization.colors import outcome_palette,digit_index,DIGIT_COLORS
from glyph.visualization.correlation_edges import limit_correlation_edges,correlation_edge_style
from matplotlib.collections import LineCollection

PALETTE=['#0d9488','#3b82f6','#8b5cf6','#e87935','#db4b88']
GRAY='#dce2e8'


def read_table(path):
    path=Path(path)
    return pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path)


def layout_features(values,seed=42):
    from glyph.visualization.interactive_concept_network import _layout
    values=np.asarray(values,dtype=float)
    complete=values[np.isfinite(values).all(axis=1)]
    variable=np.std(complete,axis=0)>1e-8 if len(complete)>1 else np.zeros(values.shape[1],dtype=bool)
    coordinates=np.zeros((values.shape[1],2))
    if variable.any():
        correlation=np.corrcoef(complete[:,variable].T) if variable.sum()>1 else np.ones((1,1))
        coordinates[variable]=_layout(np.nan_to_num(correlation),seed)
    constant=np.where(~variable)[0]
    if len(constant):
        coordinates[constant,0]=1.08
        coordinates[constant,1]=np.linspace(.1,.9,len(constant))
    return coordinates,~variable,len(complete)


def run_tables(path,interpretations=None):
    from glyph import Glyph
    obj=Glyph.load_run(path,device='cpu')
    if obj.phewas_results is None or obj.phewas_results.empty:
        raise ValueError('Saved association results are required; run hypothesis_search first')
    rows=[];correlation_edges=[]
    labels=(pd.read_pickle(interpretations) if Path(interpretations).suffix=='.pkl' else read_table(interpretations)) if interpretations else obj.autointerp_results
    published=obj.published_interpretations is not None and interpretations is None
    for (modality,k),mask in obj.sae_alive_features.items():
        alive=np.asarray(mask.cpu() if hasattr(mask,'cpu') else mask,dtype=bool)
        indices=np.where(alive)[0]
        if not len(indices):continue
        raw=obj.sae_features[modality]['train']
        raw=raw.cpu().numpy() if hasattr(raw,'cpu') else raw
        coords=None
        constant=np.zeros(len(indices),dtype=bool)
        saved=Path(path)/f'concept_map_{modality}_k{k}_train.html'
        if saved.exists():
            match=re.search(r'const DATA=(.*?),NS=',saved.read_text())
            data=json.loads(match.group(1)) if match else {}
            lookup={int(node['feature_idx']):node for node in data.get('nodes',[])}
            if set(lookup)==set(indices):
                coords=np.array([[lookup[i]['x'],lookup[i]['y']] for i in indices])
        saved_edges=[]
        if coords is not None:
            kept=limit_correlation_edges(data.get('edges',[]),len(data.get('nodes',[])))
            saved_edges=[(int(data['nodes'][a]['feature_idx']),int(data['nodes'][b]['feature_idx']),w) for a,b,w in kept]
        if coords is None:
            coords,constant,n=layout_features(raw[:,indices],obj.cfg.seed)
            from glyph.visualization.interactive_concept_network import _graph_edges
            matrix=np.nan_to_num(np.corrcoef(raw[:,indices].T)) if len(indices)>1 else np.ones((1,1))
            kept=_graph_edges(matrix,obj.cfg.visualization.correlation_percentile)
            saved_edges=[(int(indices[a]),int(indices[b]),w) for a,b,w in kept]
        correlation_edges.extend(dict(source=f'{modality}:{k}:{a}',target=f'{modality}:{k}:{b}',weight=w) for a,b,w in saved_edges)
        selected=labels.copy() if labels is not None else pd.DataFrame()
        if not selected.empty:
            selected=selected[(selected.modality==modality)&(selected.top_k==k)].drop_duplicates('feature_idx')
            selected=selected.set_index('feature_idx')
        offset=2*len({r['modality'] for r in rows})
        for j,index in enumerate(indices):
            interp=selected.loc[index] if not selected.empty and index in selected.index else {}
            rows.append(dict(concept_id=f'{modality}:{k}:{index}',name=f'{modality} concept {index}',
                modality=modality,x=float(coords[j,0]+offset),y=float(coords[j,1]),
                interpretation='' if pd.isna(interp.get('interpretation','')) else str(interp.get('interpretation','')),
                interpretation_accuracy=interp.get('test_accuracy',np.nan),
                active_count=int((raw[:,index]>1e-8).sum()), activation_rate=float((raw[:,index]>1e-8).mean()),
                interpretation_accuracy_context='Original release' if published else 'Held-out interpretation evaluation',
                layout_status='constant' if constant[j] else 'variable'))
    concepts=pd.DataFrame(rows)
    associations=obj.phewas_results.copy()
    associations['concept_id']=[f'{r.modality}:{r.top_k}:{r.feature_idx}' for r in associations.itertuples()]
    metadata=dict(scope='All Discovery-alive concepts in the saved run',source_name=Path(path).name,correlation_edges=correlation_edges,
        notes='Separate modality layouts; distances between panels have no meaning. Saved unadjusted associations; no models were refitted.')
    from glyph.mechinterp.sae import reconstruction_metrics
    metadata['cohorts']={split:dict(n=len(data['sample_ids']),events={name:int(np.sum(obj.outcomes[split][name]==1)) for name in obj.cfg.data.outcome_cols}) for split,data in obj.embeddings.items()}
    metadata['quality']={}
    for modality,sae in obj.sae_models.items():
        stats=obj.sae_normalization[modality]
        quality=dict(selection=sae.selection_report,reconstruction={})
        if stats['kind']=='trained':
            for split,data in obj.embeddings.items():
                normalized=(data['embeddings'][modality]-stats['mean'])/stats['std']
                quality['reconstruction'][split]=reconstruction_metrics(sae.models[sae.selected_top_k],normalized)
        quality['embedding_probe']={str(key):value.get('all_metrics',{}) for key,value in (obj.linear_probe_results or {}).items() if modality in key}
        quality['concept_probe']={str(key):value.get('all_metrics',{}) for key,value in (obj.sae_linear_probe_results or {}).items() if modality in key}
        metadata['quality'][modality]=quality
    metadata['testing_families']={split:sorted(frame.bonferroni_n_tests_global.dropna().astype(int).unique().tolist()) if 'bonferroni_n_tests_global' in frame else [] for split,frame in associations.groupby('split')}
    return concepts,associations,metadata


def validate(concepts,associations):
    concepts=concepts.copy();associations=associations.copy()
    required={'concept_id','x','y','modality'}
    if required-set(concepts):raise ValueError(f'Missing concept columns: {sorted(required-set(concepts))}')
    required={'concept_id','outcome','split','odds_ratio','odds_ratio_p_value_bonferroni'}
    if required-set(associations):raise ValueError(f'Missing association columns: {sorted(required-set(associations))}')
    for table in [concepts,associations]:
        if table.concept_id.isna().any():raise ValueError('Concept IDs must not be missing')
        table['concept_id']=table.concept_id.astype(str)
    if concepts.empty or concepts.concept_id.duplicated().any():raise ValueError('Supply a nonempty, unique full concept space')
    if not np.isfinite(concepts[['x','y']].to_numpy(dtype=float)).all():raise ValueError('Coordinates must be finite')
    if associations.duplicated(['concept_id','outcome','split']).any():raise ValueError('Duplicate association keys')
    if not set(associations.concept_id).issubset(set(concepts.concept_id)):
        raise ValueError('Association IDs are absent from the supplied concept space')
    p=pd.to_numeric(associations.odds_ratio_p_value_bonferroni,errors='coerce')
    if ((p<0)|(p>1)).any():raise ValueError('Corrected p-values must be in [0,1]')
    associations['odds_ratio_p_value_bonferroni']=p
    associations['odds_ratio']=pd.to_numeric(associations.odds_ratio,errors='coerce')
    for col,default in [('name',''),('interpretation',''),('interpretation_accuracy',np.nan),('interpretation_accuracy_context','Saved interpretation evaluation'),('layout_status','variable'),('active_count',np.nan),('activation_rate',np.nan),('annotation',''),('color_group','')]:
        if col not in concepts:concepts[col]=default
    concepts['name']=concepts['name'].fillna('').where(concepts['name'].fillna('').ne(''),concepts.concept_id)
    concepts['interpretation']=concepts.interpretation.fillna('')
    accuracy=pd.to_numeric(concepts.interpretation_accuracy,errors='coerce')
    if ((accuracy<0)|(accuracy>1)).any():raise ValueError('Interpretation accuracy must be a fraction in [0,1]')
    return concepts,associations


def significant(table,alpha):
    return table[(table.odds_ratio_p_value_bonferroni<alpha)&np.isfinite(table.odds_ratio)&(table.odds_ratio>0)].copy()


def prepare(concepts,associations,split,comparison,alpha,zoom_count,focus_concepts=None,focus_outcome=None):
    current=associations[associations.split==split].copy()
    if current.empty:raise ValueError(f'No saved associations for split {split}')
    hits=significant(current,alpha)
    dominant=hits.sort_values(['odds_ratio_p_value_bonferroni','concept_id','outcome']).drop_duplicates('concept_id')
    display=dict(zip(associations.outcome,associations.get('outcome_display',associations.outcome)))
    colors=outcome_palette(sorted(hits.outcome.unique()))
    assigned={row.concept_id:colors[row.outcome] for row in dominant.itertuples()}
    if concepts.color_group.fillna('').ne('').any():
        groups=concepts.color_group.fillna('Unclassified').replace('','Unclassified').unique()
        colors=outcome_palette(groups)
        groups_by_id=concepts.set_index('concept_id').color_group.fillna('Unclassified').replace('','Unclassified').to_dict()
        assigned={row.concept_id:colors[groups_by_id[row.concept_id]] for row in dominant.itertuples()}
    concepts=concepts.copy()
    concepts['_color']=concepts.concept_id.map(assigned).fillna(GRAY)
    lookup=concepts.set_index('concept_id')
    ranked=dominant.copy()
    ranked['_has_interp']=ranked.concept_id.map(lookup.interpretation.str.len().gt(0))
    ranked=ranked.sort_values(['_has_interp','odds_ratio_p_value_bonferroni','concept_id'],ascending=[False,True,True])
    chosen=[]
    for row in (ranked.itertuples() if zoom_count else []):
        point=lookup.loc[row.concept_id]
        if any(np.hypot(point.x-lookup.loc[x['concept_id'],'x'],point.y-lookup.loc[x['concept_id'],'y'])<.18 for x in chosen):continue
        if point.color_group and any(lookup.loc[x['concept_id'],'color_group']==point.color_group for x in chosen):continue
        chosen.append(row._asdict())
        if len(chosen)>=zoom_count:break
    if focus_concepts:
        if not isinstance(focus_concepts,(list,tuple)) or len(focus_concepts)>zoom_count or len(set(focus_concepts))!=len(focus_concepts):raise ValueError('Focus concepts must be a unique list no longer than zoom_count')
        chosen=[]
        for identifier in focus_concepts:
            rows=current[current.concept_id.eq(identifier)]
            if focus_outcome is not None:rows=rows[rows.outcome.eq(focus_outcome)]
            if rows.empty:raise ValueError(f'No training association for focused concept {identifier} and outcome {focus_outcome}')
            chosen.append(rows.sort_values(['odds_ratio_p_value_bonferroni','outcome']).iloc[0].to_dict())
    comparison_rows=associations[associations.split==comparison].set_index(['concept_id','outcome'])
    for row in chosen:
        point=lookup.loc[row['concept_id']]
        row['zoom_number']=chosen.index(row)+1
        row['name']=str(point['name']);row['interpretation']=str(point.interpretation)
        row['annotation']=str(point.annotation)
        row['interpretation_accuracy']=number(point.interpretation_accuracy)
        row['interpretation_accuracy_context']=str(point.interpretation_accuracy_context)
        row['outcome_display']=str(display.get(row['outcome'],row['outcome']))
        row['x']=float(point.x);row['y']=float(point.y)
        key=(row['concept_id'],row['outcome'])
        other=comparison_rows.loc[key] if key in comparison_rows.index else None
        row['comparison']=None if other is None else {key:number(other.get(key)) for key in
            ['odds_ratio','odds_ratio_ci_lower','odds_ratio_ci_upper','odds_ratio_p_value_bonferroni','c_f']}
        row['comparison_reproduced']=bool(other is not None and np.isfinite(other.odds_ratio)
            and other.odds_ratio_p_value_bonferroni<alpha and (other.odds_ratio-1)*(row['odds_ratio']-1)>0)
    available=current[np.isfinite(current.odds_ratio_p_value_bonferroni)]
    comparable=hits.merge(associations[associations.split==comparison],on=['concept_id','outcome'],suffixes=('_discovery','_comparison'))
    measured=comparable[np.isfinite(comparable.odds_ratio_comparison)&np.isfinite(comparable.odds_ratio_p_value_bonferroni_comparison)]
    reproduced=measured[(measured.odds_ratio_p_value_bonferroni_comparison<alpha)&
                        ((measured.odds_ratio_comparison-1)*(measured.odds_ratio_discovery-1)>0)]
    facts=dict(n_concepts=len(concepts),n_association_rows=len(current),n_finite_p_values=len(available),
        n_significant_pairs=len(hits),n_significant_concepts=hits.concept_id.nunique(),
        n_outcomes=current.outcome.nunique(),n_outcomes_with_hits=hits.outcome.nunique(),
        n_comparable_hits=len(measured),n_reproduced_hits=len(reproduced),split=split,comparison_split=comparison,alpha=alpha,
        zoom_selection='Strongest corrected associations with cached interpretations where available, preferring nonoverlapping regions.',
        zooms=[clean(row) for row in chosen])
    return concepts,facts,colors,{str(k):str(v) for k,v in display.items()}


def number(value):
    if value is None or pd.isna(value):return None
    value=float(value)
    return value if np.isfinite(value) else None


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items() if not str(k).startswith('_')}
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,(np.integer,np.floating)):return number(value)
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def pvalue(value):
    return 'not available' if value is None else '0 (stored)' if value==0 else f'{value:.2g}'


def effect(row):
    value=row.get('odds_ratio');lower=row.get('odds_ratio_ci_lower');upper=row.get('odds_ratio_ci_upper')
    if value is None or pd.isna(value):return 'OR unavailable'
    fmt=lambda number:f'{number:.2f}' if .01<=abs(number)<1000 else f'{number:.2g}'
    return 'OR '+fmt(value)+(f' (95% CI {fmt(lower)}–{fmt(upper)})' if lower is not None and upper is not None else '')


def bounds(concepts,row):
    same=concepts[concepts.modality==concepts.set_index('concept_id').loc[row['concept_id'],'modality']]
    span=max(float(np.ptp(same.x)),float(np.ptp(same.y)),.2)
    distances=np.sort(np.hypot(same.x-row['x'],same.y-row['y']))
    radius=float(np.clip(distances[min(3,len(distances)-1)]*1.05,.06*span,.14*span))
    return row['x']-radius,row['x']+radius,row['y']-radius,row['y']+radius


def draw_map(ax,concepts,zooms=None,window=None,marker_scale=1.):
    rates=pd.to_numeric(concepts.activation_rate,errors='coerce').to_numpy()
    visible=np.ones(len(concepts),dtype=bool) if window is None else (
        concepts.x.between(window[0],window[1]) & concepts.y.between(window[2],window[3])).to_numpy()
    sizes=prevalence_node_areas(rates,n_nodes=int(visible.sum()))*marker_scale
    coordinates={row.concept_id:(row.x,row.y) for row in concepts.loc[visible].itertuples()}
    edges=[e for e in concepts.attrs.get('correlation_edges',[]) if e['source'] in coordinates and e['target'] in coordinates]
    segments=[[coordinates[e['source']],coordinates[e['target']]] for e in edges]
    if segments:
        widths,opacity=correlation_edge_style([e['weight'] for e in edges],zoom=window is not None)
        ax.add_collection(LineCollection(segments,colors='#94a3b8',linewidths=widths*np.sqrt(marker_scale),alpha=opacity,zorder=0))
    background=concepts._color.eq(GRAY).to_numpy()
    for mask,layer in [(background,1),(~background,2)]:
        if mask.any():
            ax.scatter(concepts.loc[mask,'x'],concepts.loc[mask,'y'],s=sizes[mask],
                c=concepts.loc[mask,'_color'],edgecolors='white',linewidths=.18*np.sqrt(marker_scale),
                alpha=.95,rasterized=False,zorder=layer)
    if window:
        ax.set_xlim(window[0],window[1]);ax.set_ylim(window[2],window[3])
    else:
        ax.margins(.06)
    for index,row in enumerate(zooms or []):
        if not window:
            box=bounds(concepts,row)
            ax.add_patch(Rectangle((box[0],box[2]),box[1]-box[0],box[3]-box[2],fill=False,edgecolor='#334155',linewidth=.8))
        ax.scatter([row['x']],[row['y']],s=240,facecolors='none',edgecolors='#0f172a',linewidths=1.5,zorder=4)
        ax.annotate(str(row.get('zoom_number',index+1)),(row['x'],row['y']),xytext=(10,-12),textcoords='offset points',fontsize=10,fontweight='bold',color='#0f172a',zorder=5)
    ax.set_aspect('equal',adjustable='box');ax.invert_yaxis();ax.axis('off')


def save_figure(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=220,bbox_inches='tight',facecolor='white')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight',facecolor='white')


def map_legend(fig,ax,colors,display,handles):
    indices=[digit_index(name) for name in colors]
    if len(indices)==10 and set(indices)==set(range(10)):
        from matplotlib.colors import ListedColormap,BoundaryNorm
        from matplotlib.cm import ScalarMappable
        cax=ax.inset_axes([.14,-.085,.72,.025])
        cmap=ListedColormap(DIGIT_COLORS);norm=BoundaryNorm(np.arange(-.5,10.5),10)
        bar=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',ticks=range(10))
        bar.ax.tick_params(labelsize=8,length=0,pad=2);bar.set_label('Digit',fontsize=8,labelpad=1)
        bar.outline.set_visible(False)
        ax.legend(handles=[handles[-1]],loc='upper center',bbox_to_anchor=(.5,-.015),frameon=False,fontsize=7.5)
    else:
        ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.025),ncol=3,frameon=False,fontsize=7.5)


def read_study(path):
    if path is None:return None
    path=Path(path)
    return dict(intervals=pd.read_csv(path/'auc_intervals.csv'),comparisons=pd.read_csv(path/'auc_comparisons.csv'),
        metrics=pd.read_csv(path/'metrics.csv'),associations=pd.read_csv(path/'individual_associations.csv'),
        ranking=pd.read_csv(path/'training_rankings.csv') if (path/'training_rankings.csv').exists() else None,
        manifest=json.loads((path/'manifest.json').read_text()),
        bootstrap=json.loads((path/'bootstrap_audit.json').read_text()) if (path/'bootstrap_audit.json').exists() else [],
        search=json.loads((path/'search_funnel.json').read_text()) if (path/'search_funnel.json').exists() else None,
        adjustment=json.loads((path/'confounder_adjustment.json').read_text()) if (path/'confounder_adjustment.json').exists() else None,
        adjusted=pd.read_csv(path/'adjusted_individual_associations.csv') if (path/'adjusted_individual_associations.csv').exists() else None)


def annotate(concepts,table):
    concepts=concepts.copy()
    if concepts.modality.nunique()!=1:raise ValueError('Measurement annotations require one explicitly selected modality')
    if table.feature_idx.duplicated().any():raise ValueError('Annotation feature indices must be unique')
    name='measurement' if 'measurement' in table else 'strongest_training_measurement'
    train='train_r' if 'train_r' in table else 'train_correlation'
    test='test_r' if 'test_r' in table else 'test_correlation'
    lookup=table.set_index('feature_idx')
    for i,row in concepts.iterrows():
        feature=int(row.concept_id.split(':')[-1])
        if feature not in lookup.index:continue
        annotation=lookup.loc[feature];measurement=str(annotation[name])
        group=next((label for key,label in [('concav','Concavity'),('concave','Concavity'),('compact','Compactness'),('texture','Texture'),('smooth','Smoothness'),('symmetr','Symmetry'),('fractal','Boundary complexity')] if key in measurement),'Cell size')
        concepts.loc[i,'name']=f'C{feature} · {measurement}'
        concepts.loc[i,'annotation']=f"Tracks {measurement}: training r={annotation[train]:.2f}; test r={annotation[test]:.2f}. Correlation descriptor, not semantic interpretation."
        concepts.loc[i,'color_group']=group
    return concepts


def metric_lines(metadata):
    lines=[]
    cohorts=metadata.get('cohorts',{})
    if cohorts:lines.append(' / '.join(f"{split} {value['n']}"+(f" ({next(iter(value['events'].values()))} events)" if len(value.get('events',{}))==1 else '') for split,value in cohorts.items()))
    for modality,quality in metadata.get('quality',{}).items():
        val=quality.get('reconstruction',{}).get('val',{}).get('r2')
        selection=quality.get('selection',{})
        if val is not None:
            test=quality.get('reconstruction',{}).get('test',{}).get('r2')
            warning=f" · below {selection.get('r2_targets',[.85])[0]:.2f} target" if selection.get('reached_minimum') is False else ''
            lines.append(f"Reconstruction R²: validation {val:.3f}"+(f"; test {test:.3f}" if test is not None else '')+f" · k={selection.get('selected_top_k','?')}"+warning)
        raw=list(quality.get('embedding_probe',{}).values());sparse=list(quality.get('concept_probe',{}).values())
        if raw and sparse and 'test' in raw[0] and 'test' in sparse[0]:
            lines.append(f"Test probe AUC: input vectors {raw[0]['test']:.3f}; full dictionary {sparse[0]['test']:.3f}")
    return lines


def prediction_rows(study,split):
    data=study['intervals'].query('split == @split')
    return data[data.arm.ne('Literature + Glyph')]



def capped_model_description(manifest):
    cap=manifest.get('comparison_top_n',10)
    if manifest.get('combined_top_n_strategy','retain_literature')=='pooled':
        return f'The capped combined model pools literature and Glyph candidates and selects {cap} total predictors using their training-only ranking; either or both sources may be selected.'
    return f'The capped combined model retains up to {cap} training-ranked literature predictors and fills any remaining places with the highest-ranked Glyph concepts.'


def displayed_auc_comparisons(study,split):
    shown=set(prediction_rows(study,split).arm);cap=int(study['manifest'].get('comparison_top_n',10))
    pairs=[('Literature',f'Glyph top {cap}'),('Literature',f'Literature + Glyph to {cap}'),
           ('Glyph',f'Glyph top {cap}'),(f'Glyph top {cap}',f'Literature + Glyph to {cap}')]
    rows=[]
    for first,second in pairs:
        if first not in shown or second not in shown:continue
        found=study['comparisons'][(study['comparisons'].split==split)&(
            ((study['comparisons'].reference==first)&(study['comparisons'].comparison==second))|
            ((study['comparisons'].reference==second)&(study['comparisons'].comparison==first)))]
        if len(found):rows.append(dict(reference=first,comparison=second,p_value=float(found.iloc[0].p_value)))
    return rows


def significance_label(value):
    return '***' if value<.001 else '**' if value<.01 else '*' if value<.05 else 'n.s.'


def plot_auc(ax,study,split):
    data=prediction_rows(study,split)
    for i,row in enumerate(data.itertuples()):
        ax.errorbar(row.auc,i,xerr=[[max(0,row.auc-row.ci_lower)],[max(0,row.ci_upper-row.auc)]],fmt='o',color=('#5eead4' if ('top ' in row.arm or ' to ' in row.arm) else '#0d9488') if row.arm.startswith('Literature') and '+' not in row.arm else (('#93c5fd' if ('top ' in row.arm or ' to ' in row.arm) else '#3b82f6') if row.arm.startswith('Glyph') else ('#c4b5fd' if ('top ' in row.arm or ' to ' in row.arm) else '#8b5cf6')),capsize=3,markersize=6)
        calibration=study['metrics'][(study['metrics'].arm==row.arm)&(study['metrics'].split==split)].iloc[0].brier
        ax.text(.985,i+.25,f'{row.auc:.3f} [{row.ci_lower:.3f}, {row.ci_upper:.3f}] · Brier {calibration:.3f}',ha='right',fontsize=7.5,transform=ax.get_yaxis_transform())
    counts=study['metrics'].query('split == @split').set_index('arm')['features']
    labels=[f'{arm} ({int(counts[arm])})' for arm in data.arm]
    ax.set_yticks(range(len(data)),labels,fontsize=8);ax.set_ylim(len(data)-.3,-.65)
    ax.set_xlim(max(.5,float(data.ci_lower.min())-.008),1.003);ax.set_xlabel('Test AUROC · 95% paired-bootstrap interval',fontsize=8)
    low,high=ax.get_xlim();ax.set_xticks([tick for tick in ax.get_xticks() if low<=tick<=1])
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
    ax.set_title('Literature baseline vs Glyph',loc='left',fontweight='bold',fontsize=10)
    positions={arm:i for i,arm in enumerate(data.arm)}
    for index,pair in enumerate(displayed_auc_comparisons(study,split)):
        x=1.025+index*.15;a=positions[pair['reference']];b=positions[pair['comparison']]
        ax.plot([x,x+.018,x+.018,x],[a,a,b,b],transform=ax.get_yaxis_transform(),clip_on=False,color='#64748b',linewidth=.65)
        label=significance_label(pair['p_value'])+'\np='+pvalue(pair['p_value'])
        ax.text(x+.029,(a+b)/2,label,transform=ax.get_yaxis_transform(),clip_on=False,fontsize=6.2,va='center',color='#334155')



def matched_reproduction(study,split):
    table=study['associations'];result={}
    for arm in table.arm.unique():
        discovery=table[(table.arm==arm)&(table.split=='train')&(table.corrected_p_value<.05)]
        heldout=table[(table.arm==arm)&(table.split==split)]
        joined=discovery.merge(heldout,on='marker',suffixes=('_train','_test'))
        valid=joined[np.isfinite(joined.odds_ratio_test)&np.isfinite(joined.corrected_p_value_test)]
        hits=valid[(valid.corrected_p_value_test<.05)&((valid.odds_ratio_train-1)*(valid.odds_ratio_test-1)>0)]
        result[arm]=dict(discovery_hits=len(discovery),evaluable=len(valid),reproduced=len(hits))
    return result



def displayed_association_rows(study,split,max_per_source=6):
    table=study['associations'];data=table[table.split.eq(split)]
    selected=[];ranking=study.get('ranking')
    for arm in data.arm.unique():
        candidates=data[data.arm.eq(arm)]
        if ranking is not None:
            order=ranking.loc[ranking.source.eq(arm)].sort_values('source_rank').column.tolist()
        else:
            training=table[table.arm.eq(arm)&table.split.eq('train')]
            order=training.sort_values(['p_value','marker'],na_position='last').marker.tolist() if len(training) else (
                [row['column'] for row in study['manifest']['markers']] if arm=='Literature' else study['manifest']['glyph_features'])
        lookup={name:i for i,name in enumerate(dict.fromkeys(order))}
        candidates=candidates.assign(_rank=candidates.marker.map(lookup).fillna(len(lookup)))
        selected.append(candidates.sort_values(['_rank','marker'],kind='stable').head(max_per_source).drop(columns='_rank'))
    return pd.concat(selected,ignore_index=True) if selected else data.reset_index(drop=True)


def plot_associations(ax,study,split):
    data=displayed_association_rows(study,split)
    labels=[]
    for i,row in enumerate(data.itertuples()):
        label=row.marker.replace('concept_','C')
        labels.append(('Lit · ' if row.arm=='Literature' else 'Glyph · ')+label)
        if not np.isfinite(row.odds_ratio) or row.odds_ratio<=0:
            ax.text(.03,i,'not estimable',transform=ax.get_yaxis_transform(),fontsize=7,color='#94a3b8')
            continue
        lo=max(.2,row.ci_lower);hi=min(1000,row.ci_upper);center=np.clip(row.odds_ratio,.2,1000)
        color=PALETTE[0] if row.arm=='Literature' else PALETTE[1]
        ax.plot([lo,hi],[i,i],color=color,lw=1.3)
        ax.plot(center,i,'o',color=color,ms=4.5)
        if row.ci_upper>1000:ax.plot(1000,i,'>',color=color,ms=5,clip_on=False)
        if row.ci_lower<.2:ax.plot(.2,i,'<',color=color,ms=5,clip_on=False)
    ax.set_xscale('log');ax.set_xlim(.2,1000);ax.axvline(1,color='#94a3b8',ls='--',lw=.7)
    ax.set_yticks(range(len(labels)),labels,fontsize=8);ax.invert_yaxis()
    ax.set_xlabel(f'{split} unadjusted OR per training SD · 95% CI',fontsize=8)
    counts=matched_reproduction(study,split)
    summary='; '.join(f"{arm} {value['reproduced']}/{value['evaluable']}" for arm,value in counts.items())
    ax.set_title('Matched reproduction · '+summary,loc='left',fontsize=9,fontweight='bold')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.12)


def activation_scale_examples(obj,frame,modality,index,outcome,split):
    values=obj.sae_features[modality][split][:,index].cpu().numpy()
    ids=np.asarray(obj.embeddings[split]['sample_ids']).astype(str)
    finite=np.flatnonzero(np.isfinite(values));positive=values[finite][values[finite]>1e-8]
    if not len(positive):return [],None
    targets=[0.,*np.quantile(positive,[.33,.66,1.]).tolist()];anchors=[];used=set()
    for rank,target in zip([0,33,66,100],targets):
        eligible=finite[values[finite]<=1e-8] if rank==0 else finite[values[finite]>1e-8]
        inactive=rank==0 and len(eligible)>0
        if not len(eligible):eligible=finite;target=float(values[finite].min())
        order=sorted(eligible,key=lambda i:(abs(float(values[i])-target),str(ids[i])))
        distance=abs(float(values[order[0]])-target)
        nearest=[i for i in order if abs(abs(float(values[i])-target)-distance)<1e-12]
        position=next((i for i in nearest if i not in used),nearest[0]);used.add(position)
        sample=ids[position];path=Path(str(frame.loc[sample,modality]))
        if path.suffix.lower() not in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp'}:continue
        if not path.is_file():raise FileNotFoundError(f'Activation-scale image is missing: {path}')
        labels=obj.outcomes.get(split,{}).get(outcome);value=float(labels[position]) if labels is not None else np.nan
        anchors.append(dict(sample_id=sample,split=split,activation=float(values[position]),
            target_percentile=rank,inactive=inactive,target_activation=target,
            outcome_value=value if value in [0,1] else None,source_file=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),_path=str(path)))
    maximum=float(positive.max());low=float(np.quantile(positive,.95));epsilon=max(1e-6,maximum*.001)
    counts,bins=np.histogram(values[finite],bins=np.linspace(0,max(maximum,epsilon),33))
    tail_counts,tail_bins=np.histogram(positive,bins=np.linspace(low if low<maximum else low-epsilon,maximum if low<maximum else maximum+epsilon,17))
    distribution=dict(n=len(finite),n_zero=int((values[finite]<=1e-8).sum()),n_active=len(positive),
        bins=bins.tolist(),counts=counts.tolist(),tail_bins=tail_bins.tolist(),tail_counts=tail_counts.tolist(),
        positive_p95=low,positive_max=maximum,anchor_rule='Zero activation, then 33rd, 66th and 100th percentiles among positive training activations; nearest distinct images; outcomes unused.')
    return anchors,distribution


def collect_image_examples(run, associations, zooms, split, comparison, count=2):
    from glyph import Glyph
    obj=Glyph.load_run(run,device='cpu')
    frame=pd.read_csv(obj.cfg.data.metadata,dtype={'sample_id':str}).set_index('sample_id')
    if not frame.index.is_unique:raise ValueError('Image metadata needs unique sample IDs')
    training=associations[associations.split==split].copy()
    keys=[name for name in ['odds_ratio_p_value_bonferroni','odds_ratio_p_value','concept_id'] if name in training]
    candidates=chain(zooms,(row.to_dict() for _,row in training.sort_values(keys).iterrows()))
    chosen=[];seen=set()
    for row in candidates:
        if row['concept_id'] in seen:continue
        modality,_,index=row['concept_id'].rsplit(':',2)
        if modality not in frame or modality not in obj.sae_features:continue
        index=int(index);samples=[]
        for partition in dict.fromkeys([split,comparison]):
            if partition not in obj.embeddings:continue
            values=obj.sae_features[modality][partition][:,index].cpu().numpy()
            ids=np.asarray(obj.embeddings[partition]['sample_ids']).astype(str)
            # Rank by activation only; outcomes never choose which images appear.
            active=np.flatnonzero(np.isfinite(values) & (values>1e-8))
            ranked=active[np.argsort(-values[active],kind='stable')]
            for position in ranked[:2]:
                sample=ids[position];path=Path(str(frame.loc[sample,modality]))
                if path.suffix.lower() not in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp'}:
                    continue
                if not path.is_file():raise FileNotFoundError(f'Image example is missing: {path}')
                labels=obj.outcomes.get(partition,{}).get(row['outcome'])
                value=float(labels[position]) if labels is not None else np.nan
                samples.append(dict(sample_id=sample,split=partition,activation=float(values[position]),
                    outcome_value=value if value in [0,1] else None,source_file=path.name,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),_path=str(path)))
        if not samples:continue
        item=dict(concept_id=row['concept_id'],name=row.get('name',f'{modality} concept {index}'),
            outcome=row.get('outcome_display',row['outcome']),outcome_key=row['outcome'],statistics=row,samples=samples)
        item['scale_samples'],item['distribution']=activation_scale_examples(obj,frame,modality,index,row['outcome'],split)
        if not item['statistics'].get('comparison'):
            other=associations[(associations.concept_id==row['concept_id']) &
                (associations.outcome==row['outcome']) & (associations.split==comparison)]
            item['statistics']['comparison']=other.iloc[0].to_dict() if len(other) else None
        chosen.append(item);seen.add(row['concept_id'])
        if len(chosen)==count:break
    if not chosen:raise ValueError('No supported local 2D images were found for the saved concepts')
    return chosen


def concept_example_figure(item,concepts,split,comparison,outcome_labels=None):
    from PIL import Image
    fig=plt.figure(figsize=(8.27,11.69),facecolor='white')
    fig.text(.065,.956,'GLYPH / CONCEPT EVIDENCE',fontsize=10,color=PALETTE[0],fontweight='bold')
    fig.text(.065,.919,textwrap.shorten(f"{item['name']} → {item['outcome']}",width=74),fontsize=14,fontweight='bold')
    row=item['statistics'];zoom=concepts.loc[concepts.concept_id.eq(item['concept_id'])].iloc[0].to_dict()
    if 'zoom_number' in row:zoom['zoom_number']=row['zoom_number']
    box=bounds(concepts,zoom)
    overview=fig.add_axes([.065,.755,.16,.12]);draw_map(overview,concepts,marker_scale=.05)
    for collection in overview.collections:collection.set_rasterized(True)
    overview.set_title('Whole concept space',fontsize=7.5,loc='left',color='#475569')
    overview.add_patch(Rectangle((box[0],box[2]),box[1]-box[0],box[3]-box[2],facecolor='#8b5cf6',alpha=.12,edgecolor='none'))
    overview.add_patch(Rectangle((box[0],box[2]),box[1]-box[0],box[3]-box[2],fill=False,edgecolor='#7c3aed',linewidth=.9,zorder=5))
    ax=fig.add_axes([.265,.755,.16,.12]);draw_map(ax,concepts,[zoom],box)
    ax.set_title('Selected region',fontsize=7.5,loc='left',color='#6d28d9')
    ax.add_patch(FancyBboxPatch((0,0),1,1,boxstyle='round,pad=.015,rounding_size=.02',transform=ax.transAxes,fill=False,edgecolor='#a78bfa',linewidth=.7,clip_on=False))
    for source,target in [((box[1],box[2]),(0,1)),((box[1],box[3]),(0,0))]:
        overview.add_artist(ConnectionPatch(xyA=source,coordsA=overview.transData,xyB=target,coordsB=ax.transAxes,color='#8b5cf6',linewidth=.75,alpha=.65,clip_on=False,zorder=.6))
    text=f"{split}: {effect(row)}; corrected p={pvalue(row.get('odds_ratio_p_value_bonferroni'))}"
    if row.get('comparison'):text+=f"\n{comparison}: {effect(row['comparison'])}; corrected p={pvalue(row['comparison'].get('odds_ratio_p_value_bonferroni'))}"
    else:text+=f'\n{comparison}: no evaluable saved association estimate'
    fig.text(.46,.855,'\n'.join(textwrap.fill(line,67) for line in text.splitlines()),fontsize=8,va='top',linespacing=1.5)
    interpretation=row.get('interpretation') or row.get('annotation')
    if interpretation:fig.text(.46,.783,textwrap.fill(textwrap.shorten(str(interpretation),width=140),64),fontsize=7,color='#475569',va='top')
    distribution=item.get('distribution');scale=item.get('scale_samples',[])
    full_ax=None;tail_ax=None;scale_colors=['#64748b','#0d9488','#3b82f6','#8b5cf6']
    if distribution:
        ax=fig.add_axes([.10,.645,.80,.077]);full_ax=ax;bins=np.array(distribution['bins']);counts=distribution['counts']
        ax.bar(bins[:-1],counts,width=np.diff(bins),align='edge',color='#94a3b8',alpha=.8)
        ax.set_yscale('symlog',linthresh=1);ax.set_ylabel('Images (log)',fontsize=7)
        ax.set_title(f"Activation distribution · {distribution['n_active']}/{distribution['n']} active · markers are active-sample percentiles",loc='left',fontsize=8,fontweight='bold')
        ax.axvspan(distribution['positive_p95'],distribution['positive_max'],facecolor='#a78bfa',alpha=.20,edgecolor='#7c3aed',linewidth=1.1)
        middle=(distribution['positive_p95']+distribution['positive_max'])/2
        ax.text(middle,.85,'TOP 5% OF ACTIVE' if distribution['positive_p95']<distribution['positive_max'] else 'ACTIVE VALUE',transform=ax.get_xaxis_transform(),ha='center',fontsize=6.5,fontweight='bold',color='#6d28d9',bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
        for sample,color in zip(scale,scale_colors):
            ax.axvline(sample['activation'],color=color,linewidth=.8,linestyle='--')
            ax.scatter([sample['activation']],[0],transform=ax.get_xaxis_transform(),s=22,color=color,edgecolors='white',linewidths=.5,zorder=5,clip_on=False)
        ax.tick_params(labelsize=6);ax.set_xlabel('Raw concept activation',fontsize=7,labelpad=1);ax.spines[['top','right']].set_visible(False)
    def show_sample(sample,i,y,title,color='#64748b'):
        ax=fig.add_axes([.065+i*.225,y,.205,.16])
        with Image.open(sample['_path']) as source:ax.imshow(source.convert('RGB'),interpolation='nearest')
        ax.axis('off')
        ax.add_patch(FancyBboxPatch((0,0),1,1,boxstyle='round,pad=.012,rounding_size=.025',transform=ax.transAxes,facecolor='none',edgecolor=color,linewidth=1.2,clip_on=False,zorder=4))
        ax.set_title(title,fontsize=7.3,color=color,fontweight='bold',bbox=dict(facecolor='white',edgecolor='none',pad=2))
        value=sample['outcome_value'];names=(outcome_labels or {}).get(item['outcome_key'],{})
        outcome=names.get(str(int(value)),f'outcome {int(value)}') if value is not None else 'unlabelled'
        ax.text(.5,-.07,f"{sample['sample_id']} · a={sample['activation']:.3g}\n{sample['split']} · {outcome}",transform=ax.transAxes,ha='center',va='top',fontsize=6.4,color='#475569')
        return ax
    def connect_example(histogram,card,sample,color):
        if histogram is None:return
        link=ConnectionPatch(xyA=(sample['activation'],0),coordsA=histogram.get_xaxis_transform(),xyB=(.5,1.015),coordsB=card.transAxes,connectionstyle='arc3,rad=.08',arrowstyle='-|>',mutation_scale=7,color=color,linewidth=.85,alpha=.8,clip_on=False,zorder=.6)
        histogram.add_artist(link)
        histogram.xaxis.label.set_bbox(dict(facecolor='white',edgecolor='none',pad=1.5))
    for i,sample in enumerate(scale):
        rank=sample['target_percentile'];title='Inactive (zero)' if sample['inactive'] else ('Minimum (no zeros)' if rank==0 else f"{rank}{'rd' if rank==33 else 'th'} active percentile")
        card=show_sample(sample,i,.425,title,scale_colors[i])
        connect_example(full_ax,card,sample,scale_colors[i])
    if distribution:
        ax=fig.add_axes([.10,.278,.80,.067]);tail_ax=ax;bins=np.array(distribution['tail_bins'])
        ax.bar(bins[:-1],distribution['tail_counts'],width=np.diff(bins),align='edge',color='#8b5cf6',alpha=.55)
        ax.axvspan(distribution['positive_p95'],distribution['positive_max'],color='#8b5cf6',alpha=.08)
        tail_samples=sorted(item['samples'],key=lambda sample:(sample['activation'],sample['split'],sample['sample_id']))
        tail_colors=['#a78bfa','#8b5cf6','#7c3aed','#5b21b6']
        for sample,color in zip(tail_samples,tail_colors):
            ax.axvline(sample['activation'],color=color,linewidth=.85,linestyle='--')
            ax.scatter([sample['activation']],[0],transform=ax.get_xaxis_transform(),s=22,color=color,edgecolors='white',linewidths=.5,zorder=5,clip_on=False)
        fig.text(.10,.365,'MAGNIFIED HIGH-ACTIVATION TAIL',fontsize=8,fontweight='bold',color='#6d28d9',bbox=dict(boxstyle='round,pad=.3',facecolor='#ede9fe',edgecolor='#c4b5fd',linewidth=.6))
        ax.set_title('Training-active 95th–100th percentiles shaded; strong train/test examples marked',loc='left',fontsize=6.5,color='#6d28d9')
        ax.set_xlabel('Raw concept activation · training histogram, train/test example markers',fontsize=7,labelpad=1)
        ax.set_ylabel('Images',fontsize=7);ax.tick_params(labelsize=6);ax.spines[['top','right']].set_visible(False)
    tail_samples=sorted(item['samples'],key=lambda sample:(sample['activation'],sample['split'],sample['sample_id']))
    for i,sample in enumerate(tail_samples):
        color=['#a78bfa','#8b5cf6','#7c3aed','#5b21b6'][i]
        card=show_sample(sample,i,.075,f"{sample['split']} · high activation",color)
        connect_example(tail_ax,card,sample,color)
    if full_ax is not None and tail_ax is not None:
        full_ax.figure.canvas.draw()
        start=fig.transFigure.inverted().transform(full_ax.get_xaxis_transform().transform((distribution['positive_max'],.80)))
        path=PlotPath([start,(.966,start[1]),(.966,.365),(.912,.345)],[PlotPath.MOVETO,PlotPath.LINETO,PlotPath.LINETO,PlotPath.LINETO])
        fig.add_artist(FancyArrowPatch(path=path,transform=fig.transFigure,arrowstyle='-|>',mutation_scale=9,linewidth=1.1,linestyle='--',color='#8b5cf6',alpha=.85))
        fig.text(.976,.49,'95–100% magnified',rotation=90,ha='center',va='center',fontsize=6.5,color='#7c3aed')
    note='Scale images selected on training activations only. Zero is shown separately; other percentiles exclude zeros. Strong examples are selected by activation within train/test, without outcome-based selection. Images do not validate a semantic interpretation.'
    fig.text(.065,.03,textwrap.fill(note,112),fontsize=7,color='#64748b')
    return fig


def build(concepts,associations,output,title,metadata,split='train',comparison='test',alpha=.05,zoom_count=2,preview_html=False,literature=None,image_run=None):
    if not 0<alpha<1:raise ValueError('alpha must be between zero and one')
    concepts,associations=validate(concepts,associations)
    metadata=dict(metadata)
    lookup={identifier:i for i,identifier in enumerate(concepts.concept_id)}
    supplied=metadata.get('correlation_edges',[])
    if any(e['source'] not in lookup or e['target'] not in lookup for e in supplied):raise ValueError('Correlation edge references an unknown concept')
    kept=limit_correlation_edges([[lookup[e['source']],lookup[e['target']],e['weight']] for e in supplied],len(concepts))
    identifiers=concepts.concept_id.tolist()
    metadata['correlation_edges']=[dict(source=identifiers[a],target=identifiers[b],weight=w) for a,b,w in kept]
    n=metadata.get('cohorts',{}).get(split,{}).get('n')
    if n:
        concepts['activation_rate']=pd.to_numeric(concepts.activation_rate,errors='coerce').fillna(pd.to_numeric(concepts.active_count,errors='coerce')/n)
    for identifier,note in metadata.get('concept_notes',{}).items():
        if identifier not in lookup:raise ValueError(f'Unknown concept note: {identifier}')
        concepts.loc[concepts.concept_id.eq(identifier),'annotation']=str(note)
    concepts,facts,colors,display=prepare(concepts,associations,split,comparison,alpha,zoom_count,metadata.get('focus_concepts'),metadata.get('focus_outcome'))
    if metadata.get('focus_concepts'):facts['zoom_selection']=metadata.get('focus_selection_note','User-selected concepts/outcome; saved statistics are unchanged.')
    concepts.attrs['correlation_edges']=metadata.get('correlation_edges',[])
    facts['title']=title;facts['metadata']=metadata
    examples=collect_image_examples(image_run,associations,facts['zooms'],split,comparison) if image_run else []
    if examples:facts['image_examples']=examples
    study=read_study(literature)
    l_count=len(study['manifest']['markers']) if study else 0
    g_count=len(study['manifest']['glyph_features']) if study else 0
    matched_count=min(l_count,g_count)
    discovery_family=', '.join(map(str,metadata.get('testing_families',{}).get(split,[]))) or 'the saved family'
    if study:
        facts['literature']=dict(manifest=study['manifest'],auc_intervals=study['intervals'].to_dict('records'),
            auc_comparisons=study['comparisons'].to_dict('records'),metrics=study['metrics'].to_dict('records'),association_reproduction=matched_reproduction(study,comparison),displayed_associations=displayed_association_rows(study,comparison)[['arm','marker']].to_dict('records'),displayed_comparisons=displayed_auc_comparisons(study,comparison),search=study.get('search'),bootstrap=study.get('bootstrap'),adjustment=study.get('adjustment'))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    concepts.drop(columns='_color').to_csv(output/'concept_space.csv',index=False)
    pd.DataFrame(facts['zooms']).drop(columns='comparison',errors='ignore').to_csv(output/'zoom_evidence.csv',index=False)
    if study and study.get('adjusted') is not None:study['adjusted'].to_csv(output/'adjusted_associations.csv',index=False)
    (output/'facts.json').write_text(json.dumps(clean(facts),indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'text.color':'#0f172a','axes.labelcolor':'#0f172a'})
    zooms=facts['zooms'];quality=metric_lines(metadata)
    handles=[plt.Line2D([],[],marker='o',linestyle='',color=color,label=textwrap.shorten(display.get(group,group),width=25),markersize=7) for group,color in colors.items()]
    handles.append(plt.Line2D([],[],marker='o',linestyle='',color=GRAY,label='No significant estimate',markersize=7))
    header=f"{facts['n_significant_pairs']} significant associations · {facts['n_concepts']} concepts mapped"
    coverage=f"{facts['n_reproduced_hits']}/{facts['n_comparable_hits']} evaluable hits reproduce in {comparison}; {facts['n_significant_pairs']-facts['n_comparable_hits']} discovery hits lack evaluable comparison estimates."
    caveat=str(metadata.get('notes','Saved unadjusted associations. Interpretation evidence is shown only when available.'))
    color_note=('Color: strongest correlated training measurement, for significant nodes only.' if concepts.color_group.ne('').any() else 'Color: strongest significant outcome.')
    scope=f"{metadata.get('scope','Entire supplied concept space')}. {color_note} Node area is proportional to activation prevalence, with a tiny visibility floor and a common density adjustment. Missing prevalences use a uniform small marker. Lines show saved correlations, with at most eight per concept."
    map_fig,map_ax=plt.subplots(figsize=(8.5,6.6));draw_map(map_ax,concepts,zooms)
    map_legend(map_fig,map_ax,colors,display,handles)
    map_ax.set_title(f"Full concept space · {len(concepts)} nodes",loc='left',fontweight='bold',fontsize=12)
    save_figure(map_fig,output/'concept_space')
    for i,row in enumerate(zooms):
        fig,ax=plt.subplots(figsize=(2.8,1.65));draw_map(ax,concepts,[row],bounds(concepts,row))
        ax.set_title(textwrap.shorten(row['name'],width=36),loc='left',fontsize=9,fontweight='bold')
        save_figure(fig,output/f'zoom_{i+1}');plt.close(fig)
    if study:
        fig,ax=plt.subplots(figsize=(7.4,max(2.8,len(prediction_rows(study,comparison))*.48)));plot_auc(ax,study,comparison);fig.tight_layout();save_figure(fig,output/'literature_auc');plt.close(fig)
        fig,ax=plt.subplots(figsize=(5.3,2.8));plot_associations(ax,study,comparison);fig.tight_layout();save_figure(fig,output/'literature_associations');plt.close(fig)
    primary=None
    if study:
        visible_arms=set(prediction_rows(study,comparison).arm)
        selected=study['comparisons'].query('split == @comparison and primary == True')
        selected=selected[selected.comparison.isin(visible_arms)&selected.reference.isin(visible_arms)]
        if len(selected):primary=selected.iloc[0]
    evidence_line=(f"Full combination versus literature: ΔAUC {primary.delta_auc:+.4f} (95% CI {primary.ci_lower:+.4f} to {primary.ci_upper:+.4f}), paired p={primary.p_value:.3g}." if primary is not None else '')
    if study:
        capped=study['comparisons'][(study['comparisons'].split==comparison)&(study['comparisons'].reference=='Literature')&study['comparisons'].comparison.str.startswith('Literature + Glyph to ')]
        if len(capped):
            row=capped.iloc[0]
            evidence_line+=f' Capped combination: ΔAUC {row.delta_auc:+.4f} (95% CI {row.ci_lower:+.4f} to {row.ci_upper:+.4f}), paired p={row.p_value:.3g} (exploratory, unadjusted).'
    interpretation_note=metadata.get('interpretation_status','Interpretation evidence is shown when supplied.')
    lines=[f'# {title}','',header+'.',coverage,'']+[line+'.' for line in quality]
    lines += ['',f'**Concept space.** {scope} Both OR directions; Bonferroni-corrected p < {alpha:g}.','','![Entire concept space](concept_space.png)','']
    example_images={item['concept_id']:f'concept_{i+1}.png' for i,item in enumerate(examples)}
    for i,row in enumerate(zooms):
        lines += [f"**{row['name']} → {row['outcome_display']}.** {split}: {effect(row)}; corrected p={pvalue(row.get('odds_ratio_p_value_bonferroni'))}."]
        if row['comparison']:lines += [f"{comparison}: {effect(row['comparison'])}; corrected p={pvalue(row['comparison'].get('odds_ratio_p_value_bonferroni'))}."]
        if row['annotation']:lines += [row['annotation']]
        if row['interpretation']:
            accuracy=row['interpretation_accuracy'];lines += [f"Suggested interpretation: {row['interpretation']}"+(f" {row['interpretation_accuracy_context']} accuracy {accuracy:.0%}." if accuracy is not None else '')]
        if row['concept_id'] in example_images:lines += ['',f"![Zoom, statistics and activating images]({example_images[row['concept_id']]})"]
        else:lines += ['',f'![Concept zoom](zoom_{i+1}.png)']
        lines += ['']
    represented={row['concept_id'] for row in zooms}
    for i,item in enumerate(examples):
        if item['concept_id'] not in represented:
            lines += [f"**Exploratory concept: {item['name']}.** Selected using training associations; significance is reported on the page.",'',f'![Activation scale and context](concept_{i+1}.png)','']
    if study:
        lines += ['**Literature baseline comparison.** '+evidence_line,'',capped_model_description(study['manifest']),'','![AUC comparison](literature_auc.png)','']
        for row in prediction_rows(study,comparison).itertuples():lines += [f'- {row.arm}: AUC {row.auc:.3f} (95% CI {row.ci_lower:.3f}–{row.ci_upper:.3f}).']
        for pair in displayed_auc_comparisons(study,comparison):lines += [f"- {pair['reference']} vs {pair['comparison']}: {significance_label(pair['p_value'])}, paired p={pvalue(pair['p_value'])} (exploratory, unadjusted)."]
        lines += ['',f"Literature markers: {', '.join(row['column'] for row in study['manifest']['markers'])}. Prediction uses {l_count} literature markers, {g_count} Glyph concepts and their combination. The forest panel shows up to six training-ranked markers per source; all matched estimates and reproduction counts are retained. Individual associations use {matched_count} versus {matched_count}, with a separate {2*matched_count}-test family at corrected p<0.05. The full discovery analysis retains {discovery_family} tests; these reproduction denominators differ.",'']
        sources={}
        for marker in study['manifest']['markers']:sources.setdefault(marker['source_url'],marker.get('citation',marker.get('doi',marker.get('id',marker['column']))))
        lines += ['Sources: '+', '.join(f'[{label}]({url})' for url,label in sources.items())+'.']
        if study.get('search'):
            funnel=study['search'];lines += [f"Literature search: {funnel.get('combined_unique',0)} unique retrieved papers; {funnel.get('titles_screened',0)} titles screened; {funnel.get('selected_unique_papers',0)} selected papers yielding {funnel.get('markers',0)} markers. See the saved query log for search limits and coverage."]
        if study.get('adjustment'):
            setting=study['adjustment'];method={'fmr_orthogonal':'FMR orthogonal','fmr_sigmoid':'FMR sigmoid (legacy)'}.get(setting.get('method'),'FMR');lines += [f"Separate confounder adjustment: {method}, tree depth {setting['max_depth']}, device {setting['device']}. These estimates are separate from unadjusted associations and predictive AUCs. [Adjusted results](adjusted_associations.csv)."]
        lines += [' '.join(study['manifest'].get('caveats',[])),f"AUC intervals condition on frozen models and use {next((row['replicates'] for row in study.get('bootstrap',[]) if row['split']==comparison),'the saved number of')} paired subject-bootstrap resamples; a perfect empirical ranking yields a degenerate interval.",'']
    lines += [interpretation_note,'',f'*{caveat}*']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    with PdfPages(output/'report.pdf') as pdf:
        fig=plt.figure(figsize=(8.27,11.69),facecolor='white')
        fig.text(.065,.956,'GLYPH / DISCOVERY BRIEF',fontsize=10,color=PALETTE[0],fontweight='bold')
        fig.text(.065,.919,textwrap.shorten(title,width=56),fontsize=19,fontweight='bold')
        fig.text(.065,.882,header,fontsize=12,fontweight='bold')
        fig.text(.065,.854,textwrap.fill(coverage,105),fontsize=9,color='#475569')
        fig.text(.065,.796,'\n'.join(quality[:3]),fontsize=9,color='#334155',linespacing=1.6)
        ax=fig.add_axes([.045,.25,.91,.49]);draw_map(ax,concepts,zooms)
        map_legend(fig,ax,colors,display,handles)
        fig.text(.065,.134,textwrap.fill(scope,107),fontsize=8,color='#475569')
        fig.text(.065,.069,textwrap.fill(caveat+' '+metadata.get('interpretation_status',''),110),fontsize=8,color='#64748b')
        pdf.savefig(fig);plt.close(fig)
        if study:
            fig=plt.figure(figsize=(8.27,11.69),facecolor='white')
            fig.text(.065,.95,'PREDICTION / BASELINE COMPARISON',fontsize=15,fontweight='bold')
            ax=fig.add_axes([.35,.535,.38,.30]);plot_auc(ax,study,comparison)
            fig.text(.065,.465,textwrap.fill(evidence_line,103),fontsize=9,color='#0f172a')
            fig.text(.065,.405,textwrap.fill('Feature rankings use training only; validation selects ridge penalties and the best optimizer checkpoint. Combined models penalize literature and Glyph separately. Parentheses give predictor counts. '+capped_model_description(study['manifest']),105),fontsize=8,color='#475569')
            ax=fig.add_axes([.25,.155,.68,.19]);plot_associations(ax,study,comparison)
            note=f'{l_count} available literature markers; {g_count} available Glyph concepts. Individual associations use {matched_count} per source ({2*matched_count}-test correction), separately from the full discovery family of {discovery_family}. Forest: up to six training-ranked markers per source; counts use the full matched set. Brackets show paired-bootstrap p-values, unadjusted (*<.05, **<.01, ***<.001; n.s. otherwise). Bootstrap intervals condition on fitted models. These results do not establish novelty or external validation.'
            fig.text(.065,.069,textwrap.fill(note,112),fontsize=7.5,color='#64748b')
            source=study['manifest']['markers'][0]
            fig.text(.065,.027,'Literature source: '+textwrap.shorten(source.get('citation',source.get('doi',source['source_url'])),width=105),fontsize=7.2,color='#64748b')
            pdf.savefig(fig);plt.close(fig)
        for i,item in enumerate(examples):
            fig=concept_example_figure(item,concepts,split,comparison,metadata.get('outcome_labels'))
            fig.savefig(output/f'concept_{i+1}.png',dpi=150)
            from PIL import Image
            with Image.open(output/f'concept_{i+1}.png') as page:
                page.crop((0,0,page.width,int(page.height*.25))).save(output/f'context_header_{i+1}.png')
                page.crop((0,int(page.height*.25),page.width,int(page.height*.615))).save(output/f'activation_scale_{i+1}.png')
                page.crop((0,int(page.height*.615),page.width,int(page.height*.96))).save(output/f'activation_tail_{i+1}.png')
            pdf.savefig(fig);plt.close(fig)
        # Preserve zoom evidence even when raw input images are unavailable.
        shown={item['concept_id'] for item in examples}
        for i,row in enumerate(zooms):
            if row['concept_id'] in shown:continue
            fig=plt.figure(figsize=(8.27,11.69),facecolor='white')
            fig.text(.065,.95,textwrap.shorten(row['name'],width=65),fontsize=15,fontweight='bold')
            ax=fig.add_axes([.1,.45,.8,.42]);draw_map(ax,concepts,[row],bounds(concepts,row))
            text=f"{split}: {effect(row)}; corrected p={pvalue(row.get('odds_ratio_p_value_bonferroni'))}"
            if row['comparison']:text+=f"\n{comparison}: {effect(row['comparison'])}; corrected p={pvalue(row['comparison'].get('odds_ratio_p_value_bonferroni'))}"
            fig.text(.065,.32,textwrap.fill(text,100),fontsize=10)
            if row['interpretation']:fig.text(.065,.22,textwrap.fill(row['interpretation'],100),fontsize=9)
            pdf.savefig(fig);plt.close(fig)
        if examples:
            from PIL import Image
            pages=[Image.open(output/f'concept_{i+1}.png').convert('RGB') for i in range(len(examples))]
            preview=Image.new('RGB',(max(page.width for page in pages),sum(page.height for page in pages)),'white')
            offset=0
            for page in pages:preview.paste(page,(0,offset));offset+=page.height;page.close()
            preview.save(output/'image_examples.png')
    plt.close(map_fig)
    if preview_html:
        image=lambda name:'data:image/png;base64,'+base64.b64encode((output/name).read_bytes()).decode()
        blocks=f'<img src="{image("concept_space.png")}" alt="Full concept space">'
        for i,row in enumerate(zooms):
            if row['concept_id'] in example_images:
                index=int(example_images[row['concept_id']].split('_')[-1].split('.')[0])
                blocks+=f'<img src="{image(f"context_header_{index}.png")}" alt="Whole graph, selected region and concept statistics"><img src="{image(f"activation_scale_{index}.png")}" alt="Activation scale with percentile examples"><details><summary>Expand high-activation detail (95th–100th percentile)</summary><img src="{image(f"activation_tail_{index}.png")}" alt="High-activation examples"></details>';continue
            blocks+=f'<section><img src="{image(f"zoom_{i+1}.png")}"><div><h2>{html.escape(row["name"])}</h2><p>{html.escape(row["annotation"])}</p><p>{html.escape(effect(row))}</p>'
            if row['comparison']:blocks+=f'<p>Test: {html.escape(effect(row["comparison"]))}</p>'
            if row['interpretation']:blocks+=f'<p>{html.escape(row["interpretation"])}</p>'
            blocks+='</div></section>'
        represented={row['concept_id'] for row in zooms}
        for index,item in enumerate(examples,1):
            if item['concept_id'] in represented:continue
            blocks+=f'<p>Exploratory concept selected from training associations; see its reported significance.</p><img src="{image(f"context_header_{index}.png")}"><img src="{image(f"activation_scale_{index}.png")}"><details><summary>Expand high-activation detail (95th–100th percentile)</summary><img src="{image(f"activation_tail_{index}.png")}"></details>'
        if study:blocks+=f'<h2>Literature baseline comparison</h2><p>{html.escape(evidence_line)}</p><img src="{image("literature_auc.png")}"><img src="{image("literature_associations.png")}">'
        content=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>body{{font:15px/1.6 system-ui;color:#0f172a;max-width:1000px;margin:40px auto;padding:0 24px}}h1{{font-size:30px}}img{{max-width:100%;height:auto}}section{{display:grid;grid-template-columns:230px 1fr;gap:20px;border-top:1px solid #e2e8f0;padding:20px 0}}footer{{color:#64748b;font-size:12px}}@media(max-width:650px){{section{{display:block}}}}</style></head><body><h1>{html.escape(title)}</h1><b>{html.escape(header)}</b><p>{html.escape(coverage)}</p><p>{'<br>'.join(html.escape(line) for line in quality)}</p>{blocks}<footer>{html.escape(interpretation_note)} {html.escape(caveat)}</footer></body></html>'''
        (output/'report.html').write_text(content)
    return facts


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--run',type=Path)
    inputs.add_argument('--concepts',type=Path)
    parser.add_argument('--associations',type=Path)
    parser.add_argument('--metadata',type=Path,help='Optional JSON metadata, including focus_concepts, focus_outcome and concept_notes for selected illustrations')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--title',default='Glyph discovery brief')
    parser.add_argument('--split',default='train')
    parser.add_argument('--comparison-split',default='test')
    parser.add_argument('--alpha',type=float,default=.05)
    parser.add_argument('--zoom-count',type=int,choices=[0,1,2],default=2)
    parser.add_argument('--preview-html',action='store_true')
    parser.add_argument('--annotations',type=Path)
    parser.add_argument('--literature',type=Path)
    parser.add_argument('--interpretations',type=Path,help='Trusted local saved interpretation results')
    parser.add_argument('--image-examples',action='store_true',help='Add actual activating training/comparison images from the saved run')
    args=parser.parse_args(argv)
    if args.image_examples and not args.run:parser.error('--image-examples requires --run')
    if args.run:
        concepts,associations,metadata=run_tables(args.run,args.interpretations)
    else:
        if not args.associations:parser.error('--associations is required with --concepts')
        concepts,associations=read_table(args.concepts),read_table(args.associations)
        metadata={}
    if args.metadata:metadata.update(json.loads(args.metadata.read_text()))
    if args.annotations:concepts=annotate(concepts,read_table(args.annotations))
    inputs = list(args.run.glob('*')) if args.run else [args.concepts, args.associations]
    if args.metadata: inputs.append(args.metadata)
    if args.annotations: inputs.append(args.annotations)
    if args.interpretations: inputs.append(args.interpretations)
    if args.literature: inputs.extend(args.literature.glob('*.csv'));inputs.append(args.literature/'manifest.json')
    metadata['provenance'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(inputs) if path.is_file()}
    metadata['report_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    facts=build(concepts,associations,args.output,args.title,metadata,args.split,args.comparison_split,args.alpha,args.zoom_count,args.preview_html,args.literature,args.run if args.image_examples else None)
    print(json.dumps({k:v for k,v in facts.items() if k not in ['zooms','metadata','literature','image_examples']},indent=2))


if __name__=='__main__':main()
