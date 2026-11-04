"""Frozen-marker literature comparison: train-fit, validation-select, paired test evaluation.

Numerical ridge fitting adapted from the research axis_risk_models.py engine.
No research checkout, live search or LLM credentials are required at runtime.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

DEVICE = 'cpu'
ALPHAS = np.array([.00001,.0001,.001,.01,.1,1.,10.,100.])

def encode(x,tr):
    mu=np.nanmean(x[tr],axis=0);mu=np.nan_to_num(mu)
    missing=~np.isfinite(x);z=np.where(missing,mu,x);sd=z[tr].std(axis=0);sd=np.where(sd>1e-8,sd,1)
    # Missingness is a separate predictor, including for a missing binary rule.
    return np.column_stack([(z-mu)/sd,missing.astype(float)]).astype('float64'),mu,sd

def fit_batch_wide(x,y,alphas):
    """The same convex ridge objective without a quadratic feature Hessian."""
    from scipy.optimize import minimize
    from scipy.special import expit
    keep=np.any(x!=0,axis=0);reduced=x[:,keep]
    design=np.column_stack([np.ones(len(x)),reduced]);penalty=np.ones(design.shape[1]);penalty[0]=0
    coefficients=np.zeros((design.shape[1],len(alphas)));current=np.zeros(design.shape[1])
    current[0]=np.log(y.mean()/(1-y.mean()));maximum=0.
    for j in np.argsort(alphas)[::-1]:
        alpha=float(alphas[j])
        def objective(beta):
            eta=design@beta;prob=expit(eta)
            loss=np.mean(np.logaddexp(0,eta)-y*eta)+.5*alpha*np.dot(penalty,beta*beta)
            gradient=design.T@(prob-y)/len(y)+alpha*penalty*beta
            return loss,gradient
        result=minimize(objective,current,jac=True,method='L-BFGS-B',
            options=dict(maxiter=5000,gtol=1e-9,ftol=1e-15,maxls=40))
        current=result.x;error=float(np.abs(objective(current)[1]).max());maximum=max(maximum,error)
        if error>1e-7:raise RuntimeError(f'Wide ridge fit did not converge: maximum gradient {error}')
        coefficients[:,j]=current
    expanded=np.zeros((x.shape[1]+1,len(alphas)));expanded[0]=coefficients[0]
    expanded[1:][keep]=coefficients[1:]
    return expanded,predict(x,expanded),maximum


def fit_batch(x,y,alphas):
    if x.shape[1]>512:return fit_batch_wide(x,y,alphas)
    d=torch.device(DEVICE);X=torch.tensor(np.column_stack([np.ones(len(x)),x]),dtype=torch.float64,device=d)
    Y=torch.tensor(y[:,None],dtype=torch.float64,device=d);A=torch.tensor(alphas,dtype=torch.float64,device=d)
    B=torch.zeros((X.shape[1],len(alphas)),dtype=torch.float64,device=d);B[0]=np.log(y.mean()/(1-y.mean()))
    ridge=torch.ones_like(B)*A;ridge[0]=0
    def loss(b):
        eta=X@b
        return (torch.nn.functional.softplus(eta)-Y*eta).mean(0)+.5*(ridge*b*b).sum(0)
    # Independent Newton steps and line searches for every grid candidate.
    # A shared LBFGS line search can leave weak penalties unconverged when
    # the same batch also contains very strong penalties.
    for iteration in range(100):
        prob=torch.sigmoid(X@B);gradient=X.T@(prob-Y)/len(y)+ridge*B
        if float(gradient.abs().max())<1e-9:break
        weights=prob*(1-prob)
        hessian=torch.einsum('ni,nk,nj->kij',X,weights,X)/len(y)+torch.diag_embed(ridge.T)
        hessian+=torch.eye(X.shape[1],dtype=torch.float64,device=d)[None,:,:]*1e-12
        step=torch.linalg.solve(hessian,gradient.T.unsqueeze(-1)).squeeze(-1).T
        fraction=torch.clamp(2/step.abs().amax(0).clamp_min(1e-12),max=1.)
        oldloss=loss(B)
        for backtrack in range(30):
            candidate=B-step*fraction;newloss=loss(candidate)
            acceptable=newloss<=oldloss+1e-13
            if bool(acceptable.all()):break
            fraction=torch.where(acceptable,fraction,fraction*.5)
        B=candidate
    prob=torch.sigmoid(X@B);gradient=X.T@(prob-Y)/len(y)+ridge*B;error=float(gradient.abs().max())
    if error>1e-7:raise RuntimeError(f'Ridge grid did not converge: maximum gradient {error}')
    return B.cpu().numpy(),prob.cpu().numpy(),error

def predict(x,b):
    from scipy.special import expit
    return expit(np.column_stack([np.ones(len(x)),x])@b)

def validate_inputs(cohort, glyph, markers, manifest):
    required={'sample_id','split','y'}
    if required-set(cohort):raise ValueError('Cohort needs sample_id, split and y')
    if not set(cohort.y.unique()).issubset({0,1}):raise ValueError('Use outcome-eligible binary observations only')
    if not {'train','val'}.issubset(set(cohort.split)):raise ValueError('Separate train and val splits are required')
    if not set(cohort.split)-{'train','val'}:raise ValueError('Supply a held-out split')
    for table in [cohort,glyph,markers]:
        if table.sample_id.isna().any() or table.sample_id.duplicated().any():raise ValueError('IDs must be unique and nonmissing')
    if 'patient_id' in cohort and (cohort.patient_id.isna().any() or cohort.groupby('patient_id').split.nunique().max()>1):
        raise ValueError('Patients must not cross fitting, tuning and evaluation splits')
    for name,table in [('glyph',glyph),('markers',markers)]:
        if set(table.sample_id)!=set(cohort.sample_id):raise ValueError(f'{name} IDs must match the common cohort exactly')
    definitions=manifest.get('markers',[])
    if not definitions or any(not row.get('source_url') or not row.get('definition') for row in definitions):
        raise ValueError('Every literature marker needs a definition and supporting source URL')
    expected=[row['column'] for row in definitions]
    if len(set(expected))!=len(expected) or set(markers)-{'sample_id'}!=set(expected):raise ValueError('Marker table must match the frozen manifest')
    chosen=manifest.get('glyph_features',[])
    if not chosen or len(chosen)!=len(set(chosen)) or not set(chosen).issubset(glyph):raise ValueError('Supply unique frozen Glyph selections')
    if not manifest.get('glyph_selection_rule'):raise ValueError('Record how Glyph concepts were selected without held-out labels')
    return expected,chosen


def training_rankings(cohort,glyph,markers,manifest):
    """Two-sided univariate logistic score tests, using training rows only."""
    literature,chosen=validate_inputs(cohort,glyph,markers,manifest)
    train=cohort.split.eq('train').to_numpy();y=cohort.y.to_numpy(float)[train];rows=[]
    for source,table,columns in [('Literature',markers,literature),('Glyph',glyph,chosen)]:
        values=table.set_index('sample_id').loc[cohort.sample_id,columns].to_numpy(float)[train]
        known=np.isfinite(values);n=known.sum(axis=0);clean=np.where(known,values,0.)
        mean=np.divide(clean.sum(axis=0),n,out=np.zeros(len(columns)),where=n>0)
        centered=np.where(known,values-mean,0.)
        prevalence=np.divide((known*y[:,None]).sum(axis=0),n,out=np.zeros(len(columns)),where=n>0)
        score=(centered*(y[:,None]-prevalence)).sum(axis=0)
        information=prevalence*(1-prevalence)*(centered**2).sum(axis=0)
        z=np.divide(score,np.sqrt(information),out=np.zeros(len(columns)),where=information>1e-12)
        pvalues=2*norm.sf(np.abs(z));pvalues[information<=1e-12]=1.
        rows.extend(dict(source=source,column=column,training_p_value=float(pvalue)) for column,pvalue in zip(columns,pvalues))
    ranking=pd.DataFrame(rows).sort_values(['training_p_value','source','column'],kind='stable').reset_index(drop=True)
    ranking['pooled_rank']=np.arange(1,len(ranking)+1)
    ranking['source_rank']=ranking.groupby('source').cumcount()+1
    return ranking


def comparison_feature_sets(cohort,glyph,markers,manifest):
    literature,chosen=validate_inputs(cohort,glyph,markers,manifest)
    ranking=training_rankings(cohort,glyph,markers,manifest)
    cap=manifest.get('comparison_top_n',10)
    if not isinstance(cap,int) or isinstance(cap,bool) or cap<1:raise ValueError('comparison_top_n must be a positive integer')
    l=[('Literature',c) for c in literature];g=[('Glyph',c) for c in chosen]
    top=lambda source:[(r.source,r.column) for r in ranking.itertuples() if source is None or r.source==source][:cap]
    strategy=manifest.get('combined_top_n_strategy','retain_literature')
    if strategy not in ['retain_literature','pooled']:raise ValueError('combined_top_n_strategy must be retain_literature or pooled')
    combined_top=top(None) if strategy=='pooled' else top('Literature')+top('Glyph')[:max(0,cap-len(l))]
    return {'Literature':l,f'Literature top {cap}':top('Literature'),
            'Glyph':g,f'Glyph top {cap}':top('Glyph'),
            'Literature + Glyph':l+g,f'Literature + Glyph to {cap}':combined_top},ranking


def fit_validated_grid(x,y,x_val,y_val,penalty_grid,initial=None,patience=20,min_delta=1e-6):
    """Fit on train, stop on validation AUROC, and restore its best checkpoint.

    Validation log loss breaks AUROC ties. Penalty rescaling keeps independently
    penalized literature/Glyph blocks numerically well conditioned.
    """
    from scipy.optimize import minimize
    from scipy.special import expit
    if patience<1 or min_delta<0:raise ValueError('Invalid validation stopping settings')
    penalties=np.asarray(penalty_grid,dtype=float)
    if penalties.ndim!=2 or penalties.shape[1]!=x.shape[1] or not np.isfinite(penalties).all() or (penalties<=0).any():
        raise ValueError('Supply positive finite penalties for every encoded predictor')
    if len(np.unique(y_val))!=2:raise ValueError('Validation needs both classes')
    keep=np.any(x!=0,axis=0);all_coefficients=[];diagnostics=[]
    class ValidationStop(Exception):pass
    for penalty in penalties:
        scale=np.sqrt(penalty[keep]);design=np.column_stack([np.ones(len(x)),x[:,keep]/scale])
        validation=np.column_stack([np.ones(len(x_val)),x_val[:,keep]/scale])
        start=np.zeros(design.shape[1]);start[0]=np.log(y.mean()/(1-y.mean()))
        if initial is not None:start[0]=initial[0];start[1:]=np.asarray(initial)[1:][keep]*scale
        best=start.copy();best_auc=-np.inf;best_loss=np.inf;best_iteration=0;wait=0;iteration=0;trace=[];last=start.copy();stopped=False
        def objective(theta):
            eta=design@theta;prob=expit(eta)
            gradient=design.T@(prob-y)/len(y);gradient[1:]+=theta[1:]
            return np.mean(np.logaddexp(0,eta)-y*eta)+.5*np.dot(theta[1:],theta[1:]),gradient
        def observe(theta):
            nonlocal best,best_auc,best_loss,best_iteration,wait,last
            last=theta.copy();prob=expit(validation@theta)
            auc=float(roc_auc_score(y_val,prob));loss=float(log_loss(y_val,prob,labels=[0,1]))
            trace.append(dict(iteration=iteration,validation_auc=auc,validation_log_loss=loss))
            better=auc>best_auc+min_delta or (abs(auc-best_auc)<1e-12 and loss<best_loss-min_delta)
            if better:best=theta.copy();best_auc=auc;best_loss=loss;best_iteration=iteration;wait=0
            else:wait+=1
        observe(start)
        def callback(theta):
            nonlocal iteration
            iteration+=1;observe(theta)
            if wait>=patience:raise ValidationStop()
        try:
            result=minimize(objective,start,jac=True,method='L-BFGS-B',callback=callback,
                options=dict(maxiter=5000,gtol=1e-9,ftol=1e-15,maxls=40))
            status='optimizer_converged' if result.success else 'optimizer_limit'
        except ValidationStop:stopped=True;status='validation_patience'
        coefficient=np.zeros(x.shape[1]+1);coefficient[0]=best[0];coefficient[1:][keep]=best[1:]/scale
        all_coefficients.append(coefficient)
        diagnostics.append(dict(validation_auc=best_auc,validation_log_loss=best_loss,best_iteration=best_iteration,
            optimizer_iterations=iteration,stopped_early=stopped,solver_status=status,
            gradient_max=float(np.abs(objective(best)[1]).max()),validation_trace=json.dumps(trace)))
    return np.column_stack(all_coefficients),diagnostics


def fit_models(cohort, glyph, markers, manifest, alphas=ALPHAS):
    arms,ranking=comparison_feature_sets(cohort,glyph,markers,manifest)
    cohort=cohort.reset_index(drop=True)
    tables={source:table.set_index('sample_id').loc[cohort.sample_id] for source,table in [('Literature',markers),('Glyph',glyph)]}
    train=cohort.split.eq('train').to_numpy();val=cohort.split.eq('val').to_numpy();y=cohort.y.to_numpy(float)
    if any(len(np.unique(y[m]))!=2 for m in [train,val]):raise ValueError('Training and validation need both classes')
    settings=manifest.get('prediction_early_stopping',{})
    if set(settings)-{'patience','min_delta'}:raise ValueError('Unsupported prediction early stopping setting')
    patience=int(settings.get('patience',20));min_delta=float(settings.get('min_delta',1e-6))
    predictions=cohort[['sample_id','split','y']].copy();tuning=[];models={};metrics=[]
    for arm,features in arms.items():
        features=sorted(features,key=lambda item:({'Literature':0,'Glyph':1}[item[0]],item[1]))
        values=np.column_stack([tables[source][column].to_numpy(float) for source,column in features])
        x,mean,sd=encode(values,np.flatnonzero(train));combined=arm.startswith('Literature + Glyph')
        grid=np.asarray(alphas,dtype=float);initial=None;literature_alpha=None
        if combined:
            literature_alpha=float(models['Literature']['alpha'])
            grid=np.unique(np.r_[grid,1e3,1e4,1e5])
            is_glyph=np.tile([source=='Glyph' for source,column in features],2)
            penalty_grid=np.array([np.where(is_glyph,a,literature_alpha) for a in grid])
            baseline=models['Literature']
            baseline_lookup={name:i for i,name in enumerate(baseline['feature_names'])};n_baseline=len(baseline_lookup)
            # Retain the exact baseline checkpoint only when every baseline
            # predictor is present. A pooled cap may legitimately omit some.
            if set(baseline_lookup).issubset({c for source,c in features if source=='Literature'}):
                initial=np.zeros(x.shape[1]+1);initial[0]=baseline['coefficients'][0,0]
                for j,(source,column) in enumerate(features):
                    if source=='Literature':
                        k=baseline_lookup[column];initial[1+j]=baseline['coefficients'][1+k,0]
                        initial[1+len(features)+j]=baseline['coefficients'][1+n_baseline+k,0]
        else:penalty_grid=np.array([np.full(x.shape[1],a) for a in grid])
        coefficients,diagnostics=fit_validated_grid(x[train],y[train],x[val],y[val],penalty_grid,initial,patience,min_delta)
        scores=np.array([item['validation_auc'] for item in diagnostics]);best=np.flatnonzero(scores==scores.max())[-1]
        selected=diagnostics[best];prob=predict(x,coefficients[:,[best]]).ravel();predictions[arm]=prob
        model=dict(coefficients=coefficients[:,[best]],mean=mean,sd=sd,alpha=grid[best],
            feature_names=np.array([column for source,column in features]),feature_sources=np.array([source for source,column in features]),
            best_iteration=selected['best_iteration'],optimizer_iterations=selected['optimizer_iterations'],
            stopped_early=selected['stopped_early'],early_stopping_patience=patience,
            regularization='separate_literature_glyph_blocks' if combined else 'shared_ridge',
            literature_alpha=literature_alpha if combined else (float(grid[best]) if arm.startswith('Literature') else np.nan),
            glyph_alpha=float(grid[best]) if combined or arm.startswith('Glyph') else np.nan)
        models[arm]=model
        tuning.extend(dict(arm=arm,alpha=float(a),literature_alpha=model['literature_alpha'],
            regularization=model['regularization'],**item) for a,item in zip(grid,diagnostics))
        for split in dict.fromkeys(cohort.split):
            mask=cohort.split.eq(split).to_numpy();both=len(np.unique(y[mask]))==2
            metrics.append(dict(arm=arm,split=split,n=int(mask.sum()),events=int(y[mask].sum()),
                features=values.shape[1],auc=roc_auc_score(y[mask],prob[mask]) if both else np.nan,
                brier=brier_score_loss(y[mask],prob[mask]),log_loss=log_loss(y[mask],prob[mask],labels=[0,1]),
                alpha=float(grid[best]),validation_auc=float(scores[best]),gradient_max=selected['gradient_max'],
                best_iteration=selected['best_iteration'],optimizer_iterations=selected['optimizer_iterations'],
                stopped_early=selected['stopped_early'],regularization=model['regularization'],
                literature_alpha=model['literature_alpha'],glyph_alpha=model['glyph_alpha']))
    return predictions,pd.DataFrame(metrics),pd.DataFrame(tuning),models


def paired_bootstrap(cohort,predictions,iterations=2000,seed=1238):
    if iterations<100:raise ValueError('Use at least 100 paired bootstrap replicates')
    arms=[name for name in predictions if name not in ['sample_id','split','y']];rng=np.random.default_rng(seed);intervals=[];comparisons=[];audits=[]
    for split in dict.fromkeys(cohort.split):
        if split in ['train','val']:continue
        mask=cohort.split.eq(split).to_numpy();sub=cohort.loc[mask].reset_index(drop=True);pred=predictions.loc[mask].reset_index(drop=True)
        y=sub.y.to_numpy();group=sub.patient_id if 'patient_id' in sub else sub.sample_id
        codes,unique=pd.factorize(group);ng=len(unique);values=[];attempts=0
        if len(np.unique(y))!=2:raise ValueError('Evaluation split needs both classes')
        while len(values)<iterations:
            attempts+=1
            if attempts>iterations*20:raise ValueError('Too few both-class resamples')
            weights=np.bincount(rng.choice(ng,ng,replace=True),minlength=ng)[codes]
            if np.dot(weights,y)==0 or np.dot(weights,1-y)==0:continue
            values.append([roc_auc_score(y,pred[arm],sample_weight=weights) for arm in arms])
        boot=np.asarray(values)
        observed=np.array([roc_auc_score(y,pred[arm]) for arm in arms])
        for j,arm in enumerate(arms):
            lo,hi=np.quantile(boot[:,j],[.025,.975]);intervals.append(dict(split=split,arm=arm,auc=float(observed[j]),ci_lower=float(lo),ci_upper=float(hi)))
        for i,j in itertools.combinations(range(len(arms)),2):
            difference=boot[:,j]-boot[:,i];delta=observed[j]-observed[i];lo,hi=np.quantile(difference,[.025,.975])
            p=(1+np.sum(np.abs(difference-delta)>=abs(delta)))/(iterations+1)
            comparisons.append(dict(split=split,reference=arms[i],comparison=arms[j],delta_auc=float(delta),
                ci_lower=float(lo),ci_upper=float(hi),p_value=float(p),primary=(arms[i],arms[j])==('Literature','Literature + Glyph')))
        audits.append(dict(split=split,replicates=iterations,attempted=attempts,rejected_single_class=attempts-iterations,
            method='Paired patient-cluster bootstrap; fixed fitted predictions; conditional on both classes',seed=seed))
    return pd.DataFrame(intervals),pd.DataFrame(comparisons),audits


def individual_associations(cohort,glyph,markers,manifest,frozen_glyph=None):
    literature,chosen=validate_inputs(cohort,glyph,markers,manifest)
    # Equal-count comparison, frozen by the manifest; never select with held-out estimates.
    count=min(len(literature),len(chosen));ranking=training_rankings(cohort,glyph,markers,manifest)
    selected={arm:ranking.loc[ranking.source.eq(arm),'column'].head(count).tolist() for arm in ['Literature','Glyph']}
    tables={'Literature':markers.set_index('sample_id').loc[cohort.sample_id], 'Glyph':glyph.set_index('sample_id').loc[cohort.sample_id]}
    rows=[];train=cohort.split.eq('train').to_numpy();family=2*count
    for arm,cols in selected.items():
        for col in cols:
            raw=tables[arm][col].to_numpy(float);mu=np.nanmean(raw[train]);sd=np.nanstd(raw[train])
            for split in ['train',*[s for s in dict.fromkeys(cohort.split) if s not in ['train','val']]]:
                mask=cohort.split.eq(split).to_numpy()&np.isfinite(raw);y=cohort.y.to_numpy()[mask]
                base=dict(arm=arm,marker=col,split=split,n=int(mask.sum()),events=int(y.sum()),n_tests=family,
                    odds_ratio=np.nan,ci_lower=np.nan,ci_upper=np.nan,p_value=np.nan,corrected_p_value=np.nan,converged=False,fit_status='insufficient_outcome_classes')
                if not np.isfinite(sd) or sd<=1e-8:
                    base['fit_status']='no_training_variation'
                elif len(np.unique(y))==2 and np.ptp((raw[mask]-mu)/sd)<=1e-8:
                    base['fit_status']='constant_marker_in_split'
                elif len(np.unique(y))==2 and sd>1e-8:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            result=sm.Logit(y,sm.add_constant((raw[mask]-mu)/sd,has_constant='add')).fit(disp=0,maxiter=200)
                        lo,hi=result.conf_int()[1];coef=result.params[1]
                        converged=bool(result.mle_retvals.get('converged')) and np.isfinite([coef,lo,hi,result.pvalues[1]]).all()
                        base['converged']=converged
                        base['fit_status']='ok' if converged else 'not_converged_or_nonfinite'
                        if converged:
                            base.update(odds_ratio=float(np.exp(coef)),ci_lower=float(np.exp(lo)),ci_upper=float(np.exp(hi)),
                                p_value=float(result.pvalues[1]),corrected_p_value=float(min(1,result.pvalues[1]*family)))
                    except (ValueError,np.linalg.LinAlgError,PerfectSeparationError):
                        base['fit_status']='fit_failed'
                if arm=='Glyph' and frozen_glyph is not None:
                    saved=frozen_glyph[(frozen_glyph.feature_idx==int(col.split('_')[-1]))&(frozen_glyph.split==split)]
                    if len(saved)>1:raise ValueError('Frozen associations must identify one dictionary and outcome')
                    base.update(odds_ratio=np.nan,ci_lower=np.nan,ci_upper=np.nan,p_value=np.nan,corrected_p_value=np.nan,converged=False)
                    if len(saved):
                        old=saved.iloc[0]
                        base.update(odds_ratio=float(old.odds_ratio),ci_lower=float(old.odds_ratio_ci_lower),ci_upper=float(old.odds_ratio_ci_upper),
                            p_value=float(old.odds_ratio_p_value),corrected_p_value=float(min(1,old.odds_ratio_p_value*family)),
                            converged=bool(np.isfinite([old.odds_ratio,old.odds_ratio_p_value]).all()),
                            estimate_source='Frozen original Glyph estimates; correction uses the matched comparison family')
                    base['fit_status']='ok' if base['converged'] else 'frozen_estimate_not_estimable'
                rows.append(base)
    return pd.DataFrame(rows)



def adjusted_associations(cohort,glyph,markers,manifest,confounder_cols,device='cpu'):
    from glyph.phewas.confounder_adjustment import FMRConfounderAdjustment,encode_confounders
    if not confounder_cols or set(confounder_cols)-set(cohort):raise ValueError('Supply existing cohort confounder columns')
    if set(confounder_cols)&{'sample_id','patient_id','split','y'}:raise ValueError('Identifiers, split labels and the outcome cannot be confounders')
    literature,chosen=validate_inputs(cohort,glyph,markers,manifest);count=min(len(literature),len(chosen))
    frames={split:frame[confounder_cols].reset_index(drop=True) for split,frame in cohort.groupby('split',sort=False)}
    encoded,names,categories=encode_confounders(frames)
    settings=manifest.get('confounder_adjustment',{})
    if set(settings)-{'n_estimators','n_splits','inner_splits','seed','n_jobs'}:raise ValueError('Unsupported adjustment settings; tree depth is fixed at 2')
    engine=FMRConfounderAdjustment(device=device,max_depth=2,**settings)
    tables={'Literature':markers.set_index('sample_id').loc[cohort.sample_id], 'Glyph':glyph.set_index('sample_id').loc[cohort.sample_id]}
    train=cohort.split.eq('train').to_numpy();rows=[]
    ranking=training_rankings(cohort,glyph,markers,manifest)
    for arm,cols in {arm:ranking.loc[ranking.source.eq(arm),'column'].head(count).tolist() for arm in ['Literature','Glyph']}.items():
        for col in cols:
            raw=tables[arm][col].to_numpy(float);mean=np.nanmean(raw[train]);sd=np.nanstd(raw[train]);sd=sd if sd>1e-8 else 1.
            for split in dict.fromkeys(cohort.split):
                if split=='val':continue
                mask=cohort.split.eq(split).to_numpy()
                result=engine.estimate((raw[mask]-mean)/sd,cohort.y.to_numpy()[mask],encoded[split])
                p=result['adjusted_odds_ratio_p_value']
                result.update(arm=arm,marker=col,split=split,n_tests=2*count,adjusted_corrected_p_value=min(1,p*2*count) if np.isfinite(p) else np.nan)
                rows.append(result)
    return pd.DataFrame(rows),dict(method=engine.method,max_depth=2,device=device,confounders=confounder_cols,encoded_columns=names,categories=categories,
        interpretation='Confounder-adjusted association using the FMR orthogonal score; not a causal-effect claim',settings={**engine.params,'n_splits':engine.n_splits,'inner_splits':engine.inner_splits,'inner_seed_rule':'seed + outer_fold_index + 1'})

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['cohort','glyph','markers','manifest','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--glyph-associations',type=Path)
    parser.add_argument('--adjust-confounders',action='store_true')
    parser.add_argument('--confounder-cols',nargs='+')
    parser.add_argument('--adjustment-device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args(argv)
    if not args.adjust_confounders and (args.confounder_cols or args.adjustment_device!='cpu'):parser.error('Use --adjust-confounders to enable adjustment options')
    cohort=pd.read_csv(args.cohort,dtype={'sample_id':str,'patient_id':str});glyph=pd.read_csv(args.glyph,dtype={'sample_id':str});markers=pd.read_csv(args.markers,dtype={'sample_id':str})
    manifest=json.loads(args.manifest.read_text())
    predictions,metrics,tuning,models=fit_models(cohort,glyph,markers,manifest)
    intervals,comparisons,audit=paired_bootstrap(cohort,predictions,args.bootstrap)
    associations=individual_associations(cohort,glyph,markers,manifest,pd.read_csv(args.glyph_associations) if args.glyph_associations else None)
    adjusted,adjustment_metadata=adjusted_associations(cohort,glyph,markers,manifest,args.confounder_cols,args.adjustment_device) if args.adjust_confounders else (None,None)
    args.output.mkdir(parents=True,exist_ok=False)
    feature_sets,ranking=comparison_feature_sets(cohort,glyph,markers,manifest)
    ranking.to_csv(args.output/'training_rankings.csv',index=False)
    (args.output/'feature_selections.json').write_text(json.dumps({arm:[dict(source=source,column=column) for source,column in features] for arm,features in feature_sets.items()},indent=2)+'\n')
    for name,frame in [('predictions',predictions),('metrics',metrics),('tuning',tuning),('auc_intervals',intervals),('auc_comparisons',comparisons),('individual_associations',associations)]:frame.to_csv(args.output/(name+'.csv'),index=False)
    if adjusted is not None:
        adjusted.to_csv(args.output/'adjusted_individual_associations.csv',index=False)
        (args.output/'confounder_adjustment.json').write_text(json.dumps(adjustment_metadata,indent=2)+'\n')
    for arm,model in models.items():np.savez(args.output/(arm.lower().replace(' + ','_plus_')+'_model.npz'),**model)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'bootstrap_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    inputs=[args.cohort,args.glyph,args.markers,args.manifest]+([args.glyph_associations] if args.glyph_associations else [])
    (args.output/'input_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2)+'\n')
    print(intervals.to_string(index=False));print(comparisons.to_string(index=False))


if __name__=='__main__':main()
