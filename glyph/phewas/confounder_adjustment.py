"""Optional confounder adjustment using full-model refitting and an orthogonal sigmoid score."""

import json
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold

from glyph.phewas.statistics import compute_odds_ratio


def encode_confounders(frames):
    """Encode categories using training levels; missing/unseen values stay missing."""
    train = frames['train']
    encoded = {split: [] for split in frames}
    names = []
    categories = {}
    for col in train.columns:
        if pd.api.types.is_numeric_dtype(train[col]):
            names.append(col)
            for split, frame in frames.items():
                encoded[split].append(frame[[col]].to_numpy(dtype=float))
        else:
            levels = list(train[col].dropna().unique())
            if not levels:
                raise ValueError(f"Confounder '{col}' has no observed training values")
            categories[col] = [str(level) for level in levels]
            names.extend(f'{col}={level}' for level in levels)
            for split, frame in frames.items():
                known_values = frame[col].where(frame[col].isin(levels))
                values = pd.Categorical(known_values, categories=levels)
                matrix = pd.get_dummies(values).to_numpy(dtype=float, copy=True)
                matrix[pd.isna(values), :] = np.nan
                encoded[split].append(matrix)
    return (
        {split: np.column_stack(parts) for split, parts in encoded.items()},
        names,
        categories,
    )


class FMRConfounderAdjustment:
    """Cross-fitted confounder adjustment, with Wald intervals and p-values.

    Uses fmr_orthogonal: nested stratified cross-fitting and full-model variance
    weights for the exposure nuisance. Orthogonality/inference require suitable
    nuisance consistency and rates; this does not establish CI coverage.
    Wald inference treats studies as independent observations.
    Feature units are supplied by the caller (PheWAS uses Discovery-train SDs).
    """

    method = 'fmr_orthogonal'

    def __init__(self, *, device='cpu', max_depth=2, n_estimators=100,
                 n_splits=5, inner_splits=None, seed=42, n_jobs=1):
        if device not in ('cpu', 'cuda'):
            raise ValueError("adjustment_device must be 'cpu' or 'cuda'")
        if max_depth < 1 or n_estimators < 1 or n_splits < 2:
            raise ValueError('Positive depth/estimators and at least two folds are required')
        inner_splits = n_splits if inner_splits is None else inner_splits
        if inner_splits < 2:
            raise ValueError('inner_splits must be at least two')
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                'Confounder adjustment requires pip install "glyph-discovery[confounders]"'
            ) from exc
        if int(xgb.__version__.split('.')[0]) < 2:
            raise ImportError('Confounder adjustment requires xgboost>=2.0')
        if device == 'cuda':
            import torch
            if not torch.cuda.is_available() or not xgb.build_info().get('USE_CUDA', False):
                raise ValueError(
                    "adjustment_device='cuda' requires an available CUDA GPU and a CUDA-enabled "
                    "XGBoost installation; choose adjustment_device='cpu' to use the CPU"
                )
        self.xgb = xgb
        self.n_splits = n_splits
        self.inner_splits = inner_splits
        self.seed = seed
        self.params = dict(tree_method='hist', device=device, max_depth=max_depth,
                           n_estimators=n_estimators, random_state=seed,
                           n_jobs=n_jobs, verbosity=0)

    def estimate(self, feature_values, outcome_values, confounders):
        t = np.asarray(feature_values, dtype=float)
        y = np.asarray(outcome_values, dtype=float)
        w = np.asarray(confounders, dtype=float)
        if t.ndim != 1 or y.shape != t.shape or w.ndim != 2 or len(w) != len(t):
            raise ValueError('Features, outcomes and confounder rows must be aligned')
        if w.shape[1] == 0:
            raise ValueError('At least one confounder is required')
        invalid_labels = np.isfinite(y) & ~np.isin(y, [-1, 0, 1])
        if invalid_labels.any():
            raise ValueError('Confounder adjustment supports binary 0/1 outcomes (-1/NaN missing)')

        valid_y = np.isin(y, [0, 1])
        valid_t = np.isfinite(t)
        valid_w = np.isfinite(w).all(axis=1)
        eligible = valid_y & valid_t & valid_w
        result = {
            'adjustment_method': self.method,
            'adjustment_weighting': 'inner_oof_full_model_variance',
            'adjustment_cross_fitting': 'nested_stratified',
            'adjustment_device': self.params['device'],
            'adjustment_max_depth': self.params['max_depth'],
            'adjustment_n_estimators': self.params['n_estimators'],
            'adjustment_n_splits': self.n_splits,
            'adjustment_inner_splits': self.inner_splits,
            'adjustment_seed': self.seed,
            'adjustment_n_total': len(y),
            'adjustment_n': int(eligible.sum()),
            'adjustment_n_events': int((y[eligible] == 1).sum()),
            'adjustment_n_excluded': int((~eligible).sum()),
            # These counts are sequential, so they sum to n_excluded.
            'adjustment_n_missing_outcome': int((~valid_y).sum()),
            'adjustment_n_missing_feature': int((valid_y & ~valid_t).sum()),
            'adjustment_n_missing_confounder': int((valid_y & valid_t & ~valid_w).sum()),
            'adjustment_status': 'ok',
            'adjusted_odds_ratio': np.nan,
            'adjusted_odds_ratio_ci_lower': np.nan,
            'adjusted_odds_ratio_ci_upper': np.nan,
            'adjusted_odds_ratio_p_value': np.nan,
            'adjusted_coefficient': np.nan,
            'adjusted_standard_error': np.nan,
            'matched_unadjusted_odds_ratio': np.nan,
            'matched_unadjusted_odds_ratio_ci_lower': np.nan,
            'matched_unadjusted_odds_ratio_ci_upper': np.nan,
            'matched_unadjusted_odds_ratio_p_value': np.nan,
        }
        t, y, w = t[eligible], y[eligible], w[eligible]
        if len(np.unique(y)) < 2:
            result['adjustment_status'] = 'insufficient_outcome_classes'
            return result
        if np.std(t) < 1e-12:
            result['adjustment_status'] = 'constant_feature'
            return result
        matched = compute_odds_ratio(t, y, seed=self.seed)
        for key in ('odds_ratio', 'odds_ratio_ci_lower', 'odds_ratio_ci_upper'):
            result[f'matched_unadjusted_{key}'] = matched[key]
        result['matched_unadjusted_odds_ratio_p_value'] = matched['p_value']
        if len(t) < self.n_splits or len(t) - int(np.ceil(len(t) / self.n_splits)) < self.inner_splits:
            result['adjustment_status'] = 'insufficient_samples_for_folds'
            return result

        y = y.astype(int)
        if np.bincount(y, minlength=2).min() < self.n_splits:
            result['adjustment_status'] = 'insufficient_events_for_folds'
            return result
        outer = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed)
        r_hat = np.empty(len(t))
        m_hat = np.empty(len(t))
        try:
            for fold, (train, test) in enumerate(outer.split(w, y)):
                t_train, y_train, w_train = t[train], y[train], w[train]
                if np.bincount(y_train, minlength=2).min() < self.inner_splits:
                    result['adjustment_status'] = 'insufficient_events_for_folds'
                    return result
                inner = StratifiedKFold(
                    self.inner_splits, shuffle=True, random_state=self.seed + fold + 1)
                pseudo_y = np.empty(len(train))
                a_oof = np.empty(len(train))
                a_test = []
                for fit, holdout in inner.split(w_train, y_train):
                    joint = self.xgb.XGBClassifier(
                        **self.params, objective='binary:logistic', eval_metric='logloss').fit(
                            np.column_stack([t_train[fit], w_train[fit]]), y_train[fit])
                    if self.params['device'] == 'cuda':
                        config = json.loads(joint.get_booster().save_config())
                        actual_device = config['learner']['generic_param']['device']
                        if not actual_device.startswith('cuda'):
                            raise RuntimeError(
                                'XGBoost fell back to CPU despite adjustment_device="cuda"; '
                                'check the CUDA installation or explicitly request "cpu"')
                    pseudo_y[holdout] = joint.predict(
                        np.column_stack([t_train[holdout], w_train[holdout]]),
                        output_margin=True)
                    a = self.xgb.XGBRegressor(
                        **self.params, objective='reg:squarederror').fit(w_train[fit], t_train[fit])
                    a_oof[holdout] = a.predict(w_train[holdout])
                    a_test.append(a.predict(w[test]))
                residual = t_train - a_oof
                denominator = np.sum(residual ** 2)
                if not np.isfinite(denominator) or denominator <= 0:
                    result['adjustment_status'] = 'no_residual_feature_variation'
                    return result
                beta_check = np.sum(pseudo_y * residual) / denominator
                refit = self.xgb.XGBRegressor(
                    **self.params, objective='reg:squarederror').fit(w_train, pseudo_y)
                r_hat[test] = refit.predict(w[test]) - beta_check * np.mean(a_test, axis=0)

                # Only inner OOF logits from this outer training fold supply weights.
                v = expit(pseudo_y) * expit(-pseudo_y)
                if not np.isfinite(v).all() or np.any(v <= 0):
                    result['adjustment_status'] = 'degenerate_nuisance_weights'
                    return result
                v = v / np.mean(v)
                model = self.xgb.XGBRegressor(
                    **self.params, objective='reg:squarederror').fit(
                        w_train, t_train, sample_weight=v)
                m_hat[test] = model.predict(w[test])
            if not (np.isfinite(r_hat).all() and np.isfinite(m_hat).all()):
                result['adjustment_status'] = 'nonfinite_nuisance_predictions'
                return result
            t_res = t - m_hat

            def score(theta):
                return np.mean((y - expit(theta * t + r_hat)) * t_res)

            if not score(-20) * score(20) < 0:
                result['adjustment_status'] = 'score_root_not_bracketed'
                return result
            theta = brentq(score, -20, 20, maxiter=200)
            eta = theta * t + r_hat
            psi = (y - expit(eta)) * t_res
            jacobian = -np.mean(expit(eta) * expit(-eta) * t * t_res)
            if not np.isfinite(jacobian) or jacobian >= -1e-12:
                result['adjustment_status'] = 'degenerate_score_derivative'
                return result
            se = np.sqrt(np.mean((psi - psi.mean()) ** 2) / len(t)) / abs(jacobian)
            if not np.isfinite(se) or se <= 0:
                result['adjustment_status'] = 'invalid_standard_error'
                return result
            critical = norm.ppf(0.975)
            with np.errstate(over='raise', invalid='raise'):
                result.update(
                    adjusted_odds_ratio=float(np.exp(theta)),
                    adjusted_odds_ratio_ci_lower=float(np.exp(theta - critical * se)),
                    adjusted_odds_ratio_ci_upper=float(np.exp(theta + critical * se)),
                    adjusted_odds_ratio_p_value=float(2 * norm.sf(abs(theta / se))),
                    adjusted_coefficient=float(theta),
                    adjusted_standard_error=float(se),
                )
        except (ValueError, FloatingPointError, self.xgb.core.XGBoostError) as exc:
            result['adjustment_status'] = 'fit_failed'
            result['adjustment_error'] = str(exc)
        return result
