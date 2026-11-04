"""
Automated interpretation of SAE features using LLMs (Gemini API).

For each top hypothesis from PheWAS, finds the highest-activating samples,
sends their textual representations to the LLM for interpretation, then
validates the interpretation on held-out test samples using a discriminative
scoring approach.
"""

import re
import json
import time
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class AutoInterp:
    """
    LLM-based interpretation of SAE features.

    Two-stage process per feature:
    1. Interpretation: Show high-activating (cohort) samples alongside random
       reference samples to the LLM and ask what characterizes the cohort.
    2. Discrimination: Mix high-activating test samples with random test samples,
       provide the interpretation, and ask the LLM to classify each as
       belonging (1) or not belonging (0) to the cohort. Score as accuracy.

    Args:
        cfg: Hydra config with autointerp settings
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.top_n = cfg.autointerp.top_n_hypotheses
        self.bottom_n = cfg.autointerp.bottom_n_hypotheses
        self.top_pct = cfg.autointerp.top_pct_activating
        self.n_fit = cfg.autointerp.n_fit_samples
        self.n_test = cfg.autointerp.n_test_samples
        self.seed = cfg.seed

        try:
            from google import genai
        except ImportError as exc:
            raise ImportError("Install glyph-discovery[autointerp] to use automated interpretation.") from exc

        llm = cfg.autointerp.llm
        self.model_name = llm.model
        self.temperature = llm.temperature
        self.max_tokens = llm.max_tokens

        from glyph.interpretation.settings import connection
        provider,key,project=connection(llm)
        if provider == 'vertex':
            if not project:raise ValueError('Set vertex_project or GOOGLE_CLOUD_PROJECT for Vertex interpretation')
            self.client = genai.Client(vertexai=True, project=project,
                                       location=llm.vertex_location)
        elif provider in ('gemini_api', 'gemini'):
            if not key:
                raise ValueError(
                    "Gemini API key not set. Provide it via "
                    "GEMINI_API_KEY environment variable or autointerp.llm.gemini_api_key."
                )
            self.client = genai.Client(api_key=key)
        else:
            raise ValueError(f"Unknown LLM provider: {llm.provider}")

    def _get_text_column(self, modality: str) -> str:
        """Resolve which metadata column holds text for a given modality."""
        text_map = self.cfg.autointerp.get('autointerp_text_map', None)
        if text_map is None:
            text_map = self.cfg.data.get('autointerp_text_map', None)
        if text_map is None:
            raise ValueError(
                f"No autointerp_text_map configured. Set "
                f"autointerp.autointerp_text_map or data.autointerp_text_map "
                f"to map modality columns to text columns "
                f"(e.g. {{image_path: report}})."
            )
        if hasattr(text_map, '_content'):
            text_map = dict(text_map)
        if modality not in text_map:
            raise ValueError(
                f"Modality '{modality}' not found in autointerp_text_map. "
                f"Available mappings: {dict(text_map)}"
            )
        return text_map[modality]

    def _load_metadata(self) -> pd.DataFrame:
        metadata = pd.read_csv(self.cfg.data.metadata,dtype={"sample_id":str})
        if "sample_id" in metadata and (metadata.sample_id.isna().any() or metadata.sample_id.duplicated().any()):
            raise ValueError("Interpretation metadata needs unique nonmissing sample IDs")
        if metadata.columns[0].startswith('Unnamed'):
            metadata = metadata.drop(columns=[metadata.columns[0]])
        return metadata

    def _get_sample_text(
        self,
        metadata: pd.DataFrame,
        sample_id: str,
        text_col: str,
        id_col: str,
    ) -> str:
        """Get the text representation of a sample."""
        row = metadata[metadata[id_col] == sample_id]
        if row.empty:
            return f"[No data found for sample {sample_id}]"
        text = str(row[text_col].iloc[0])
        if text == 'nan' or not text.strip():
            return f"[No text available for sample {sample_id}]"
        if '/' in text and '\n' not in text and len(text) < 500:
            try:
                return Path(text).read_text().strip()
            except (FileNotFoundError, PermissionError, OSError):
                pass
        return text.strip()

    def _call_gemini(self, prompt: str) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        'temperature': self.temperature,
                        'max_output_tokens': self.max_tokens,
                    },
                )
                return response.text
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"    Interpretation request failed ({type(e).__name__}); retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise RuntimeError("Interpretation API request failed after three attempts; response and credentials omitted") from None

    def _select_top_activating_samples(
        self,
        feature_idx: int,
        sae_features: torch.Tensor,
        sample_ids: List[str],
        n_samples: int,
        rng: np.random.RandomState,
    ) -> List[str]:
        """
        Select samples from the top-activating pool for a given feature.
        Returns sample IDs sorted by activation strength (highest first).
        """
        if isinstance(sae_features, torch.Tensor):
            activations = sae_features[:, feature_idx].cpu().numpy()
        else:
            activations = sae_features[:, feature_idx]

        n_total = len(activations)
        n_top = max(1, int(n_total * self.top_pct))

        top_indices = np.argsort(activations)[-n_top:]
        top_indices = top_indices[activations[top_indices] > 0]

        if len(top_indices) == 0:
            return []

        n_select = min(n_samples, len(top_indices))
        selected = rng.choice(top_indices, size=n_select, replace=False)

        # Sort by activation strength (descending) so the prompt lists
        # the most strongly-activating samples first
        selected = selected[np.argsort(activations[selected])[::-1]]

        return [sample_ids[i] for i in selected]

    def _select_reference_samples(
        self,
        feature_idx: int,
        sae_features: torch.Tensor,
        sample_ids: List[str],
        n_samples: int,
        rng: np.random.RandomState,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Select random low-activating samples as reference (non-cohort) contrast.
        These are samples that do NOT strongly activate the feature.
        """
        if isinstance(sae_features, torch.Tensor):
            activations = sae_features[:, feature_idx].cpu().numpy()
        else:
            activations = sae_features[:, feature_idx]

        # Strictly non-activating samples
        low_mask = activations <= 1e-8

        if exclude_ids:
            exclude_set = set(exclude_ids)
            low_mask &= np.array([sid not in exclude_set for sid in sample_ids])

        low_indices = np.where(low_mask)[0]
        if len(low_indices) == 0:
            return []

        n_select = min(n_samples, len(low_indices))
        selected = rng.choice(low_indices, size=n_select, replace=False)
        return [sample_ids[i] for i in selected]

    def _format_patient_data(
        self,
        sample_ids: List[str],
        metadata: pd.DataFrame,
        text_col: str,
        id_col: str,
        label: str = "Patient",
    ) -> str:
        """Format samples as numbered patient data entries for the prompt."""
        lines = []
        for i, sid in enumerate(sample_ids, 1):
            text = self._get_sample_text(metadata, sid, text_col, id_col)
            lines.append(f"--- {label} {i} ---\n{text}\n")
        return "\n".join(lines)

    def _extract_interpretation(self, response: str) -> str:
        """
        Extract the interpretation from the LLM response.
        Looks for the line starting with '* This cohort is characterized by'.
        Falls back to the full response if the pattern is not found.
        """
        # Search from the end — the conclusion is always the last starred line
        lines = response.split('\n')
        for line in reversed(lines):
            stripped = line.strip()
            if stripped.startswith('*') and 'characterized by' in stripped.lower():
                return stripped.lstrip('*').strip()
        # Fall back to last *-prefixed line
        for line in reversed(lines):
            stripped = line.strip()
            if stripped.startswith('*') and len(stripped) > 2:
                return stripped[1:].strip()
        return response.strip()

    def _parse_discriminative_predictions(
        self,
        response: str,
        n_samples: int,
        labels: List[int],
    ) -> Dict[str, any]:
        """
        Parse 1/0 predictions from the discriminative scoring response
        and compute accuracy against ground truth labels.

        Args:
            response: Raw LLM response text
            n_samples: Total number of samples in the mixed set
            labels: Ground truth labels (1=cohort, 0=reference)

        Returns:
            Dict with predictions, accuracy, and match counts
        """
        if n_samples!=len(labels):raise ValueError('Labels and expected prediction count differ')
        section=response.split('* PREDICTIONS:',1)[-1].strip()
        matches=re.findall(r'(?m)^\s*(\d+)[.:]\s*([01])(?:\s|$)',section)
        predictions=[]
        if matches:
            numbers=[int(number) for number,_ in matches]
            if len(numbers)==n_samples and len(set(numbers))==n_samples and set(numbers)==set(range(1,n_samples+1)):
                values={int(number):int(value) for number,value in matches};predictions=[values[i] for i in range(1,n_samples+1)]
        elif section.startswith('['):
            try:
                values=json.loads(section)
                if isinstance(values,list) and len(values)==n_samples and all(type(v) is int and v in [0,1] for v in values):predictions=values
            except json.JSONDecodeError:pass
        valid=len(predictions)==n_samples and n_samples>0
        correct=sum(p==label for p,label in zip(predictions,labels)) if valid else None
        tp=sum(p==1 and label==1 for p,label in zip(predictions,labels)) if valid else None
        tn=sum(p==0 and label==0 for p,label in zip(predictions,labels)) if valid else None
        positive=sum(labels);negative=n_samples-positive
        sensitivity=tp/positive if valid and positive else None
        specificity=tn/negative if valid and negative else None
        accuracy=correct/n_samples if valid else None
        return dict(predictions=predictions,accuracy=accuracy,correct=correct,true_positives=tp,n_positive=positive,n_total=n_samples,
            sensitivity=sensitivity,specificity=specificity,balanced_accuracy=(sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
            status='scored' if valid else 'invalid_or_incomplete_predictions')

    def run_for_modality(
        self,
        modality: str,
        sae_features: Dict[str, torch.Tensor],
        sample_ids: Dict[str, List[str]],
        hypotheses_df: pd.DataFrame,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run AutoInterp for a single modality.

        For each top hypothesis:
        1. Select top-activating train samples (cohort) and random reference
           train samples. Send to LLM with full_reports_prompt to get interpretation.
        2. Select top-activating test samples (positive) and random reference
           test samples (negative). Shuffle and send with the interpretation
           using discriminative_autointerp_score_prompt. Parse predictions
           and compute accuracy.
        """
        metadata = self._load_metadata()
        text_col = self._get_text_column(modality)

        id_col = 'sample_id' if 'sample_id' in metadata.columns else metadata.columns[0]

        rng = np.random.RandomState(self.seed)

        interp_template = self.cfg.autointerp.full_reports_prompt
        discrim_template = self.cfg.autointerp.discriminative_autointerp_score_prompt

        outcome_cols = hypotheses_df['outcome'].unique()
        top_k_values = hypotheses_df['top_k'].unique()

        results = []
        feature_cache = {}

        for outcome_col in outcome_cols:
            for top_k in top_k_values:
                mask = (
                    (hypotheses_df['outcome'] == outcome_col) &
                    (hypotheses_df['modality'] == modality) &
                    (hypotheses_df['top_k'] == top_k) &
                    (hypotheses_df['split'] == 'train')
                )
                outcome_hyp = hypotheses_df[mask].copy()

                if outcome_hyp.empty:
                    continue

                # Top N (risk features, highest OR)
                top_hyp = outcome_hyp[outcome_hyp.odds_ratio>1].nsmallest(self.top_n, 'train_odds_ratio_rank')
                # Bottom N (protective features, lowest OR)
                bottom_hyp = outcome_hyp[outcome_hyp.odds_ratio<1].nlargest(self.bottom_n, 'train_odds_ratio_rank') \
                    if self.bottom_n > 0 else outcome_hyp.iloc[0:0]
                outcome_hyp = pd.concat([
                    top_hyp.assign(_direction='risk'),
                    bottom_hyp.assign(_direction='protective'),
                ]).drop_duplicates(subset=['feature_name'])

                if verbose:
                    print(f"\n  AutoInterp: {outcome_col} ({modality}, k={top_k})"
                          f" - {len(top_hyp)} risk + {len(bottom_hyp)} protective hypotheses")

                pbar = tqdm(outcome_hyp.iterrows(), total=len(outcome_hyp),
                            desc=f"  AutoInterp ({modality}, {outcome_col[:20]})", unit="feat")
                for _, hyp_row in pbar:
                    feature_idx = hyp_row['feature_idx']
                    feature_name = hyp_row['feature_name']
                    rank = hyp_row['train_odds_ratio_rank']
                    direction = hyp_row['_direction']
                    pbar.set_postfix(feature=feature_name, OR=f"{hyp_row['odds_ratio']:.2f}", dir=direction)

                    cache_key=(modality,int(top_k),int(feature_idx))
                    if cache_key in feature_cache:
                        results.append(dict(feature_cache[cache_key],outcome=outcome_col,train_odds_ratio_rank=rank,odds_ratio=hyp_row['odds_ratio'],direction=direction))
                        continue

                    # --- Stage 1: Interpretation ---

                    # Select cohort samples (high-activating) from train
                    cohort_ids = self._select_top_activating_samples(
                        feature_idx=feature_idx,
                        sae_features=sae_features['train'],
                        sample_ids=sample_ids['train'],
                        n_samples=self.n_fit,
                        rng=rng,
                    )

                    if len(cohort_ids) < 3:
                        if verbose:
                            print(f"      Skipping: only {len(cohort_ids)} "
                                  f"activating samples found")
                        continue

                    # Select reference samples (low-activating) from train
                    reference_ids = self._select_reference_samples(
                        feature_idx=feature_idx,
                        sae_features=sae_features['train'],
                        sample_ids=sample_ids['train'],
                        n_samples=len(cohort_ids),
                        rng=rng,
                        exclude_ids=cohort_ids,
                    )

                    n_pairs=min(len(cohort_ids),len(reference_ids))
                    if n_pairs<3:
                        continue
                    cohort_ids=cohort_ids[:n_pairs];reference_ids=reference_ids[:n_pairs]

                    # Format patient data
                    cohort_text = self._format_patient_data(
                        cohort_ids, metadata, text_col, id_col, label="Cohort Patient",
                    )
                    reference_text = self._format_patient_data(
                        reference_ids, metadata, text_col, id_col, label="Random Reference Patient",
                    )

                    # Build interpretation prompt
                    interp_prompt = interp_template.format(
                        patient_data=cohort_text,
                        reference_patients=reference_text,
                    )

                    pbar.set_description(f"  AutoInterp ({modality}) [gen]")
                    interp_response = self._call_gemini(interp_prompt)
                    interpretation = self._extract_interpretation(interp_response)

                    if verbose:
                        interp_short = interpretation[:100] + "..." \
                            if len(interpretation) > 100 else interpretation
                        tqdm.write(f"      {feature_name}: {interp_short}")

                    # --- Stage 2: Discriminative scoring ---

                    # Select positive test samples (high-activating)
                    test_positive_ids = self._select_top_activating_samples(
                        feature_idx=feature_idx,
                        sae_features=sae_features['test'],
                        sample_ids=sample_ids['test'],
                        n_samples=self.n_test,
                        rng=rng,
                    )

                    # Select equal number of negative test samples (low-activating)
                    n_neg = len(test_positive_ids)
                    test_negative_ids = self._select_reference_samples(
                        feature_idx=feature_idx,
                        sae_features=sae_features['test'],
                        sample_ids=sample_ids['test'],
                        n_samples=n_neg,
                        rng=rng,
                        exclude_ids=test_positive_ids,
                    )

                    discrim_score = {}
                    discrim_prompt_text = ""
                    discrim_response = ""

                    if len(test_positive_ids) > 0 and len(test_negative_ids) > 0:
                        # Build mixed set: interleave positive and negative,
                        # then shuffle
                        n_pairs=min(len(test_positive_ids),len(test_negative_ids))
                        test_positive_ids=test_positive_ids[:n_pairs];test_negative_ids=test_negative_ids[:n_pairs]
                        mixed_ids = list(test_positive_ids) + list(test_negative_ids)
                        labels = [1] * len(test_positive_ids) + [0] * len(test_negative_ids)

                        # Shuffle together
                        shuffle_idx = rng.permutation(len(mixed_ids))
                        mixed_ids = [mixed_ids[i] for i in shuffle_idx]
                        labels = [labels[i] for i in shuffle_idx]

                        mixed_text = self._format_patient_data(
                            mixed_ids, metadata, text_col, id_col, label="Sample",
                        )

                        discrim_prompt_text = discrim_template.format(
                            patient_data=mixed_text,
                            interpretation=interpretation,
                        )

                        pbar.set_description(f"  AutoInterp ({modality}) [rate]")
                        discrim_response = self._call_gemini(discrim_prompt_text)
                        discrim_score = self._parse_discriminative_predictions(
                            discrim_response, len(mixed_ids), labels,
                        )

                        if verbose and discrim_score['status']=='scored':
                            tqdm.write(f"      {feature_name} acc: "
                                       f"{discrim_score['correct']}/{discrim_score['n_total']} "
                                       f"({discrim_score['accuracy']:.0%})")
                    else:
                        if verbose:
                            print(f"      Skipping discrimination: insufficient "
                                  f"test samples (pos={len(test_positive_ids)}, "
                                  f"neg={len(test_negative_ids)})")
                        discrim_score = {
                            'predictions': [],
                            'accuracy': None,
                            'correct': 0,
                            'true_positives': 0,
                            'n_positive': 0,
                            'n_total': 0,
                        }
                        labels = []
                        mixed_ids = []

                    result_row = {
                        'feature_idx': feature_idx,
                        'feature_name': feature_name,
                        'outcome': outcome_col,
                        'modality': modality,
                        'top_k': top_k,
                        'train_odds_ratio_rank': rank,
                        'odds_ratio': hyp_row['odds_ratio'],
                        'direction': direction,
                        'interpretation': interpretation,
                        'fit_sample_ids': cohort_ids,
                        'fit_sample_count': len(cohort_ids),
                        'reference_sample_ids': reference_ids,
                        'reference_sample_count': len(reference_ids),
                        'test_positive_ids': test_positive_ids,
                        'test_negative_ids': test_negative_ids,
                        'test_mixed_ids': mixed_ids,
                        'test_labels': labels,
                        'test_predictions': discrim_score.get('predictions', []),
                        'test_accuracy': discrim_score.get('accuracy'),
                        'test_balanced_accuracy': discrim_score.get('balanced_accuracy'),
                        'test_sensitivity': discrim_score.get('sensitivity'),
                        'test_specificity': discrim_score.get('specificity'),
                        'interpretation_status': discrim_score.get('status','insufficient_test_samples'),
                        'interpretation_model': self.model_name,
                        'test_correct': discrim_score.get('correct', 0),
                        'test_n_total': discrim_score.get('n_total', 0),
                        'interp_prompt': interp_prompt,
                        'interp_response': interp_response,
                        'discrim_prompt': discrim_prompt_text,
                        'discrim_response': discrim_response,
                    }
                    results.append(result_row)
                    feature_cache[cache_key]=result_row

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)

    def run_all(
        self,
        sae_features: Dict[str, Dict[str, torch.Tensor]],
        sample_ids: Dict[str, List[str]],
        hypotheses: pd.DataFrame,
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run AutoInterp for all modalities.

        Args:
            sae_features: Dict[modality][split] -> feature tensor
            sample_ids: Dict[split] -> list of sample IDs
            hypotheses: Full PheWAS results DataFrame
            output_dir: Where to save intermediate results
            verbose: Whether to print progress

        Returns:
            Combined DataFrame with all interpretation results
        """
        all_results = []
        modalities = hypotheses['modality'].unique()

        for modality in modalities:
            if modality not in sae_features:
                print(f"  Warning: No SAE features for modality '{modality}', skipping")
                continue

            try:
                self._get_text_column(modality)
            except ValueError as e:
                print(f"  Warning: {e}. Skipping AutoInterp for '{modality}'.")
                continue

            modality_hyp = hypotheses[hypotheses['modality'] == modality]

            if verbose:
                print(f"\n  Running AutoInterp for modality: {modality}")

            df = self.run_for_modality(
                modality=modality,
                sae_features=sae_features[modality],
                sample_ids=sample_ids,
                hypotheses_df=modality_hyp,
                verbose=verbose,
            )

            if not df.empty:
                all_results.append(df)

                if output_dir:
                    out_path = Path(output_dir)
                    out_path.mkdir(parents=True, exist_ok=True)
                    modality_name = modality.replace('_path', '')
                    save_path = out_path / f"autointerp_results_{modality_name}.pkl"
                    df.to_pickle(save_path)
                    if verbose:
                        print(f"  Saved intermediate results to: {save_path}")

        if not all_results:
            return pd.DataFrame()

        combined = pd.concat(all_results, ignore_index=True)

        if output_dir:
            save_path = Path(output_dir) / "autointerp_results.pkl"
            combined.to_pickle(save_path)
            if verbose:
                print(f"\n  Saved combined AutoInterp results to: {save_path}")

        return combined
