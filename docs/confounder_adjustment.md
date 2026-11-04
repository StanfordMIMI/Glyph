# Optional confounder adjustment

The package now uses `fmr_orthogonal` from the research implementation, ported
into `glyph/phewas/confounder_adjustment.py`. There is no runtime dependency on
the research repository. The public option remains confounder adjustment:

```python
glyph.hypothesis_search(
    adjust_confounders=True,
    confounder_cols=['age', 'sex', 'bmi'],
    adjustment_device='cuda',  # omit for CPU
)
```

Adjustment is off by default. Install `glyph-discovery[confounders]` to enable it.
The default remains 100 XGBoost rounds, depth 2, five outer and inner folds.
An explicit CUDA request requires a working CUDA XGBoost installation; the
existing check prevents silent CPU fallback. Per-study independence and the
Discovery-train feature units remain unchanged.

## Estimator update on 23 October 2026

The score is `(Y - expit(theta*T + r(W))) * (T - m(W))`. The exposure nuisance
uses `m(W) = E[v*T | W] / E[v | W]`, with variance `v = mu(T,W)*(1-mu(T,W))`
from the full outcome model, including T. This replaces the earlier weights
calculated from the baseline nuisance alone.

All nuisance fits are contained within each outer training fold. Inner
out-of-fold raw logits provide the pseudo-outcome and full-model variance
weights, normalized to mean one within the outer training fold. Both fold
layers are stratified; inner seeds are `seed + outer_fold_index + 1`.
The score root, signed Jacobian, centered influence variance and normal critical
value follow the research function. As in that reference, orthogonality and
valid Wald inference require appropriate nuisance consistency and convergence
rates; point-estimate parity does not establish confidence-interval coverage.
The outputs are adjusted associations without an additional causal-effect claim.

New result rows and manifests identify `fmr_orthogonal`. Existing adjusted column
names, multiple-testing corrections, eligible/missing counts and failure
statuses are retained. Older saved runs are not recalculated or relabeled;
the report harness distinguishes the legacy `fmr_sigmoid` method when reading
old comparison metadata. The literature comparison harness uses the same class.

## Verification

A frozen, standalone numerical reference is saved in
`tests/reference/fmr_orthogonal_20261023.py`, including the original function
checksum. Risk, protective and null synthetic cases compare the OR, p-value
and confidence limits at relative/absolute tolerance 1e-12, with explicit
depth 2, fold counts and seed. A separate test records actual XGBoost calls to
verify that variance weights come from inner out-of-fold full logits and never
use an outer held-out sample, and that those weights have mean one.

The tests also cover opt-in behavior, optional dependencies, GPU fallback,
missing observations, categorical alignment, insufficient events, saved
manifests, comparison corrections and reporting. No clinical cohort or demo
was rerun. See `test_outputs/fmr-orthogonal-tests.log` and its JUnit XML for
local verification results: 49 tests passed. A fresh locally installed wheel
also reproduced the independent reference OR, p-value and both confidence limits
with absolute differences of zero on its synthetic check; its evidence is in
`test_outputs/fmr-orthogonal/wheel-verification.json`.
