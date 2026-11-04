# **Detailed Analysis: Medical Hypothesis Generation Codebase**

## **Executive Summary**

The Medical Hypothesis Generation system is a sophisticated research platform that uses **Sparse Autoencoders (SAEs)** to discover interpretable features from multimodal medical data, then conducts **Phenome-Wide Association Studies (PheWAS)** to generate and rank scientific hypotheses, which are finally interpreted using **Large Language Models (LLMs)**.

---

## **1. Overall Architecture & Pipeline**

### **Core Workflow**

```
Multimodal Medical Data → Embedding Extraction → SAE Training → PheWAS → AutoInterp → Hypotheses
```

### **Data Flow Detail**

1. **Input**: Medical imaging (CT scans), clinical text (radiology reports, clinical notes), lab values, demographics
2. **Embedding Extraction**: Pretrained encoders convert raw data to dense embeddings
3. **SAE Training**: Learn sparse, interpretable features from embeddings
4. **Linear Probe**: Evaluate embeddings vs sparse codes for outcome prediction
5. **PheWAS**: Test each sparse feature against each outcome for associations
6. **Hypothesis Ranking**: Rank feature-outcome pairs by odds ratios
7. **AutoInterpretation**: Use LLMs to characterize cohorts defined by high-activation examples

---

## **2. Key Components Deep Dive**

### **A. Data Loading (`Data/get_data.py`)**

**Purpose**: Universal data loader with caching for multimodal medical data

**Key Features**:
- Loads from `metadata.csv` with outcome labels and modality paths
- Supports multiple modalities: images, text (findings, clinical notes), labs, demographics
- Lazy loading with pickle caching to avoid recomputation
- Normalization statistics computed from training set
- Returns datasets, dataloaders, and normalization stats

**Medical-Specific Aspects**:
- Hardcoded paths to Stanford/Tri-Valley datasets
- Disease labels (5-year prognosis, phecodes, 30 diagnosis labels)
- Demographics/lifestyle (age, sex, smoking, BMI)
- Lab results (cholesterol, HDL) with temporal windowing
- Medication orders (BP medications)
- Complex EHR preprocessing (sparse lab data, PREVENT data)

---

### **B. Embedding Extraction (`Pretraining/util/extract_model_output.py`)**

**Purpose**: Extract embeddings from pretrained models using PyTorch Lightning

**Implementation**:
```python
class EmbeddingExtractor(LightningModule):
    - Wraps model in Lightning module
    - Uses predict_step() for inference
    - Extracts specified output_fields from model
    - Returns dict of numpy arrays
```

**Key Features**:
- GPU acceleration via PyTorch Lightning Trainer
- Batch processing with memory management (gc.collect(), torch.cuda.empty_cache())
- Flexible output field extraction (can extract from any layer)
- Returns: `{field_name: np.ndarray}` for all requested fields

**Usage**:
```python
labels = extract_vectors_for_split(config, dataloader, model,
                                   output_fields=['embedding_layer/output'],
                                   fields=['anon_accession'])
```

---

### **C. Sparse Autoencoder Training (`Analysis/run/fit_sae.py`)**

**Purpose**: Train sparse autoencoders to discover interpretable features

#### **Architecture**

**Model**: TopK Sparse Autoencoder
- **Encoder**: Linear layer + TopK activation (only K largest activations kept)
- **Decoder**: Linear layer with L2-normalized weights
- **Loss**: MSE reconstruction + L1 sparsity + ghost gradient regularization

**Key Innovations**:

1. **Matryoshka SAEs** (Nested Dictionaries)
   - Train multiple dictionary sizes simultaneously: [128, 512, 2048, 8192]
   - Compute reconstruction loss for each nested prefix
   - Final loss = average of all prefix losses
   - Allows multi-resolution feature extraction

2. **TopK Activation**
   - Only top K features activate per sample
   - Enforces sparsity structurally (not just via L1 penalty)
   - More stable than standard L1-regularized autoencoders

3. **Ghost Gradient Recovery**
   - Identifies "dead" features (never activate)
   - Computes auxiliary loss on dead features using residual
   - Revives dead features during training
   - Critical for utilizing full dictionary capacity

4. **Weight Normalization**
   - Decoder weights normalized to unit norm at each step
   - Prevents feature magnitude collapse
   - Gradients projected to tangent space (remove parallel component)

#### **Training Details**

```python
class TrainSparseAutoencoder(ConceptModel):
    - Optimizer: Adam with configurable LR
    - Early stopping: Monitors validation_min_loss_epoch
    - Metrics: MSE loss, L1 loss, ghost loss, R², n_alive features
    - Checkpointing: Saves best model by validation loss
```

**Loss Function**:
```python
loss = mse_loss + sparsity_coefficient * L1_norm + aux_scale * ghost_loss
```

**Outputs**:
- `sae_output.pkl`: All sparse activations for all samples
- `sae_alive_output.pkl`: Only alive features (filtered by activation frequency)
- `sae_alive_output_dense.pkl`: Dense activations before TopK

---

### **D. Linear Probe (`Analysis/run/fit_statistical_model.py`)**

**Purpose**: Evaluate embeddings and sparse codes for outcome prediction

#### **Two Implementations**

**1. CPU-based sklearn** (`run_logistic_regression`):
- ElasticNet logistic regression
- Grid search: C ∈ [0.01, 0.1, 1.0], l1_ratio ∈ [0.2, 0.5, 0.8]
- Stratified 2-fold CV on training set
- Slower but more compatible

**2. GPU-based cuML** (`run_logistic_regression_gpu`):
- RAPIDS cuML for GPU acceleration
- Grid search: C ∈ [0.01, 0.03, 0.1, 1, 3, 10, 30, 100]
- L2 penalty only (elasticnet not fully supported in cuML)
- 10-50x faster on large feature sets

**3. PyTorch** (`run_linear_probe`):
- Custom linear probe with Adam optimizer
- Grid search: LR ∈ [1e-4, 3e-4, 1e-3, 3e-3], WD ∈ [0, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3]
- Early stopping: patience=20, max_epochs=400
- Returns model with predict_proba method for compatibility

**Key Outputs**:
- AUC on train/validation/test splits
- Model coefficients and standard errors
- Odds ratios with 95% confidence intervals

---

### **E. PheWAS Study (`run_phewas` method)**

**Purpose**: Test associations between each sparse feature and each outcome

#### **Statistical Analysis Per Feature-Outcome Pair**

For each feature `f` and outcome `y` in each split (train/validation/test):

**1. Activation Counts**:
```python
c_f   = count(f > 0)                    # Feature activations
c_yf  = count((y == 1) & (f > 0))       # Co-occurrences
c_y   = count(y == 1)                   # Outcome positives
p_y_f = c_yf / c_f                      # P(outcome | feature)
```

**2. Confidence Intervals**:
- Wilson score interval for p(y|f)
- Handles small sample sizes better than normal approximation

**3. Binomial Test**:
- Null hypothesis: p(y|f) = p(y) (feature doesn't change outcome probability)
- Two-sided p-value

**4. Isolated Logistic Regression** (`fit_isolated_model`):
- Fit univariate logistic regression: logit(y) = β₀ + β₁·f
- Compute:
  - Coefficient β₁ with standard error
  - Odds ratio: OR = exp(β₁)
  - 95% CI: [exp(β₁ - 1.96·SE), exp(β₁ + 1.96·SE)]
  - AUC of single feature

**5. Controlled Analysis**:
- Include confounders: logit(y) = β₀ + β₁·f + β₂·age + β₃·sex
- Compute controlled odds ratio (adjusting for demographics)

**6. Modality Attribution**:
- SAE decoder weights analyzed to attribute each feature to modalities
- Compute energy in each modality subspace: `energy[modality] = sum(W_dec[f, modality_dims]²)`
- Report % energy per modality (image, text, labs, etc.)

**7. Mutual Information**:
- Compute information gain between feature and outcome

**8. Correlation Analysis**:
- Average, max, and 95th percentile correlation with other features
- Identifies redundant features

#### **Filtering**:
```python
if c_f < 25:
    continue  # Skip features with <25 activations
```

**Critical for statistical power**: Ensures sufficient sample size for reliable statistics

#### **Hypothesis Ranking**

After PheWAS, hypotheses are ranked by:
1. **Primary**: `isolated_odds_ratio_ci_lower` (lower bound of 95% CI)
   - Conservative: only hypotheses with confident effect sizes rank high
2. **Secondary**: `isolated_odds_ratio` (point estimate)
3. **Filter**: Only features with ≥25 activations in validation set

**Output**: `stats_df` with columns:
- Feature name, outcome, split
- Counts: c_f, c_yf, c_y
- Probabilities: p_y, p(y|f), confidence intervals
- Odds ratios: isolated, controlled, with CIs
- AUCs: embeddings_auc, isolated_auc, full_model_auc
- Modality attribution: image_pct, text_pct, labs_pct
- Correlations, mutual information, etc.

---

### **F. AutoInterpretation (`Analysis/attribution/autointerp.py`)**

**Purpose**: Generate human-readable interpretations of sparse features using LLMs

#### **Workflow**

**1. Feature Selection**:
```python
def select_for_autointerp(phewas_results):
    # Top K hypotheses by isolated_odds_ratio_ci_lower
    # Or: specific feature-outcome pairs of interest
```

**2. Sample High-Activation Examples**:
- For each selected feature, find samples where feature activates strongly
- Retrieve patient data (text reports, lab values, images) for these samples

**3. Occlusion Filtering** (for text modalities):
- **Purpose**: Identify which parts of text are most important for feature activation
- **Method**: Sentence-level occlusion
  - Remove each sentence sequentially
  - Measure drop in feature activation
  - Rank sentences by importance
- **Output**: Text with importance scores per sentence
- **Benefit**: LLM sees only relevant context, improves interpretation quality

**4. Prompt LLM**:
```python
def auto_interpret(samples, interpretation_field):
    prompt = f"""
    You are analyzing a cohort of patients where Feature_{i} activates.
    Here are examples of their {interpretation_field}:

    {formatted_samples}

    Characterize this cohort in 1-2 sentences.
    """
    return llm_client.generate(prompt)
```

**5. Rate Interpretation** (`autorate_interpretation`):
- Hold out test set of high-activation examples
- Ask LLM to predict if feature activates given interpretation
- Compute accuracy: % of test samples correctly classified
- Filters out low-quality interpretations

**6. Action Suggestions** (`action_suggest`):
- For high-confidence hypotheses, generate clinical action recommendations
- Example: "Patients with [feature] have 3x higher risk of [outcome]. Consider [intervention]."

#### **LLM Clients**

**Supported**:
1. **Gemini API** (`GeminiClient`)
2. **Ollama** (`OllamaClient`) - Local inference
3. **vLLM** (`VLLMClient`) - Fast local serving

**Multimodal Support**:
- Text: Pass as string
- Images: Encode as base64, include in prompt
- Labs: Format as structured text (Lab | Value | Units | Proportion | Years Prior)

#### **Output**

- `autointerp_results.pkl`: Feature interpretations, ratings, action suggestions
- Figures: Visualization of high-activation examples with interpretations

---

## **3. Configuration System (Hydra)**

**Hierarchical Configs**:
```
config/
├── default.yaml              # Base config
├── preset/                   # Model/data presets
│   ├── 1_abdominal_ct_clinical_notes.yaml
│   ├── 2_abdominal_ct_text.yaml
│   ├── 3_abdominal_ct_image.yaml
│   └── ...
├── task/                     # Task configs (SAE, PheWAS, AutoInterp)
├── dataset/                  # Dataset configs (stanford_shc, tri_valley)
└── paths/                    # File paths
```

**Composition**:
```yaml
defaults:
  - preset: 2_abdominal_ct_text
  - task: fit_sae
  - dataset: stanford_shc
  - paths: default_paths
```

**Sweeps**: Hydra + wandb integration for hyperparameter sweeps

---

## **4. Key Technologies**

**ML/DL**:
- PyTorch 2.x + PyTorch Lightning
- cuML (RAPIDS) for GPU-accelerated statistics
- scikit-learn for CPU-based models
- SHAP for explainability

**Medical Imaging**:
- MONAI framework
- Custom 3D CNN architectures (CT-FM, Merlin)
- nibabel/nrrd for DICOM/NIfTI loading

**NLP**:
- HuggingFace transformers
- Qwen3, Longformer for text embeddings
- Gemini, Ollama, vLLM for interpretation

**Experiment Tracking**:
- Weights & Biases (wandb)
- Hydra for config management

---

## **5. Strengths of the Codebase**

1. **Sophisticated SAE Training**:
   - Matryoshka SAEs for multi-resolution features
   - Ghost gradients to prevent dead features
   - Weight normalization for stability

2. **Rigorous Statistical Analysis**:
   - Multiple hypothesis testing with proper corrections
   - Controlled analysis (confounder adjustment)
   - Conservative ranking (CI lower bound)

3. **GPU Acceleration**:
   - cuML for 10-50x speedup on large feature sets
   - PyTorch for embedding extraction and linear probes

4. **Caching & Lazy Loading**:
   - Embeddings cached to disk
   - Labels cached as pickles
   - Avoids expensive recomputation

5. **LLM-based Interpretation**:
   - Occlusion filtering for relevance
   - Rating system for quality control
   - Multimodal support (text, images, structured data)

---

## **6. Challenges for Distillation (MechSciDev)**

### **Medical-Specific Components to Remove**:
1. **Data Loading**: Hardcoded Stanford/Tri-Valley datasets, EHR preprocessing
2. **Outcome Labels**: Disease-specific (5-year prognosis, phecodes, diagnoses)
3. **Modalities**: CT images, radiology reports, clinical notes, labs
4. **Demographics**: Patient age, sex, smoking, BMI

### **Over-Engineering to Simplify**:
1. **Complex Config System**: 20+ YAML files with nested composition
2. **Caching Complexity**: Multiple cache layers with path management
3. **Hydra Sweeps**: Can use simpler config approach
4. **Wandb Integration**: Heavy reliance on wandb for logging

### **Core Components to Preserve**:
1. **SAE Training**: TopK, Matryoshka, Ghost gradients (exact copy)
2. **PheWAS Logic**: Activation counts, odds ratios, CIs, ranking (exact copy)
3. **Linear Probe**: Grid search over hyperparameters (exact copy)
4. **AutoInterp**: LLM-based interpretation framework (adapted)

---

## **7. Recommended Distillation Strategy**

### **Keep Exactly**:
- SAE architecture and training loop (fit_sae.py)
- PheWAS statistics computation (run_phewas in fit_statistical_model.py)
- Linear probe implementations (run_linear_probe, run_logistic_regression_gpu)
- Embedding extraction (extract_vectors_for_split)

### **Simplify**:
- **Data Loading**: Generic CSV-based metadata loader
  - User provides: metadata.csv with `[sample_id, modality_1_path, ..., outcome_1, ...]`
  - Load any modality type (not just medical)
- **Config**: Simple YAML with user.yaml + dev.yaml (no Hydra composition)
- **Caching**: Single cache directory with clear naming
- **Logging**: Optional wandb, print to console by default

### **Make Domain-Agnostic**:
- Replace "patient", "clinical", "medical" terminology
- Remove hardcoded modalities (image→generic embedding, text→generic text)
- Generic feature names ("Feature_0" not "Concept_0 from CT scan")

---

## **8. Pseudocode for MechSciDev**

Based on CLAUDE.md:

```python
class MS:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.dataset = Dataset(self.config)  # Generic CSV loader
        self.model = Model(self.config)  # User-provided or pretrained
        self.embedding_extractor = EmbeddingExtractor(self.config, self.model)
        self.sae = SAE(self.config)
        self.phewas = PheWAS(self.config)
        self.autointerp = AutoInterp(self.config)

    def __call__(self):
        # 1. Extract embeddings
        z = self.embedding_extractor.embed(self.dataset)
        self.save(z, 'embeddings.pkl')

        # 2. Baseline: Linear probe on embeddings
        auc_embeddings = self.linear_probe(z, self.dataset.y)
        self.save(auc_embeddings, 'auc_embeddings.pkl')

        # 3. Train SAE
        f = self.sae.fit(z)
        self.save(f, 'sparse_features.pkl')

        # 4. Evaluate: Linear probe on sparse features
        auc_sae = self.linear_probe(f, self.dataset.y)
        self.save(auc_sae, 'auc_sae.pkl')

        # 5. PheWAS: Test all feature-outcome pairs
        phewas_results = self.phewas(f, self.dataset.y)
        self.save(phewas_results, 'phewas_results.pkl')

        # 6. AutoInterp: Interpret top hypotheses
        top_hypotheses = self.select_for_autointerp(phewas_results)
        interpretations = self.autointerp(top_hypotheses)
        self.save(interpretations, 'interpretations.pkl')
```

---

## **9. Directory Structure Analysis**

### **Original Codebase Structure**
```
/dataNAS/people/rholland/MedicalHypothesisGeneration/
├── Analysis/          # Core analysis pipeline (SAE, PheWAS, statistics)
│   ├── run/
│   │   ├── fit_sae.py                    # SAE training (PRESERVE EXACTLY)
│   │   └── fit_statistical_model.py      # Linear probe + PheWAS (PRESERVE CORE LOGIC)
│   ├── attribution/
│   │   └── autointerp.py                 # LLM interpretation (ADAPT)
│   ├── figures/                          # Visualization (SIMPLIFY)
│   └── statistics/                       # Statistical utilities (PRESERVE)
├── Data/              # Data loading, preprocessing
│   ├── get_data.py                       # Main data loader (REWRITE - TOO MEDICAL-SPECIFIC)
│   ├── raw_database/                     # Medical datasets (REMOVE)
│   ├── vector_database/                  # Embedding cache (PRESERVE CONCEPT)
│   └── preprocess/                       # Preprocessing (ADAPT)
├── Model/             # Neural network models
│   ├── concept_model.py                  # Base model class (ADAPT)
│   ├── sparse_autoencoder.py             # SAE implementation (PRESERVE EXACTLY)
│   └── get_model.py                      # Model factory (SIMPLIFY)
├── Pretraining/       # Model pretraining utilities
│   └── util/
│       └── extract_model_output.py       # Embedding extraction (PRESERVE EXACTLY)
├── config/            # Hydra configuration files (SIMPLIFY - USE SIMPLE YAML)
├── utilities/         # Shared utility functions (PRESERVE USEFUL PARTS)
└── wandb/             # Weights & Biases logging (MAKE OPTIONAL)
```

---

## **10. Key Files to Extract/Adapt**

### **Files to Copy Almost Exactly**

1. **`Analysis/run/fit_sae.py`** (388 lines)
   - Class: `TrainSparseAutoencoder`
   - Function: `fit_sae()`, `filter_alive_concepts()`
   - Changes needed: Remove Hydra, simplify config access

2. **`Pretraining/util/extract_model_output.py`** (63 lines)
   - Class: `EmbeddingExtractor`, `EmbeddingExtractionCallback`
   - Function: `extract_vectors_for_split()`
   - Changes needed: Minimal - just config simplification

3. **`Model/sparse_autoencoder.py`**
   - TopK SAE implementation with Matryoshka support
   - Ghost gradient recovery
   - Decoder weight normalization
   - Changes needed: Remove medical-specific inference_map integration

### **Files to Extract Core Logic From**

4. **`Analysis/run/fit_statistical_model.py`** (1000+ lines)
   - Extract:
     - `run_linear_probe()` (lines 97-251)
     - `run_logistic_regression_gpu()` (lines 253-316)
     - `run_phewas()` (lines 817-1066)
     - `fit_isolated_model()` (lines 754-814)
     - `compute_risk_model_statistics()` (lines 424-500)
   - Changes needed: Remove medical-specific data loading, simplify

5. **`Analysis/attribution/autointerp.py`** (200+ lines)
   - Extract:
     - `auto_interpret()` - LLM prompting
     - `autorate_interpretation()` - Quality rating
     - `action_suggest()` - Action recommendations
   - Changes needed: Make domain-agnostic prompts

### **Files to Rewrite**

6. **`Data/get_data.py`** (150+ lines)
   - Too medical-specific
   - Rewrite as generic CSV loader
   - Preserve: caching concept, normalization, train/val/test splits

---

## **11. Implementation Details to Preserve**

### **SAE Training Hyperparameters**

From `fit_sae.py`:
```python
# Training
optimizer = torch.optim.Adam(lr=config.task.lr)  # Typical: 1e-3 to 3e-4
max_steps = 50000  # Or until convergence

# Loss weights
sparsity_coefficient = 0.001  # L1 penalty weight
aux_scale = 1.0  # Ghost gradient weight
lp_norm = 1  # L1 norm for sparsity

# Monitoring
dead_feature_threshold = 1e-8  # Threshold for considering feature "dead"

# TopK
top_k = [5, 10, 20, 40]  # Number of active features per sample

# Matryoshka
matryoshka = [128, 512, 2048, 8192]  # Nested dictionary sizes

# Kaux for ghost gradients
kaux = 256  # Number of auxiliary features for dead feature recovery
```

### **Linear Probe Hyperparameters**

From `fit_statistical_model.py`:
```python
# PyTorch Linear Probe
lr_grid = [1e-4, 3e-4, 1e-3, 3e-3]
wd_grid = [0.0, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3]
max_epochs = 400
patience = 20
auc_tol = 1e-4

# cuML Logistic Regression
C_grid = [0.01, 0.03, 0.1, 1, 3, 10, 30, 100]
penalty = "l2"
max_iter = 5000
```

### **PheWAS Statistics**

From `run_phewas()`:
```python
# Filtering
min_activations = 25  # Skip features with < 25 activations

# Confidence intervals
ci_method = 'wilson'  # Wilson score interval for proportions
alpha = 0.05  # 95% confidence intervals

# Hypothesis ranking
primary_metric = 'isolated_odds_ratio_ci_lower'  # Conservative ranking
secondary_metric = 'isolated_odds_ratio'

# Statistical tests
binomial_test = binomtest(..., alternative='two-sided')
```

---

## **12. Code Patterns to Preserve**

### **Pattern 1: Decoder Weight Normalization**

From `fit_sae.py` lines 161-169:
```python
# Before forward pass
if stage == "train":
    with torch.no_grad():
        W_dec = self.sae.inference_map['...'].W_dec
        W_dec.data /= torch.norm(W_dec.data, dim=1, keepdim=True)

# After backward pass
if stage == "train" and W_dec.grad is not None:
    # Project gradients to tangent space (remove parallel component)
    parallel_component = (W_dec.grad * W_dec.data).sum(dim=1)
    W_dec.grad -= torch.einsum("d,dj->dj", parallel_component, W_dec.data)
```

**Critical**: This prevents feature magnitude collapse and ensures stable training.

### **Pattern 2: Matryoshka Loss Computation**

From `fit_sae.py` lines 93-122:
```python
if matryoshka_enabled:
    nested_sizes = [128, 512, 2048, 8192]
    total_loss = 0.0

    for m_i in nested_sizes:
        # Reconstruct using first m_i features
        x_hat_prefix = sae.reconstruct_with_prefix(z_sparse, m_i)
        prefix_loss = reconstruction_loss_fn(x_hat_prefix, x_input)

        # Track alive features for this prefix
        z_prefix = z_sparse[:, :m_i]
        n_alive_prefix = (z_prefix.abs() > dead_threshold).float().mean(0).sum()

        total_loss += prefix_loss

    mse_loss = total_loss / len(nested_sizes)
```

### **Pattern 3: Ghost Gradient Recovery**

From `fit_sae.py` lines 133-138:
```python
ghost_loss = torch.tensor(0.0)
if use_ghost_grads and training:
    dead_mask = (z_sparse.mean(dim=0).abs() < dead_feature_threshold)
    if dead_mask.sum() > 0:
        _, _, x_hat_aux = sae.forward_topk_sae(x_input, K=kaux, mask=dead_mask)
        residual = x_input - x_hat
        ghost_loss = aux_scale * (residual.detach() - x_hat_aux).pow(2).mean()

loss = mse_loss + l1_loss + ghost_loss
```

### **Pattern 4: PheWAS Feature Loop**

From `fit_statistical_model.py` lines 951-1010:
```python
for feature_idx, feature_name in enumerate(feature_names):
    feature_values = X[split][:, feature_idx]
    outcome_values = y[split]

    # Activation counts
    c_f = np.sum(feature_values > 0)
    c_yf = np.sum((outcome_values > 0) & (feature_values > 0))
    c_y = np.sum(outcome_values > 0)

    # Skip if insufficient activations
    if c_f < 25:
        continue

    # Probability and confidence interval
    p_y_f = c_yf / c_f if c_f > 0 else 0
    ci_lower, ci_upper = proportion_confint(c_yf, c_f, alpha=0.05, method='wilson')

    # Binomial test
    p_value = binomtest(c_yf, c_f, c_y / len(outcome_values), alternative='two-sided').pvalue

    # Isolated logistic regression
    isolated_stats = fit_isolated_model(feature_values, outcome_values)
    # Returns: coefficient, std_error, z_score, p_value, odds_ratio, ci_lower, ci_upper, auc

    # Store all statistics
    stats_df.loc[feature_name, f'{split}_c_f'] = c_f
    stats_df.loc[feature_name, f'{split}_c_yf'] = c_yf
    stats_df.loc[feature_name, f'{split}_p(y|f)'] = p_y_f
    stats_df.loc[feature_name, f'{split}_isolated_odds_ratio'] = isolated_stats['isolated_odds_ratio']
    # ... etc
```

### **Pattern 5: Isolated Model Fitting**

From `fit_statistical_model.py` lines 754-814:
```python
def fit_isolated_model(self, feature_values, outcome_values):
    """Fit univariate logistic regression for single feature."""
    X_feature = feature_values.reshape(-1, 1)

    # Fit model
    model = cuLogisticRegression(penalty='l2', C=1.0, max_iter=5000)
    model.fit(X_feature, outcome_values)

    # Get coefficient and standard error
    coef = model.coef_[0, 0]

    # Compute Fisher information for standard error
    p = model.predict_proba(X_feature)[:, 1]
    w = p * (1 - p)
    X_with_intercept = np.column_stack([np.ones(len(X_feature)), X_feature])
    fisher_info = X_with_intercept.T @ (X_with_intercept * w[:, None])

    try:
        cov_matrix = np.linalg.inv(fisher_info)
        standard_error = np.sqrt(cov_matrix[1, 1])
    except np.linalg.LinAlgError:
        ridge = 1e-6 * np.eye(fisher_info.shape[0])
        cov_matrix = np.linalg.inv(fisher_info + ridge)
        standard_error = np.sqrt(cov_matrix[1, 1])

    # Compute statistics
    z_score = coef / standard_error
    p_value = 2 * (1 - norm.cdf(np.abs(z_score)))

    # Odds ratio and confidence intervals
    odds_ratio = np.exp(coef)
    z_critical = norm.ppf(0.975)  # 1.96 for 95% CI
    coef_ci_lower = coef - z_critical * standard_error
    coef_ci_upper = coef + z_critical * standard_error
    or_ci_lower = np.exp(coef_ci_lower)
    or_ci_upper = np.exp(coef_ci_upper)

    # AUC
    auc = roc_auc_score(outcome_values, p)

    return {
        'isolated_coefficient': coef,
        'isolated_standard_error': standard_error,
        'isolated_z_score': z_score,
        'isolated_p_value': p_value,
        'isolated_odds_ratio': odds_ratio,
        'isolated_odds_ratio_ci_lower': or_ci_lower,
        'isolated_odds_ratio_ci_upper': or_ci_upper,
        'isolated_auc': auc,
    }
```

---

## **Summary**

The Medical Hypothesis Generation codebase is a **state-of-the-art research system** for discovering interpretable scientific hypotheses from multimodal data. Its core innovations are:

1. **Matryoshka TopK SAEs** with ghost gradient recovery
2. **Rigorous PheWAS** with confounder adjustment and conservative ranking
3. **LLM-based AutoInterpretation** with occlusion filtering and quality rating

For MechSciDev, we should **preserve the exact SAE training and PheWAS logic** while **simplifying data loading, configuration, and domain-specific aspects** to create a **general-purpose hypothesis generation tool** applicable to any domain with structured data.

### **Next Steps for Implementation**

1. **Phase 1**: Set up basic project structure (config, dataset, model classes)
2. **Phase 2**: Extract and adapt SAE training (fit_sae.py)
3. **Phase 3**: Extract embedding extraction (extract_model_output.py)
4. **Phase 4**: Extract linear probe implementations
5. **Phase 5**: Extract PheWAS logic (run_phewas)
6. **Phase 6**: Adapt AutoInterp for domain-agnostic use
7. **Phase 7**: Create simple example/demo
8. **Phase 8**: Write documentation and tests
