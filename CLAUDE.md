[Your primary function is to serve as an expert software development assistant operating within a Command Line Interface (CLI) environment. Adhere strictly to the following procedures and constraints:

# 1. Planning and Research 
- Initial Step: Always begin any new feature, task, or bug fix by entering planning mode. Detail the proposed steps, anticipated challenges, and potential alternative approaches (options) before executing any changes.

- Utilize Tools: Maximize the use of web search and available Multi-Context Processing (MCP) tools (e.g., file retrieval, code analysis) during the planning phase and whenever encountering complex bugs, obscure issues, or when needing to understand external APIs/libraries.

# 2. Testing and Verification
- Self-Testing Mandate: NEVER report a feature as complete until you have performed extensive internal testing. Your work is not done until the code has been thoroughly validated.

- Create all necessary test harnesses (scripts, temporary test files, etc.) and execute them. Critically evaluate the test results before considering the task finished.

- Testing must be integral to your workflow. Always ground new hypotheses or potential fixes in quick, focused tests designed to run as fast as possible.

- Efficiency: Avoid placing large timeouts on test commands unless absolutely necessary. Prioritize rapid feedback loops.

- Background Monitoring: You are authorized to run tests in the background and monitor their outputs to detect problems quickly and efficiently

# 3. File Management and Location 
- Transient Tests: For ephemeral, transient tests that are only needed for immediate debugging or validation, create and use temporary files and directories within the /tmp directory.

- Permanent Integration Tests: For integration tests, unit tests, or any other tests intended to be a permanent part of the repository, place them specifically within the tests/ directory at the project root.

- I dislike docstrings for very simple functions, and where the docstring is twice as long as the actual code, this reduces my readability; 

# 4. Git and Working Directory 
- Commit Authority: DO NOT make or suggest any git commit operations until ALL of the following conditions are met:
    - The feature/fix is fully implemented and tested to your satisfaction.
    - The human user has explicitly granted permission to commit the changes.

- Working Directory: Your normal and assumed working directory is always the project root. All file paths and commands you execute or suggest MUST be relative to this project root, even when accessing files in /tmp. Do not use commands to switch the current working directory (e.g., cd).

# Project Goal
We are developing a simple, agile open-source version of a project that aims to generate novel scientific hypotheses in a domain and data-agnostic fashion.

This should copy or draw inspiration from the original code, not reference it directly, as it is an entirely new project. The original codebase can be found at `/dataNAS/people/rholland/MedicalHypothesisGeneration/`.

## Project Structure and Functionality
### 1. Config files
Below is pseudocode for the two config files which should contain all configurable experiment parameters.
In general, arbitrary parameters such as 'top-k' or 'number of feature to autointerp per outcome' should be abstracted into the relevant hydra config file. There will be a separate hydra config subfolder for most of the following steps.

#### user.yaml
```
data_path: /../ (looks for train/ validation/)
metadata: /../
input_cols: ['image', 'labs']
outcome_cols: ['cancer_prognosis', 'artefact']
run_autointerp: True

output_results_path: intermediate_results/ # This is a relative path for intermediate outputs like autointerp_results.pkl and hypotheses.pkl. Set a reasonable default.
output_figures_tables_path: output_figures_tables/ # This is a relative path for output tables and figures. Set a reasonable default.
```

#### developer configs
Split into various hydra configs (see existing progress in MechSciDev).


### 2. Jupyter example usage
Once the tool codebase is complete, the use should be able to import it as a python package and use it in a jupyter notebook like described below in pseudocode
```
import MechSci as MS

ms = MS('path_to_config')
ms.fit()
ms.show_website()
```

### 3. Main class psuedocode
Logic to be represented in mechsci/core/mechsci.py

```python
class MS:
	def __init__(self):
		cf = ...
		self.dataset = Dataset(cf)
		self.model = Model(cf)
		self.embedding_extractor = EmbeddingExtractor(cf, self.model)
		
		self.sae = SAE(cf)
		self.phewas = PheWAS(cf)
		self.autointerp = AutoInterp(cf)

	def save(self, df)
		df.to_pickle(...)

	def select_for_autointerp(self, phewas_results):
		...
		
	def embed(self):
		return self.embedding_extractor.embed(self.dataset)

	def __call__(self)
		# 1 Extract embeddings
		z = self.embed()
		
		y = self.dataset.y
		auc_embeddings = linear_probe(z, y)
		self.save(auc_embeddings)
		
		# 2 Fit SAE
		f = self.sae.fit(z)
		self.save(f)
		
		auc_sae = log_reg(f, y)
		self.save(auc_sae)
		
		# 3 Run PheWAS
		pr = self.phewas(f, y)
		self.save(pr)
		
		# 4 Run AutoInterp
		f_to_autointerp = self.select_for_autointerp(pr)
		autointerp_results = self.autointerp(f_to_autointerp)
		self.save(autointerp_results)
```

Code should be inspired by the initial, much larger research repository. It was created for medical hypothesis discovery, but the new repository should not be specific to any domain.

However, we do not want to inherit all the over-engineered, idiosyncratic aspects of the research repository

This project is intended to be a distillation of the original version, that should
- Still use pytorch_lightning
- Use the GPU when available, but CPU when not available

#### 4. Dataset handling

- The user will supply a metadata.csv which contains outcome labels, a primary key for the sample, and a field for each data modality (f.e. image_file_path, labs_file_path).
- It also contains an optional path to textual or tabular data used for autointerp, per modality. F.e. this field will be named 'image_text' which points to a .txt file per image.
- The user will specify the path to the metadata.csv file in the relevant config. The metadata.csv will contain the following columns: sample_id, image_path, outcome, and potentially more modalities besides image in the future.
- The user will also specify an embedding_map dictionary {'image': image_model, 'modality_2': model_2} which contains a one-to-one mapping between each data modality and the relevant embedding models.
- Our code should use the correct embedding models to create and save image_embeddings.pt (which contains columns sample_id and image_embedding)
- The next steps of the code will directly use the image_embeddings.pt 

##### 4.1 Outcome labels
- Outcome columns are provided to MS() as an argument. These should be present as columns in the user provided metadata.csv file.
- The outcome columns should be numerical, or categorical. The type of each outcome should be inferred, and added to the csv as '{name}_type'. If just 0 and 1, then it's a binary classification. Continuous or float values if there are more than 0.0 and 1.0 should be treated as regression task. Assume for now that the columns are entirely numerical. 
- When running linear probes and logistic/linear regressions, the outcome type should be used to determine how to predict and evaluate the outputs.

#### 5. Embedding Models
- All models inherit an abstract class which has an "embed" method -- this accepts a batch of data of specific modality and returns the embeddings 

#### 6. Sparse AutoEncoders
- The code should fit a Sparse Autoencoder (SAE) per modality. The SAE model, accepts pre-computed embeddings and has a function called ‘embed’ which outputs the sparse activations after the top-k constraint.
- We do not currently support multi-modal SAE training, so you can assume each SAE is defined by an input modality, and a top-k parameter.
- Once training is complete, the best checkpoint is loaded and the sparse activations are extracted (defined as the output from the top-k function) using the ‘embed’ method. These are computed for each data split, and saved in the same format and folder structure as for the original embeddings, but now under a folder ‘sae_features.pkl’ as opposed to ‘embeddings’.
- Also calculate alive features (refer to the original code), and save these in a saparate pkl file 'sae_alive_features.pkl'.

#### 7. PheWAS study (for discovery)
- The PheWAS study begins by loading all sae_alive_features for a specified modality/input. Firstly, it normalizes the SAE features by subtracting the per-feature mean from the train dataset from all splits, and dividing by the per-feature standard deviation from the train set. We call this ‘sae_alive_features_normalized’.
- It then works using a nested for loop, firstly over dataset splits, secondly over outcome labels and thirdly by SAEs (i.e. by input type). The pandas row for each iteration saves these as ‘split’, ‘feature_name’ (given by the index of the feature in the SAE activations) and ‘outcome’. It also saves other experiment-specific parameters such as the modality, and embedding model name. 
- As sets of activations are defined by modality/input and top-k SAE parameter, these parameters must also be saved in each row of the data frame, even though there will be many duplications. This is a flattened data frame.
- Within each iteration, defined by a data split, sae_feature, and outcome, it computes:
	- feature statistics: prevalence of feature (proportion of how many times f > 0), 
	- outcome statistics: prevalence of outcome (proportion of how many times y > 0)
	- PheWAS outputs: odds ratio between f and y; metrics between f and y (same as for embeddings, i.e. ‘auc’, but taking only f as the input to run_logistic_regression_gpu)
- For each split, once all features are computed for each outcome, compute the rank of each feature by odds ratio on the ‘train’ set only. Call this ‘train_odds_ratio_rank’. Then, copy that ranking for all rows that share the same feature, regardless of the split.
- Results are continually added to a pandas data frame, which is saved periodically as ‘hypotheses.pkl’ in case of a crash. Create a entirely separate hypotheses.pkl for each modality/input.

#### 8. AutoInterp
- As before, make this function or class take as argument the modality/input. Start by loading the sae_alive_features for each split and top-k for that modality.
- Load the PheWAS results (hypotheses.pkl) for that modality. 
- For each outcome, find the top 5 hypotheses using train_odds_ratio_rank. These will now be interpreted. 
- Within each, find the top 5% activating samples for the associated sae feature using sae_alive_features. From this randomly sample 20 for running autointerp from the 'train' set, and different set of 10 for testing from the 'test' set.
- Using the Gemini API (with provided user-provided API key in user.yaml) form a textual prompt involving all the fit samples. Do this by populating an autointerp template defined in the relevant config file. Once run, extract the interpretation.
- Form the testing prompt by populating the test template (also defined in the relevant config file). Provide this to the API to count how many of the test samples the interpretation applies to. 
- Record the interpretation of each feature, in addition to fit samples and count, the test samples and count, the populated prompt given to the API, and the raw output, in a autointerp_results.pkl.

#### 9. Simple results visualization
- Make one or two very simple results, per modality.
- A. Create a table showing the number of hypotheses in total per SAE, and the number with a least 25 support on train.
- B. Create a second table showing the number of supported hypotheses per SAE on train with odds ratio of >1.0, >1.25 and >1.5.
- C. Produce a plot that shows, per SAE, the test rank of the top-ranked hypotheses from train.
- D. Produce a plot that shows, per SAE, the odds ratio on test split of the top-ranked hypothes from train.
- E. Produce a table per outcome that shows the top hypotheses, their ranks on all splits, odds ratios on all splits, and crucially their AutoInterp interpretation, and the AutoInterp score (i.e. % of test samples it applied to). To achieve this, you will need to join the hypotheses.pkl and autointerp_results.pkl.

#### Where to find example code
!!! Important Tip: If the new MechSciDev repository already includes this functionality, do not update that code. It has already been tested and represents a simplified version of the original research repository at: /dataNAS/people/rholland/MedicalHypothesisGeneration.

1. Load the data: /dataNAS/people/rholland/MedicalHypothesisGeneration/Data/get_data.py
    - Tip: This is the most idiosyncratic part of the code, as the tool must be applicable to any new dataset (see above)
2. Extract embeddings from a model: /dataNAS/people/rholland/MedicalHypothesisGeneration/Pretraining/util/extract_model_output.py, specifically extract_vectors_for_split
    - Tip: This can be done with lazy-loading/caching, to avoid re-computing
3. Compute the AUC of embeddings: /dataNAS/people/rholland/MedicalHypothesisGeneration/Analysis/run/fit_statistical_model.py, specifically run_linear_probe and compute_testing_dataset_aucs
    - Tip: We want to keep the grid search over weight decay and LR
4. Fit the SAEs, and extract the final activations for all datasets: /dataNAS/people/rholland/MedicalHypothesisGeneration/Analysis/run/fit_sae.py
    - Tip: For this, we do want to copy over the exact training regime and parameters
5. Compute the AUC of sparse codes: /dataNAS/people/rholland/MedicalHypothesisGeneration/Analysis/run/fit_statistical_model.py, specifically run_logistic_regression_gpu and compute_testing_dataset_aucs
    - Tip: As above, we want to keep the grid search
6. Run the PheWAS study: /dataNAS/people/rholland/MedicalHypothesisGeneration/Analysis/run/fit_statistical_model.py, specifically run_phewas
    - We do not need all of the variables. We need statistics on how often each feature activates (c_y), the outcome (p_y), and the isolated_odds_ratio
    - The 'continue' if c_f is <25 is necessary.
    - After the phewas is run, compute the rank of each hypothesis (a combination of sparse feature and outcome) for those with ≥25, within each dataset split, and grouped by SAE config: i.e. inputs, top_k and outcome (as in this new repo we do not vary matryoshka).
7. Run AutoInterp: /dataNAS/people/rholland/MedicalHypothesisGeneration/Analysis/attribution/autointerp.py, specifically autointerp_top_hypotheses_for_outcome]