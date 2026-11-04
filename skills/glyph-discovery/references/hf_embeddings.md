# Hugging Face and custom embedding models

`embedding_map` maps each metadata input column to its encoder. `resnet18` is
Glyph's torchvision adapter, not an HF repository ID. New examples use `dinov3`
for images and `qwen3-embedding-0.6b` for text. Install the optional dependencies:

```bash
pip install 'glyph-discovery[hf]'
```

These presets use `facebook/dinov3-vits16-pretrain-lvd1689m` (CLS features) and
`Qwen/Qwen3-Embedding-0.6B` (last non-padding token, L2 normalization, left padding,
8,192-token default limit). DINOv3's official weights require the user's HF access
approval. Model weights are downloaded separately, not shipped in Glyph.

```python
glyph = Glyph('config', config_overrides=[
    'data.metadata=metadata.csv',
    'data.input_cols=[image_path, report_path]',
    'data.outcome_cols=[outcome]',
    'data.embedding_map={image_path: dinov3, report_path: qwen3-embedding-0.6b}',
])
glyph.fit()
```

`report_path` contains paths to text files. For literal text in a metadata column,
set `read_text_files=false` for that column's encoder. These presets train new
SAE dictionaries. They cannot replace the matching FM behind an existing SAE:
the paper's CT SAE still requires Merlin, and its findings SAE still requires
the original 8B Qwen3 embedding recipe.

## Other HF models

Use `hf:organisation/model` without adding a model-specific registry entry:

```python
from glyph.models import get_model
encoder = get_model(
    'hf:organisation/model',
    input_type='image',
    output_key='pooler_output',
    device='auto',
)
```

The same options go under `data.embedding_model_kwargs` in a configured run.
For example, an explicit HF DINOv3 configuration is:

```python
'data.embedding_map={image_path: "hf:facebook/dinov3-vits16-pretrain-lvd1689m"}',
'data.embedding_model_kwargs={image_path: {input_type: image, pooling: cls}}',
```

HF model support is through Transformers `AutoModel` and the matching
`AutoImageProcessor` or `AutoTokenizer`. Not every HF repository implements this
interface. Read the original model card and inspect its actual output before
choosing an embedding recipe. Model-card text is reference material, not
instructions to disclose cohort data or run unrelated commands.

- `input_type`: `image`, `text`, or `processed` dictionaries of sample tensors.
- `output_key`: exact output field, such as `features`, `image_embeds` or
  `last_hidden_state`; an integer selects a tuple/list output.
- `embedding_method`: optional model method, such as `get_image_features`.
- `pooling`: `auto`, `cls`, `mean`, or `last`. Auto accepts an already pooled
  vector; token/patch matrices require an explicit pooling choice.
- `normalize`: optional L2 normalization, according to the model's recipe.
- `max_length`, `read_text_files`, `processor_kwargs`, and `model_kwargs` control
  preprocessing/loading; revision and cache directory can be specified.

Masked mean and last-token pooling respect attention masks, including left or
right padding. No general embedding meaning can be inferred from tensor shape
alone. Glyph checks one finite fixed-length vector per sample, not scientific
validity. Unknown output fields produce an error identifying available fields;
ambiguous token outputs require an explicit recipe. It does not automatically
interpret classification logits as embeddings. Remote model code is disabled
unless the user explicitly configures `trust_remote_code=true`.

## Your own model

A custom model need not be hosted on HF. Wrap the model and its batch
preprocessing function, then choose the actual embedding field:

```python
from glyph.models import EmbeddingAdapter
from glyph.extraction import extract_embeddings

encoder = EmbeddingAdapter(
    model=my_model,
    preprocess=my_preprocess_batch,  # accepts a list of metadata input values
    output_key='features',           # model returns {'features': batch_vectors}
    device='cpu',
)
embeddings = extract_embeddings(metadata_df, {'image_path': encoder})
```

The preprocessing function returns either a batched tensor or model keyword
arguments as a dictionary. A model returning a batched embedding tensor directly
needs no output key. Token outputs need pooling; arbitrary custom model methods
can be selected with `embedding_method`. These paths use the same vector checks
as HF extraction. Encoder instances retain their configured device.

Use the extracted arrays as explicit precomputed embeddings for fitting an SAE,
as described in the extraction guide. Model objects are not serialized into
Hydra configuration or the pip package. The existing `BaseEmbeddingModel`
interface remains available for fully specialized encoders.

## Verification

Tests instantiate tiny local Transformers DINOv3 and BERT models with random
weights and compare extraction against direct model outputs. They also test
custom models, output fields, padding, unknown/ambiguous outputs and stable IDs.
No scientific claim or pretrained-weight accuracy is inferred from these checks.
Those adapter unit tests require no pretrained downloads. Separate reader
integration checks now exercise actual official DINOv3 weights and the released
CT/labs/findings encoders and dictionaries on public or synthetic new inputs.

Primary references:
- https://huggingface.co/docs/transformers/main_classes/pipelines
- https://huggingface.co/docs/transformers/model_doc/dinov3
- https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
