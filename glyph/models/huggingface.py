"""HF Transformers embeddings with explicit preprocessing and pooling."""
from pathlib import Path

import torch
from PIL import Image

from glyph.models.adapters import EmbeddingAdapter


class HFEmbeddingModel(EmbeddingAdapter):
    def __init__(self, model_id, *, device='cpu', input_type='image', pooling='auto',
                 output_key=None, embedding_method=None, normalize=False,
                 max_length=None, read_text_files=True, revision=None,
                 cache_dir=None, token=None, trust_remote_code=False,
                 model_kwargs=None, processor_kwargs=None):
        if input_type not in {'image', 'text', 'processed'}:
            raise ValueError('input_type must be image, text or processed')
        if pooling not in {'auto', 'cls', 'mean', 'last'}:
            raise ValueError('pooling must be auto, cls, mean or last')
        try:
            from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
        except ImportError as exc:
            raise ImportError("HF extraction requires pip install 'glyph-discovery[hf]'") from exc
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.input_type, self.pooling = input_type, pooling
        self.output_key, self.embedding_method = output_key, embedding_method
        self.normalize, self.max_length = normalize, max_length
        self.read_text_files = read_text_files
        common = dict(revision=revision, cache_dir=cache_dir, token=token,
                      trust_remote_code=trust_remote_code)
        model = AutoModel.from_pretrained(model_id, **{**common, **(model_kwargs or {})})
        super().__init__(model, output_key=output_key, embedding_method=embedding_method,
                         pooling=pooling, normalize=normalize, device=self.device)
        options = {**common, **(processor_kwargs or {})}
        if input_type == 'image':
            self.processor = AutoImageProcessor.from_pretrained(model_id, **options)
        elif input_type == 'text':
            self.processor = AutoTokenizer.from_pretrained(model_id, **options)
        else:
            self.processor = None
        self.embedding_dim = getattr(self.model.config, 'hidden_size', None)

    def _prepare(self, values):
        if self.input_type == 'image':
            images = []
            for value in values:
                if isinstance(value, (str, Path)):
                    with Image.open(value) as image:
                        images.append(image.convert('RGB'))
                elif isinstance(value, Image.Image):
                    images.append(value.convert('RGB'))
                else:
                    raise ValueError('Image inputs must be PIL images or local image paths')
            return self.processor(images=images, return_tensors='pt')
        if self.input_type == 'text':
            texts = []
            for value in values:
                if self.read_text_files:
                    if not isinstance(value, (str, Path)):
                        raise ValueError('Text-file inputs must be paths')
                    texts.append(Path(value).read_text())
                else:
                    if not isinstance(value, str):
                        raise ValueError('Text inputs must be strings')
                    texts.append(value)
            options = dict(padding=True, return_tensors='pt')
            if self.max_length is not None:
                options.update(truncation=True, max_length=self.max_length)
            return self.processor(texts, **options)
        if not all(isinstance(value, dict) for value in values):
            raise ValueError('processed inputs must be dictionaries of per-sample tensors')
        keys = set(values[0])
        if any(set(value) != keys for value in values):
            raise ValueError('Processed samples must have the same tensor keys')
        return {key: torch.stack([torch.as_tensor(value[key]) for value in values]) for key in keys}



class DINOv3(HFEmbeddingModel):
    def __init__(self, device='cpu', **kwargs):
        options = dict(model_id='facebook/dinov3-vits16-pretrain-lvd1689m',
                       input_type='image', pooling='cls')
        options.update(kwargs)
        super().__init__(device=device, **options)


class Qwen3Embedding06B(HFEmbeddingModel):
    def __init__(self, device='cpu', **kwargs):
        options = dict(model_id='Qwen/Qwen3-Embedding-0.6B', input_type='text',
                       pooling='last', normalize=True, max_length=8192,
                       processor_kwargs={'padding_side': 'left'})
        options.update(kwargs)
        super().__init__(device=device, **options)
