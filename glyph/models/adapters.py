"""Adapt a user model and preprocessing function to Glyph's embedding interface."""
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from glyph.models.base import BaseEmbeddingModel


class EmbeddingAdapter(BaseEmbeddingModel):
    def __init__(self, model, preprocess=None, *, output_key=None,
                 embedding_method=None, pooling='auto', normalize=False,
                 device='cpu', embedding_dim=None):
        if pooling not in {'auto', 'cls', 'mean', 'last'}:
            raise ValueError('pooling must be auto, cls, mean or last')
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.model = model
        if hasattr(model, 'eval'):
            self.model = self.model.eval()
        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)
        self.preprocess = preprocess
        self.output_key, self.embedding_method = output_key, embedding_method
        self.pooling, self.normalize = pooling, normalize
        self.embedding_dim = embedding_dim or getattr(getattr(model, 'config', None), 'hidden_size', None)
        self._observed_dim = None

    def _prepare(self, values):
        if self.preprocess is not None:
            return self.preprocess(values)
        if all(isinstance(value, Mapping) for value in values):
            keys = set(values[0])
            if any(set(value) != keys for value in values):
                raise ValueError('Processed samples must have the same tensor keys')
            return {key: torch.stack([torch.as_tensor(value[key]) for value in values]) for key in keys}
        return torch.stack([torch.as_tensor(value) for value in values])

    def _move(self, value):
        if torch.is_tensor(value):
            return value.to(self.device)
        if isinstance(value, Mapping):
            return {key: self._move(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(self._move(item) for item in value)
        return value

    @staticmethod
    def _field(output, key):
        if isinstance(output, (tuple, list)) and isinstance(key, int):
            return output[key] if -len(output) <= key < len(output) else None
        return output.get(key) if hasattr(output, 'get') else getattr(output, key, None)

    def _pool(self, output, inputs):
        if self.output_key is not None:
            value = self._field(output, self.output_key)
        elif torch.is_tensor(output):
            value = output
        elif self.pooling == 'auto':
            value = None
            for key in ['sentence_embedding', 'text_embeds', 'image_embeds', 'pooler_output']:
                value = self._field(output, key)
                if value is not None:
                    break
            if value is None:
                value = self._field(output, 'last_hidden_state')
        else:
            value = self._field(output, 'last_hidden_state')
        if not torch.is_tensor(value):
            fields = list(output.keys()) if hasattr(output, 'keys') else list(vars(output)) if hasattr(output, '__dict__') else []
            raise ValueError(f'Model output is not an embedding tensor (fields: {fields}); set output_key or embedding_method')
        if value.ndim == 2:
            return value
        if value.ndim != 3:
            raise ValueError('Expected batch-by-feature or batch-by-token-by-feature output; select a suitable embedding output')
        if self.pooling == 'auto':
            raise ValueError('Model returns token/patch features; choose pooling=cls, mean or last according to its model card')
        if self.pooling == 'cls':
            return value[:, 0]
        mask = inputs.get('attention_mask') if isinstance(inputs, Mapping) else None
        if mask is None:
            if self.pooling == 'last':
                return value[:, -1]
            return value.mean(dim=1)
        if mask.shape != value.shape[:2] or not bool(mask.any(dim=1).all()):
            raise ValueError('Attention mask must match output tokens and contain valid tokens in every row')
        if self.pooling == 'mean':
            weights = mask.unsqueeze(-1).to(value.dtype)
            return (value * weights).sum(dim=1) / weights.sum(dim=1)
        positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        last = positions.masked_fill(~mask.bool(), -1).max(dim=1).values
        return value[torch.arange(len(value), device=value.device), last]

    @torch.inference_mode()
    def embed_batch(self, data_list):
        if not data_list:
            return torch.empty((0, self.get_embedding_dim()))
        inputs = self._move(self._prepare(data_list))
        forward = getattr(self.model, self.embedding_method) if self.embedding_method else self.model
        output = forward(**inputs) if isinstance(inputs, Mapping) else forward(inputs)
        result = self._pool(output, inputs).float()
        if result.shape[0] != len(data_list) or result.shape[1] == 0 or not torch.isfinite(result).all():
            raise ValueError('Embedding output must contain one finite feature vector per input')
        if self._observed_dim is not None and result.shape[1] != self._observed_dim:
            raise ValueError('Embedding dimension changed between batches')
        self._observed_dim = result.shape[1]
        if self.normalize:
            result = F.normalize(result, dim=-1)
        self.embedding_dim = result.shape[1]
        return result.detach().cpu()

    def embed(self, data):
        return self.embed_batch([data])[0]

    def get_embedding_dim(self):
        if self.embedding_dim is None:
            raise ValueError('Embedding dimension is unknown until the first extraction')
        return self.embedding_dim
