"""Encoder adapters ported from HypothesisExtractor/models (October 2026).

Heavy dependencies and HF downloads are lazy. Weight files are never packaged.
"""
import re
from pathlib import Path
import torch
import torch.nn.functional as F
from glyph.models.base import BaseEmbeddingModel


class BatchEncoder(BaseEmbeddingModel):
    def embed(self, data):
        return self.embed_batch([data])[0]

    def get_embedding_dim(self):
        return self.embedding_dim


class MerlinEncoder(BatchEncoder):
    embedding_dim = 2048

    def __init__(self, device="cpu"):
        try:
            from merlin import Merlin
            from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
                Spacingd, ScaleIntensityRanged, SpatialPadd, CenterSpatialCropd, ToTensord)
        except ImportError as exc:
            raise ImportError("CT extraction requires pip install 'glyph-discovery[ct]'") from exc
        self.device = device
        self.model = Merlin(ImageEmbedding=True).eval().to(device)
        self.transform = Compose([
            LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode="bilinear"),
            ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0., b_max=1., clip=True),
            SpatialPadd(keys=["image"], spatial_size=[224, 224, 160]),
            CenterSpatialCropd(keys=["image"], roi_size=[224, 224, 160]), ToTensord(keys=["image"]),
        ])

    def embed_batch(self, data_list):
        images = []
        for path in data_list:
            image = self.transform({"image": str(path)})["image"]
            images.append(image.as_tensor() if hasattr(image, "as_tensor") else image)
        images = torch.stack(images).to(self.device)
        with torch.inference_mode():
            output = self.model(images)
            # merlin-vlm versions return [1,B,D] or [B,1,D]. Preserve the batch axis.
            if output.ndim == 3:
                output = output.squeeze(0) if output.shape[0] == 1 else output.squeeze(1)
            if output.shape != (len(data_list), self.embedding_dim):
                raise ValueError(f"Unexpected Merlin embedding shape: {tuple(output.shape)}")
            return output.detach().cpu()


class Qwen3Encoder(BatchEncoder):
    embedding_dim = 4096

    def __init__(self, device="cpu", cache_dir=None, revision=None, token=None):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Report extraction requires pip install 'glyph-discovery[text]'") from exc
        self.device = device
        kwargs = dict(cache_dir=cache_dir, revision=revision, token=token)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-8B", padding_side="left", **kwargs)
        self.model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-8B", **kwargs).eval().to(device)

    def embed_batch(self, data_list):
        texts = []
        for path in data_list:
            text = Path(path).read_text()
            match = re.search(r"findings\s*:?\s*(.*?)(?=\bimpression\b|$)", text, re.I | re.S)
            texts.append((match.group(1).strip() if match else text.strip()) + self.tokenizer.eos_token)
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=8192,
                               return_tensors="pt").to(self.device)
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"]
            pooled = hidden[:, -1] if bool((mask[:, -1].sum() == len(mask)).item()) else hidden[
                torch.arange(len(mask), device=hidden.device), mask.sum(dim=1) - 1]
            return F.normalize(pooled, p=2, dim=1).detach().cpu()
