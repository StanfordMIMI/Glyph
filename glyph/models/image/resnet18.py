import torch
import torchvision.transforms as T
import torchvision.models as models
from typing import List, Union
from PIL import Image
from pathlib import Path

from glyph.models.base import BaseEmbeddingModel


class ResNet18(BaseEmbeddingModel):
    """
    ResNet18 image embedding model.

    Extracts 512-dimensional feature vectors from images using a pretrained
    ResNet18 model (without the classification head).
    """

    def __init__(self, device="cpu"):
        """
        Initialize ResNet18 model.

        Args:
            device: Device to run model on ('cpu', 'cuda', or 'cuda:0', etc.)
        """
        # Small + fast model
        m = models.resnet18(weights="DEFAULT")
        # remove classification head → feature extractor
        self.model = torch.nn.Sequential(*(list(m.children())[:-1]))
        self.model.eval().to(device)
        self.device = device

        self.transform = T.Compose([
            T.Resize(64),
            T.CenterCrop(64),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    def preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Preprocess a single PIL image.

        Args:
            pil_image: PIL Image object

        Returns:
            Preprocessed tensor of shape (1, 3, 64, 64)
        """
        x = self.transform(pil_image.convert("RGB"))
        return x.unsqueeze(0).to(self.device)

    def preprocess_batch(self, pil_images: List[Image.Image]) -> torch.Tensor:
        """
        Preprocess a batch of PIL images.

        Args:
            pil_images: List of PIL Image objects

        Returns:
            Batched tensor of shape (batch_size, 3, 64, 64)
        """
        tensors = [self.transform(img.convert("RGB")) for img in pil_images]
        return torch.stack(tensors).to(self.device)

    @torch.no_grad()
    def embed(self, data: Union[Image.Image, str, Path]) -> torch.Tensor:
        """
        Extract embedding from a single image.

        Args:
            data: PIL Image, or path to image file (str or Path)

        Returns:
            Embedding tensor of shape (512,)
        """
        # Load image if path is provided
        if isinstance(data, (str, Path)):
            pil_image = Image.open(data)
        else:
            pil_image = data

        x = self.preprocess(pil_image)
        feat = self.model(x)     # (1, 512, 1, 1)
        return feat.flatten()    # → (512,)

    @torch.no_grad()
    def embed_batch(self, data_list: List[Union[Image.Image, str, Path]]) -> torch.Tensor:
        """
        Extract embeddings from a batch of images.

        Args:
            data_list: List of PIL Images or paths to image files

        Returns:
            Batch of embeddings of shape (batch_size, 512)
        """
        # Load images if paths are provided
        pil_images = []
        for data in data_list:
            if isinstance(data, (str, Path)):
                pil_images.append(Image.open(data))
            else:
                pil_images.append(data)

        x = self.preprocess_batch(pil_images)
        feat = self.model(x)     # (batch_size, 512, 1, 1)
        return feat.squeeze(-1).squeeze(-1)  # → (batch_size, 512)

    def get_embedding_dim(self) -> int:
        """
        Return embedding dimension.

        Returns:
            512 (ResNet18 feature dimension)
        """
        return 512
