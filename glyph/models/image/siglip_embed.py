import torch
from pathlib import Path
from typing import List, Union

from glyph.models.base import BaseEmbeddingModel

class SiglipEmbed(BaseEmbeddingModel):
    def __init__(self, device="cpu"):
        super().__init__()

    def embed(self, data: Union[str, Path]) -> torch.Tensor:
        if isinstance(data, (str, Path)):
            torch_tensor = torch.load(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        return torch_tensor
    
    def embed_batch(self, data_list: List[Union[str, Path]]) -> torch.Tensor:
        torch_tensors = [self.embed(data) for data in data_list]
        return torch.stack(torch_tensors)
    
    def get_embedding_dim(self) -> int:
        return 2056