# src/nanoLLM/architecture/embedding.py
import torch
import torch.nn as nn
from ..config import ModelConfig

class TokenEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # In a real model, this would be nn.Embedding, but a linear layer is equivalent
        # and makes weight tying more explicit.
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)