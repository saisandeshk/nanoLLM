# src/nanoLLM/architecture/block.py
import torch
import torch.nn as nn
from .attention import Attention
from .ffn import SwiGLUFeedForward
from ..config import ModelConfig

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = SwiGLUFeedForward(config)
        self.norm1 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-normalization
        h = x + self.dropout(self.attention(self.norm1(x)))
        out = h + self.dropout(self.feed_forward(self.norm2(h)))
        return out