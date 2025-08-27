# src/nanoLLM/architecture/ffn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import ModelConfig

class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu is the SiLU activation function (Swish)
        activated_x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(activated_x)