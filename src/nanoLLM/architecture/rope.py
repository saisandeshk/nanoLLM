# src/nanoLLM/architecture/rope.py
import torch
import torch.nn as nn

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute theta values
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cosine and sine tables
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :], persistent=False)

    def forward(self, x: torch.Tensor):
        # x shape: (batch, seq_len, n_heads, head_dim)
        seq_len = x.shape[1]
        
        # Get precomputed cos and sin values
        cos = self.cos_cached[:, :seq_len, :, :]
        sin = self.sin_cached[:, :seq_len, :, :]

        # Split the tensor into two halves for rotation
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        
        # Apply rotation
        rotated_x = torch.cat((-x2, x1), dim=-1)
        x_out = (x * cos) + (rotated_x * sin)
        
        return x_out.to(x.dtype)