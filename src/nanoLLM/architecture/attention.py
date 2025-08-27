# src/nanoLLM/architecture/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import Rotary
from ..config import ModelConfig

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats the key and value heads to match the number of query heads.
    Input: (batch, n_kv_heads, seq_len, head_dim) -> Output: (batch, n_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_groups = config.n_kv_groups
        self.d_k = config.d_k
        
        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.d_k, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.d_k, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.d_k, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.d_k, config.d_model, bias=config.attention_bias)
        
        self.rotary = Rotary(self.d_k, config.max_seq_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)

        # Apply RoPE
        q = self.rotary(q)
        k = self.rotary(k)

        # Transpose for attention calculation
        q = q.transpose(1, 2) # (batch, n_heads, seq_len, d_k)
        k = k.transpose(1, 2) # (batch, n_kv_heads, seq_len, d_k)
        v = v.transpose(1, 2) # (batch, n_kv_heads, seq_len, d_k)
        
        # Repeat KV heads for GQA
        k = repeat_kv(k, self.n_kv_groups)
        v = repeat_kv(v, self.n_kv_groups)
        
        # Use PyTorch's optimized attention implementation
        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)