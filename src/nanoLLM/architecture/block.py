import torch
import torch.nn as nn
from .attention import GroupedQueryAttention
from .ffn import FeedForward

class TransformerBlockOld(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-6)
        
    def forward(self, x, mask, cos, sin):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin)
        x = x + shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x
    
import torch
import torch.nn as nn
from .attention import Qwen3Attention
from .ffn import SwiGLUFeedForward
from ..config import ModelConfig

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = Qwen3Attention(config)
        self.feed_forward = SwiGLUFeedForward(config.d_model, config.d_ff, config.dropout)
        self.norm1 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x