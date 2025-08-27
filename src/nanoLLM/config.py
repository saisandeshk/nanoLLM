from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    # Model architecture
    model_type: str = "qwen3"
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    n_kv_heads: int = 4
    sliding_window: int = 4096
    attention_bias: bool = False
    rms_norm_eps: float = 1e-6
    dropout: float = 0.1
    max_seq_len: int = 512
    vocab_size: Optional[int] = None
    
    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_kv_groups = self.n_heads // self.n_kv_heads