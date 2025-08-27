# src/nanoLLM/config.py

from pydantic import BaseModel

class ModelConfig(BaseModel):
    model_type: str = "nanoLLM"
    d_model: int = 1536
    n_layers: int = 22
    n_heads: int = 12
    d_ff: int = 4096
    n_kv_heads: int = 4
    attention_bias: bool = False
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    max_seq_len: int = 512
    vocab_size: int = 32000

    @property
    def d_k(self) -> int:
        """Dimension of the key/query vectors."""
        return self.d_model // self.n_heads

    @property
    def n_kv_groups(self) -> int:
        """Number of query heads per key/value head."""
        return self.n_heads // self.n_kv_heads