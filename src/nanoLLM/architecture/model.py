# src/nanoLLM/architecture/model.py
import torch
import torch.nn as nn
import math
from .block import TransformerBlock
from .embedding import TokenEmbedding
from ..config import ModelConfig

class nanoLLM(nn.Module): # Renamed to nanoLLM for clarity
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = TokenEmbedding(config)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie the weights of the token embedding and the final linear layer
        self.token_embedding.embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is (batch, seq_len)
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        
        for block in self.transformer_blocks:
            x = block(x)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits