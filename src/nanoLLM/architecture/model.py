import torch
import torch.nn as nn
import math
from .block import TransformerBlock
from .embedding import TokenEmbedding
from ..config import ModelConfig

class Qwen3Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model, config.dropout)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.output_dropout = nn.Dropout(config.dropout)
        
        # Tie weights
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Now this will work
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # Note: We don't initialize the TokenEmbedding weight here since it's already initialized in the class
    
    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        return logits