import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(vocab_size, d_model))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = torch.nn.functional.embedding(x, self.weight)
        return self.dropout(x)