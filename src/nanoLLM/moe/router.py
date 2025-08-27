import torch
import torch.nn as nn

class Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.emb_dim
        self.num_experts = config.num_experts
        
        self.gate = nn.Linear(self.hidden_size, self.num_experts)
        
    def forward(self, x):
        return self.gate(x)