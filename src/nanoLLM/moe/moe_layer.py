import torch
import torch.nn as nn
import torch.nn.functional as F
from .router import Router

class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.expert_capacity = config.expert_capacity
        
        # Create experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.emb_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.emb_dim)
            ) for _ in range(self.num_experts)
        ])
        
        # Router
        self.router = Router(config)
        
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        # Reshape for processing
        x_flat = x.view(-1, hidden_size)
        
        # Get router logits
        router_logits = self.router(x_flat)
        
        # Get top-k experts
        top_k_weights, top_k_indices = torch.topk(
            F.softmax(router_logits, dim=-1), 
            self.top_k, 
            dim=-1
        )
        
        # Normalize weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Initialize expert output
        expert_output = torch.zeros_like(x_flat)
        
        # Process tokens through experts
        for i in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (top_k_indices == i).any(dim=-1)
            if not expert_mask.any():
                continue
                
            # Get tokens for this expert
            expert_tokens = x_flat[expert_mask]
            
            # Get weights for this expert
            expert_weights = top_k_weights[expert_mask]
            expert_indices = top_k_indices[expert_mask]
            
            # Find positions where this expert is in top-k
            expert_positions = (expert_indices == i)
            
            # Get weights for this expert
            weights = expert_weights[expert_positions]
            
            # Process tokens through expert
            expert_out = self.experts[i](expert_tokens)
            
            # Apply weights and accumulate
            expert_output[expert_mask] += weights.unsqueeze(-1) * expert_out
            
        # Reshape back
        return expert_output.view(batch_size, seq_len, hidden_size)