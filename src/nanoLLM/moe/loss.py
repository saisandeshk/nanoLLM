import torch
import torch.nn as nn

class MoELoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        
    def forward(self, router_logits):
        # Calculate load balancing loss
        router_probs = torch.softmax(router_logits, dim=-1)
        expert_mask = torch.zeros(router_probs.size(0), self.num_experts, device=router_logits.device)
        
        # Create one-hot encoding of selected experts
        for i in range(self.top_k):
            top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)[1]
            expert_mask.scatter_(1, top_k_indices[:, i:i+1], 1)
        
        # Calculate auxiliary loss
        expert_counts = expert_mask.sum(dim=0)
        expert_probs = router_probs.mean(dim=0)
        
        aux_loss = (expert_counts * expert_probs).sum() * self.num_experts
        return aux_loss