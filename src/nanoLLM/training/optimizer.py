# src/nanoLLM/training/optimizer.py
import torch
import torch.nn as nn

def create_optimizer(model: nn.Module, config):
    """
    Creates an AdamW optimizer with weight decay, excluding biases and LayerNorm/RMSNorm weights.
    """
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    
    # Create optimizer groups
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    print(f"Optimizing {len(decay_params):,} parameters with weight decay.")
    print(f"Optimizing {len(nodecay_params):,} parameters without weight decay.")
    
    optimizer = torch.optim.AdamW(
        optim_groups, lr=config.learning_rate, betas=(0.9, 0.95), fused=True
    )
    return optimizer