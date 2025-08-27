# src/nanoLLM/training/lr_scheduler.py
import torch
import math

def create_scheduler(optimizer, config):
    """Creates a learning rate scheduler with warmup and cosine decay."""
    # Add warmup_steps to your train config (e.g., 2000)
    warmup_steps = getattr(config, 'warmup_steps', 2000)
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        progress = float(current_step - warmup_steps) / float(max(1, config.max_steps - warmup_steps))
        # Cosine decay to 10% of the original learning rate
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)