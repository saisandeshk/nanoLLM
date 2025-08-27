import torch
import math

def get_scheduler(optimizer, config):
    """Create a learning rate scheduler with warmup and cosine decay"""
    warmup_steps = config.max_steps // 20
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)