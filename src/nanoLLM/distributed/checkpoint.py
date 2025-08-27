import os
import torch
import torch.distributed as dist
from .utils import is_main_process

def save_checkpoint(model, optimizer, scheduler, config, epoch, global_step, output_dir, is_best=False):
    """Save model checkpoint with distributed support"""
    if is_main_process():
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config,
            'epoch': epoch,
            'global_step': global_step
        }
        
        if is_best:
            checkpoint_path = os.path.join(output_dir, 'best_model.pt')
        else:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint with distributed support"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('global_step', 0)