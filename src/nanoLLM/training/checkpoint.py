# src/nanoLLM/training/checkpoint.py
import os
import torch

class CheckpointManager:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, model, optimizer, scheduler, step: int, val_loss: float, is_best: bool = False):
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'step': step,
            'val_loss': val_loss
        }
        
        if is_best:
            save_path = os.path.join(self.output_dir, 'best_model.pt')
            torch.save(state, save_path)
            print(f"💾 Saved best model checkpoint at step {step} to {save_path}")
        
        # Also save a final model checkpoint
        final_path = os.path.join(self.output_dir, 'final_model.pt')
        torch.save(state, final_path)

    # load method can be added here if needed for resuming training