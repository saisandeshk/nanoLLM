# src/nanoLLM/utils/tracking.py
import os
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False

class WandbTracker:
    def __init__(self, project_name: str, config: dict):
        if not _WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Please run 'pip install wandb'")
        
        wandb.init(
            project=project_name,
            config=config,
        )

    def log(self, data: dict, step: int):
        wandb.log(data, step=step)

    def finish(self):
        wandb.finish()

class NoOpTracker:
    """A tracker that does nothing, for local runs without logging."""
    def __init__(self, *args, **kwargs):
        print("W&B not available or enabled. Using NoOpTracker.")
        pass

    def log(self, data: dict, step: int):
        pass

    def finish(self):
        pass

def get_tracker(use_wandb: bool, project_name: str, config: dict):
    if use_wandb and _WANDB_AVAILABLE:
        return WandbTracker(project_name, config)
    return NoOpTracker()