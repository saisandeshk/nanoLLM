import os
import torch
from src.nanoLLM.config import ModelConfig

def convert_checkpoint(input_path, output_path):
    """Convert a checkpoint to a simpler format with just weights and config dict"""
    print(f"Loading checkpoint from {input_path}")
    
    # Load with weights_only=False to handle all objects
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    
    # Extract model config
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        if isinstance(model_config, ModelConfig):
            # Get the config dict but exclude computed properties
            model_config_dict = model_config.__dict__.copy()
            # Remove computed properties
            computed_props = ['d_k', 'n_kv_groups']
            for prop in computed_props:
                model_config_dict.pop(prop, None)
        else:
            model_config_dict = dict(model_config)
            # Remove computed properties
            computed_props = ['d_k', 'n_kv_groups']
            for prop in computed_props:
                model_config_dict.pop(prop, None)
    elif 'config' in checkpoint:
        model_config = checkpoint['config']
        if isinstance(model_config, ModelConfig):
            model_config_dict = model_config.__dict__.copy()
            computed_props = ['d_k', 'n_kv_groups']
            for prop in computed_props:
                model_config_dict.pop(prop, None)
        else:
            model_config_dict = dict(model_config)
            computed_props = ['d_k', 'n_kv_groups']
            for prop in computed_props:
                model_config_dict.pop(prop, None)
    else:
        print("No configuration found in checkpoint")
        return
    
    # Create new checkpoint with just weights and config dict
    new_checkpoint = {
        'model_state_dict': checkpoint['model_state_dict'],
        'model_config_dict': model_config_dict,
    }
    
    # Save the new checkpoint
    torch.save(new_checkpoint, output_path)
    print(f"Converted checkpoint saved to {output_path}")

if __name__ == "__main__":
    # Convert both final and best models
    models_dir = "outputs/models"
    
    final_model_path = os.path.join(models_dir, "final_model.pt")
    if os.path.exists(final_model_path):
        convert_checkpoint(final_model_path, os.path.join(models_dir, "final_model_simple.pt"))
    
    best_model_path = os.path.join(models_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        convert_checkpoint(best_model_path, os.path.join(models_dir, "best_model_simple.pt"))