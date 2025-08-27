import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import sys
print("\n".join(sys.path))

import hydra
from omegaconf import DictConfig
import torch
from hydra.utils import get_original_cwd
from src.nanoLLM.architecture.model import Qwen3Model
from src.nanoLLM.config import ModelConfig
from src.nanoLLM.data.tokenizer import Tokenizer
from src.nanoLLM.inference.generate import Generator
from src.nanoLLM.utils.config import set_seed

@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    set_seed(cfg.train.seed)
    
    # Get the original working directory (project root)
    project_root = get_original_cwd()
    
    # Determine model path - look for simple checkpoints first
    model_path = os.path.join(project_root, "outputs", "models", "final_model_simple.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(project_root, "outputs", "models", "final_model.pt")
    
    # Check if the model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        # Try the best model
        best_model_path = os.path.join(project_root, "outputs", "models", "best_model_simple.pt")
        if not os.path.exists(best_model_path):
            best_model_path = os.path.join(project_root, "outputs", "models", "best_model.pt")
        
        if os.path.exists(best_model_path):
            print(f"Found best model at {best_model_path}, using that instead")
            model_path = best_model_path
        else:
            print("No trained model found. Please run training first.")
            return
    
    print(f"Loading model from {model_path}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Extract the model config
    if 'model_config_dict' in checkpoint:
        config = ModelConfig(**checkpoint['model_config_dict'])
    elif 'model_config' in checkpoint:
        model_config_data = checkpoint['model_config']
        if isinstance(model_config_data, dict):
            config = ModelConfig(**model_config_data)
        else:
            print(f"Unsupported config type: {type(model_config_data)}")
            return
    else:
        print("No configuration found in checkpoint")
        return
    
    # Create model
    model = Qwen3Model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load tokenizer
    tokenizer_path = os.path.join(project_root, "data", "tokenizer", "tokenizer.json")
    tokenizer = Tokenizer(tokenizer_path)
    
    # Create generator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(model, tokenizer, device)
    
    # Run demo
    generator.demo()
    
    # Interactive mode
    response = input("\n🤖 Would you like to try interactive inference? (y/n): ")
    if response.lower() in ['y', 'yes']:
        generator.interactive()

if __name__ == "__main__":
    main()