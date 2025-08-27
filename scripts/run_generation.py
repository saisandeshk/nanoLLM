# scripts/run_generation.py
import os
import sys
import torch
import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.nanoLLM.architecture import nanoLLM
from src.nanoLLM.config import ModelConfig
from src.nanoLLM.data import Tokenizer
from src.nanoLLM.inference.generate import Generator

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """
    Main function for running model inference.
    """
    project_root = get_original_cwd()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔩 Using device: {device}")

    # --- Load Tokenizer ---
    tokenizer_path = os.path.join(project_root, "data/tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found at {tokenizer_path}.")
        print("Please run 'python scripts/prepare_dataset.py' first.")
        return
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"✅ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")

    # --- Load Model ---
    # Update model config with the actual vocab size from the tokenizer
    cfg.model.vocab_size = tokenizer.vocab_size
    model_config = ModelConfig(**cfg.model)
    
    # Instantiate the model
    model = nanoLLM(model_config)
    
    # Load the checkpoint
    model_path = os.path.join(project_root, cfg.inference.model_path)
    if not os.path.exists(model_path):
        print(f"❌ Model checkpoint not found at: {model_path}")
        print("Please ensure you have a trained model or update 'configs/inference/default.yaml'.")
        return
        
    print(f"🔄 Loading model checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # The checkpoint might contain extra keys like 'optimizer_state_dict', etc.
    # We load the model's state dict, ignoring mismatched keys.
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    
    model.to(device).eval()
    print(f"✅ Model '{model_config.model_type}' loaded successfully.")

    # --- Initialize Generator ---
    generator = Generator(model, tokenizer, device)

    # --- Run Generation ---
    if cfg.inference.prompt:
        print("\n" + "="*50)
        print(f"Prompt: {cfg.inference.prompt}")
        print("="*50)
        print("Generating...\n")
        
        output = generator.generate(
            prompt=cfg.inference.prompt,
            max_new_tokens=cfg.inference.max_new_tokens,
            temperature=cfg.inference.temperature,
            top_k=cfg.inference.top_k,
        )
        print(output)
        print("\n" + "="*50)
    else:
        # Interactive mode
        print("\n" + "="*50)
        print("🤖 Starting Interactive Session")
        print("   Enter your prompt below. Type 'quit' or 'exit' to end.")
        print("="*50)
        
        while True:
            try:
                user_prompt = input("\n> ")
                if user_prompt.lower() in ["quit", "exit"]:
                    print("\n👋 Goodbye!")
                    break
                
                output = generator.generate(
                    prompt=user_prompt,
                    max_new_tokens=cfg.inference.max_new_tokens,
                    temperature=cfg.inference.temperature,
                    top_k=cfg.inference.top_k,
                )
                print(f"\n🤖: {output}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                break

if __name__ == "__main__":
    main()