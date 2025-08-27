import os
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from hydra.utils import get_original_cwd

from src.nanoLLM.architecture.model import Qwen3Model
from src.nanoLLM.config import ModelConfig
from src.nanoLLM.data.dataset import load_and_cache_data, TextTokenDataset
from src.nanoLLM.data.collator import DataCollatorForLanguageModeling
from src.nanoLLM.training.trainer import Trainer
from src.nanoLLM.utils.config import set_seed
from src.nanoLLM.utils.logging import setup_logger

# Set environment variable to avoid tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    # Setup
    set_seed(cfg.train.seed)
    logger = setup_logger(cfg.train.output_dir)
    
    # Get the original working directory (project root)
    project_root = get_original_cwd()
    
    # Create a consistent output directory for models
    model_output_dir = os.path.join(project_root, "outputs", "models")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Update the config to use the consistent output directory
    cfg.train.output_dir = model_output_dir
    
    # Load data
    texts, tokenizer, tokens = load_and_cache_data(cfg.data)
    dataset = TextTokenDataset(tokens, cfg.data.max_seq_len)
    
    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    # Data loaders
    collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, 
                             collate_fn=collator, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, 
                           collate_fn=collator, num_workers=2)
    
    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Initialize model with the actual vocab size from tokenizer
    model_config = ModelConfig(**cfg.model)
    model_config.vocab_size = tokenizer.vocab_size
    model = Qwen3Model(model_config)
    
    # Train model - pass both training config and model config
    trainer = Trainer(model, train_loader, val_loader, cfg.train, model_config)
    model, final_metrics = trainer.train()
    
    print("\n🎉 TRAINING COMPLETED!")
    print(f"🏆 Final Results:")
    print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
    print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")
    print(f"💾 Model saved to: {model_output_dir}")

if __name__ == "__main__":
    main()