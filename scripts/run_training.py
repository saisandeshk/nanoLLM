# scripts/run_training.py
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd
import torch
from torch.utils.data import DataLoader

# Add src to python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.nanoLLM.architecture import nanoLLM
from src.nanoLLM.config import ModelConfig
from src.nanoLLM.data import Tokenizer, MemmapDataset, DataCollator
from src.nanoLLM.training.trainer import Trainer
from src.nanoLLM.training.optimizer import create_optimizer
from src.nanoLLM.training.lr_scheduler import create_scheduler
from src.nanoLLM.utils.config import set_seed
from src.nanoLLM.utils.tracking import get_tracker

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Setup
    set_seed(cfg.train.seed)
    project_root = get_original_cwd()
    
    # Update output directory to be absolute
    cfg.train.output_dir = os.path.join(project_root, cfg.train.output_dir)
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    
    # Load Tokenizer
    tokenizer_path = os.path.join(project_root, "data/tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}.")
        print("Please run 'python scripts/prepare_dataset.py' first.")
        return
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Data Loaders
    train_data_path = os.path.join(project_root, "data/train.bin")
    val_data_path = os.path.join(project_root, "data/validation.bin")
    
    train_dataset = MemmapDataset(train_data_path, cfg.data.max_seq_len)
    val_dataset = MemmapDataset(val_data_path, cfg.data.max_seq_len)
    
    collator = DataCollator()
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, collate_fn=collator, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, collate_fn=collator, num_workers=4)
    
    print(f"📊 Datasets loaded: {len(train_dataset):,} train samples, {len(val_dataset):,} validation samples")
    
    # Initialize Model
    # Important: Update vocab_size from the trained tokenizer
    cfg.model.vocab_size = tokenizer.vocab_size
    model_config = ModelConfig(**cfg.model)
    model = nanoLLM(model_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model initialized: {model_config.model_type} with {total_params:,} parameters.")

    # Setup Optimizer and Scheduler
    optimizer = create_optimizer(model, cfg.train)
    scheduler = create_scheduler(optimizer, cfg.train)

    # Setup Tracker (e.g., Weights & Biases)
    # To enable, add `wandb: true` to your training config
    use_wandb = cfg.train.get("wandb", False)
    tracker = get_tracker(use_wandb, project_name="nanoLLM", config=OmegaConf.to_container(cfg, resolve=True))
    
    # Initialize Trainer and start training
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg.train,
        tracker=tracker,
    )
    trainer.train()

if __name__ == "__main__":
    main()