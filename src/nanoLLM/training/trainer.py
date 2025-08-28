# src/nanoLLM/training/trainer.py
import time
import math
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from .checkpoint import CheckpointManager

class Trainer:
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader, config, tracker):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tracker = tracker
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.scaler = GradScaler(enabled=config.use_amp)
        self.checkpoint_manager = CheckpointManager(config.output_dir)
        
        self.step = 0
        self.best_val_loss = float('inf')

    @torch.no_grad()
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader, config, tracker):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tracker = tracker
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Fixed the deprecated call
        self.scaler = GradScaler('cuda', enabled=config.use_amp)
        self.checkpoint_manager = CheckpointManager(config.output_dir)
        
        self.step = 0
        self.best_val_loss = float('inf')

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        pbar = tqdm(self.val_loader, desc="Evaluating", leave=False, total=self.config.eval_steps)
        for i, batch in enumerate(pbar):
            if i >= self.config.eval_steps:
                break
            
            # --- MODIFICATION FOR evaluate() ---
            # Instead of x, y = batch
            x = batch['input_ids'].to(self.device)
            y = batch['labels'].to(self.device)
            # --- END MODIFICATION ---
            
            with autocast(device_type='cuda', enabled=self.config.use_amp):
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            total_loss += loss.item()
        
        self.model.train()
        avg_loss = total_loss / max(1, self.config.eval_steps) # Avoid division by zero
        return {"val_loss": avg_loss, "val_perplexity": math.exp(avg_loss)}

    def train(self):
        print(f"🚀 Starting training for {self.config.max_steps} steps...")
        self.model.train()
        
        # Use an iterator to not be tied to epoch boundaries
        data_iter = iter(self.train_loader)
        pbar = tqdm(range(self.config.max_steps), desc="Training")
        
        for self.step in pbar:
            # --- MODIFICATION FOR train() ---
            try:
                batch = next(data_iter)
            except StopIteration:
                # Dataloader is exhausted, restart it
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            x = batch['input_ids'].to(self.device)
            y = batch['labels'].to(self.device)
            # --- END MODIFICATION ---
            
            # Forward and backward pass with gradient accumulation
            # The original loop for grad accum was slightly off, this is cleaner
            with autocast(device_type='cuda', enabled=self.config.use_amp):
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / self.config.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            # Step the optimizer after accumulation
            if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                
            # Logging and evaluation
            if (self.step + 1) % 10 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                log_data = {
                    "train_loss": loss.item() * self.config.gradient_accumulation_steps,
                    "learning_rate": current_lr,
                }
                self.tracker.log(log_data, step=self.step)
                pbar.set_postfix({"loss": f"{log_data['train_loss']:.4f}", "lr": f"{current_lr:.2e}"})

            if (self.step + 1) % self.config.eval_every == 0:
                eval_metrics = self.evaluate()
                self.tracker.log(eval_metrics, step=self.step)
                print(f"\nStep {self.step+1}: Val Loss: {eval_metrics['val_loss']:.4f}, Val PPL: {eval_metrics['val_perplexity']:.2f}")
                
                if eval_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = eval_metrics['val_loss']
                    self.checkpoint_manager.save(self.model, self.optimizer, self.scheduler, self.step, self.best_val_loss, is_best=True)

        print("🎉 Training finished.")
        self.checkpoint_manager.save(self.model, self.optimizer, self.scheduler, self.step, self.best_val_loss)
        self.tracker.finish()