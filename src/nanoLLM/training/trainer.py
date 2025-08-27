# src/nanoLLM/training/trainer.py
import time
import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
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
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        pbar = tqdm(self.val_loader, desc="Evaluating", leave=False, total=self.config.eval_steps)
        for i, batch in enumerate(pbar):
            if i >= self.config.eval_steps:
                break
            
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            
            with autocast(enabled=self.config.use_amp, device_type='cuda'):
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            total_loss += loss.item()
        
        self.model.train()
        avg_loss = total_loss / self.config.eval_steps
        return {"val_loss": avg_loss, "val_perplexity": math.exp(avg_loss)}

    def train(self):
        print(f"🚀 Starting training for {self.config.max_steps} steps...")
        self.model.train()
        
        pbar = tqdm(total=self.config.max_steps, desc="Training")
        
        while self.step < self.config.max_steps:
            for batch in self.train_loader:
                if self.step >= self.config.max_steps:
                    break
                
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward and backward pass
                for i in range(self.config.gradient_accumulation_steps):
                    with autocast(enabled=self.config.use_amp, device_type='cuda'):
                        logits = self.model(x)
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                        loss = loss / self.config.gradient_accumulation_steps
                    
                    self.scaler.scale(loss).backward()

                # Optimizer step
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                
                # Logging and evaluation
                if self.step % 10 == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    log_data = {
                        "train_loss": loss.item() * self.config.gradient_accumulation_steps,
                        "learning_rate": current_lr,
                    }
                    self.tracker.log(log_data, step=self.step)
                    pbar.set_postfix({"loss": f"{log_data['train_loss']:.4f}", "lr": f"{current_lr:.2e}"})

                if self.step > 0 and self.step % self.config.eval_every == 0:
                    eval_metrics = self.evaluate()
                    self.tracker.log(eval_metrics, step=self.step)
                    print(f"\nStep {self.step}: Val Loss: {eval_metrics['val_loss']:.4f}, Val PPL: {eval_metrics['val_perplexity']:.2f}")
                    
                    if eval_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = eval_metrics['val_loss']
                        self.checkpoint_manager.save(self.model, self.optimizer, self.scheduler, self.step, self.best_val_loss, is_best=True)
                
                self.step += 1
                pbar.update(1)

        pbar.close()
        print("🎉 Training finished.")
        # Save final model
        self.checkpoint_manager.save(self.model, self.optimizer, self.scheduler, self.step, self.best_val_loss)
        self.tracker.finish()