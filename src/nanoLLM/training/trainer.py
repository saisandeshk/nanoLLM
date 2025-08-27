import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from ..utils.config import set_seed
from .optimizer import Muon, setup_muon_optimizer
from .lr_scheduler import get_scheduler

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, model_config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config  # Training config
        self.model_config = model_config  # Model config (contains vocab_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Setup optimizers
        self.optimizers = setup_muon_optimizer(self.model, config)
        
        # Setup schedulers
        self.schedulers = []
        for optimizer in self.optimizers:
            scheduler = get_scheduler(optimizer, config)
            self.schedulers.append(scheduler)
        
        # Setup scaler for mixed precision
        self.scaler = GradScaler('cuda') if config.use_amp else None
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.step = 0
        self.best_val_loss = float('inf')
    
    def evaluate(self):
        """Evaluate model performance"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        total_correct = 0
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= self.config.eval_steps:
                    break
                
                x = batch['input_ids'].to(self.device)
                y = batch['labels'].to(self.device)
                
                with autocast(enabled=self.config.use_amp, device_type='cuda'):
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.view(-1, self.model_config.vocab_size), y.view(-1))
                
                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()
                predictions = logits.argmax(dim=-1)
                total_correct += (predictions == y).sum().item()
        
        avg_loss = total_loss / total_tokens
        accuracy = total_correct / total_tokens
        perplexity = math.exp(min(avg_loss, 20))
        
        self.model.train()
        return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'model_config': self.model_config,
            'step': self.step,
            'optimizers': [opt.state_dict() for opt in self.optimizers],
            'schedulers': [sched.state_dict() for sched in self.schedulers]
        }
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.config.output_dir, 'best_model.pt'))
            print(f"💾 Saved best model with val_loss: {self.best_val_loss:.4f}")
        else:
            torch.save(checkpoint, os.path.join(self.config.output_dir, 'final_model.pt'))
            print(f"💾 Saved final model to final_model.pt")
    
    def train(self):
        """Train the model"""
        print(f"\n🚀 Training model with Muon optimizer")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  📊 Total parameters: {total_params:,}")
        
        self.model.train()
        start_time = time.time()
        pbar = tqdm(total=self.config.max_steps, desc="Training")
        
        while self.step < self.config.max_steps:
            for batch_idx, batch in enumerate(self.train_loader):
                if self.step >= self.config.max_steps:
                    break
                
                x = batch['input_ids'].to(self.device)
                y = batch['labels'].to(self.device)
                
                # Forward pass with gradient accumulation
                if self.config.use_amp:
                    with autocast(device_type='cuda'):
                        logits = self.model(x)
                        loss = F.cross_entropy(logits.view(-1, self.model_config.vocab_size), y.view(-1))
                        loss = loss / self.config.gradient_accumulation_steps
                    self.scaler.scale(loss).backward()
                else:
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.view(-1, self.model_config.vocab_size), y.view(-1))
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()
                
                # Optimizer step after accumulation
                if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.use_amp:
                        for optimizer in self.optimizers:
                            self.scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                        for optimizer in self.optimizers:
                            self.scaler.step(optimizer)
                            optimizer.zero_grad()
                        for scheduler in self.schedulers:
                            scheduler.step()
                        self.scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                        for optimizer in self.optimizers:
                            optimizer.step()
                            optimizer.zero_grad()
                        for scheduler in self.schedulers:
                            scheduler.step()
                
                # Logging
                if self.step % 10 == 0:
                    with torch.no_grad():
                        predictions = logits.argmax(dim=-1)
                        accuracy = (predictions == y).float().mean().item()
                        current_loss = loss.item() * self.config.gradient_accumulation_steps
                        perplexity = math.exp(min(current_loss, 20))
                    
                    pbar.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'acc': f'{accuracy:.3f}',
                        'ppl': f'{perplexity:.1f}',
                        'lr': f'{self.optimizers[0].param_groups[0]["lr"]:.2e}'
                    })
                
                # Evaluation
                if self.step % self.config.eval_every == 0 and self.step > 0:
                    eval_metrics = self.evaluate()
                    print(f"\nStep {self.step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                          f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                          f"Val PPL: {eval_metrics['val_perplexity']:.2f}")
                    
                    if eval_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = eval_metrics['val_loss']
                        self.save_checkpoint(is_best=True)
                
                self.step += 1
                if self.step % 10 == 0:
                    pbar.update(10)
        
        pbar.close()
        training_time = time.time() - start_time
        print(f"  ⏱️ Training completed in {training_time:.1f} seconds")
        
        # Final evaluation
        final_eval = self.evaluate()
        print(f"  📊 Final - Loss: {final_eval['val_loss']:.4f}, "
              f"Acc: {final_eval['val_accuracy']:.4f}, PPL: {final_eval['val_perplexity']:.2f}")
        
        # Save final model
        self.save_checkpoint()
        
        return self.model, final_eval