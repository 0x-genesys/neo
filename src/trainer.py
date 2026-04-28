"""
Training infrastructure with checkpointing, logging, and validation.
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import os
import time
from tqdm import tqdm
import numpy as np
from pathlib import Path


class Trainer:
    """Production-ready trainer with all bells and whistles."""
    
    def __init__(self, model, train_loader, val_loader, tokenizer, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config
        
        # Device setup
        self.device = self._setup_device()
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.use_amp = config['system']['mixed_precision']
        self.scaler = GradScaler() if self.use_amp else None
        
        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Logging
        self.writer = None
        self.log_buffer = []  # Buffer for batched logging
        if config['logging']['log_dir']:
            log_dir = Path(config['logging']['log_dir'])
            log_dir.mkdir(parents=True, exist_ok=True)
            # Reduce flush frequency to save I/O
            self.writer = SummaryWriter(log_dir, flush_secs=300)  # Flush every 5 minutes instead of default 120s
            print(f"TensorBoard logging to: {log_dir} (flush every 5 min)")
        
        # Checkpointing
        self.checkpoint_dir = Path(config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Weights & Biases
        self.use_wandb = config['logging']['use_wandb']
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config['logging']['wandb_project'],
                    entity=config['logging']['wandb_entity'],
                    config=config
                )
                self.wandb = wandb
            except ImportError:
                print("wandb not installed, disabling W&B logging")
                self.use_wandb = False
        
        print(f"Trainer initialized on device: {self.device}")
        print(f"Mixed precision training: {self.use_amp}")
    
    def _setup_device(self):
        """Setup training device with proper detection and memory checking."""
        from .device_utils import select_device, check_mixed_precision_support, print_device_recommendations
        
        device_name = self.config['system']['device']
        device = select_device(device_name, verbose=True)
        
        # Check GPU memory if using CUDA
        if device.type == 'cuda':
            import torch.cuda as cuda
            total_memory = cuda.get_device_properties(device).total_memory / 1e9  # GB
            print(f"\n🔍 GPU Memory Check:")
            print(f"   Total memory: {total_memory:.2f}GB")
            
            # Estimate model memory requirements
            model_params = sum(p.numel() for p in self.model.parameters())
            model_memory_gb = model_params * 2 / 1e9  # FP16
            optimizer_memory_gb = model_params * 8 / 1e9  # Adam states
            
            batch_size = self.config['training']['batch_size']
            context_length = self.config['model']['context_length']
            d_model = self.config['model']['d_model']
            num_layers = self.config['model']['num_layers']
            
            # Rough activation estimate (with gradient checkpointing)
            if self.config['model'].get('use_gradient_checkpointing', False):
                activation_memory_gb = batch_size * context_length * d_model * num_layers * 2 / 1e9  # Reduced by checkpointing
            else:
                activation_memory_gb = batch_size * context_length * d_model * num_layers * 4 * 2 / 1e9
            
            estimated_memory = model_memory_gb + optimizer_memory_gb + activation_memory_gb + 1.5  # +1.5GB buffer
            
            print(f"   Estimated usage: {estimated_memory:.2f}GB")
            print(f"     - Model (FP16): {model_memory_gb:.2f}GB")
            print(f"     - Optimizer: {optimizer_memory_gb:.2f}GB")
            print(f"     - Activations: {activation_memory_gb:.2f}GB")
            print(f"     - Buffer: 1.50GB")
            
            if estimated_memory > total_memory * 0.9:
                print(f"\n⚠️  WARNING: Estimated memory ({estimated_memory:.1f}GB) is close to GPU limit ({total_memory:.1f}GB)")
                print(f"   Recommendations:")
                print(f"   1. Reduce batch_size (current: {batch_size})")
                print(f"   2. Enable gradient_checkpointing (current: {self.config['model'].get('use_gradient_checkpointing', False)})")
                print(f"   3. Reduce context_length (current: {context_length})")
                print(f"   4. Use config: gpu_training_117m_15gb.yaml for 15GB GPUs")
                print(f"\n   Consider using: python train.py --config config/gpu_training_117m_15gb.yaml")
            else:
                print(f"   ✅ Memory estimate looks good ({estimated_memory/total_memory*100:.0f}% of GPU)")
        
        # Check mixed precision support
        if self.config['system']['mixed_precision']:
            if not check_mixed_precision_support(device):
                print("⚠️  Mixed precision not recommended for this device. Disabling.")
                self.config['system']['mixed_precision'] = False
        
        # Print recommendations
        print_device_recommendations(device)
        
        return device
    
    def _create_optimizer(self):
        """Create optimizer with weight decay."""
        # Separate parameters that should and shouldn't have weight decay
        decay = set()
        no_decay = set()
        
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters(recurse=False):  # Don't recurse to avoid duplicates
                fpn = f'{mn}.{pn}' if mn else pn
                
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, (nn.Linear, nn.Embedding)):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, nn.LayerNorm):
                    no_decay.add(fpn)
        
        # Get actual parameters (handles weight tying)
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        
        # Only include parameters that actually exist
        decay_params = [param_dict[pn] for pn in sorted(list(decay)) if pn in param_dict]
        no_decay_params = [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict]
        
        optim_groups = [
            {
                "params": decay_params,
                "weight_decay": self.config['training']['weight_decay']
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0
            }
        ]
        
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config['training']['learning_rate'],
            betas=self.config['optimizer']['betas'],
            eps=self.config['optimizer']['eps']
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup support."""
        import math
        from torch.optim.lr_scheduler import LambdaLR
        
        warmup_steps = self.config['training']['warmup_steps']
        max_steps = self.config['training']['max_steps']
        min_lr = self.config['scheduler']['min_lr']
        max_lr = self.config['training']['learning_rate']
        min_lr_ratio = min_lr / max_lr
        
        def lr_lambda(step):
            """
            Learning rate schedule with linear warmup and cosine decay.
            
            Phase 1 (0 to warmup_steps): Linear warmup from 0 to 1.0
            Phase 2 (warmup_steps to max_steps): Cosine decay from 1.0 to min_lr_ratio
            """
            if step < warmup_steps:
                # Linear warmup: gradually increase from 0 to 1
                return step / max(1, warmup_steps)
            else:
                # Cosine decay: smoothly decrease from 1 to min_lr_ratio
                progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
                progress = min(progress, 1.0)  # Clamp to [0, 1]
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
        
        scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        
        print(f"✅ Learning rate scheduler created:")
        print(f"   - Warmup steps: {warmup_steps}")
        print(f"   - Max steps: {max_steps}")
        print(f"   - Max LR: {max_lr:.2e}")
        print(f"   - Min LR: {min_lr:.2e}")
        
        return scheduler
    
    def save_checkpoint(self, filename='checkpoint.pt', is_best=False):
        """Save training checkpoint."""
        # Handle DataParallel wrapper
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, filepath):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Handle DataParallel wrapper
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Checkpoint loaded from: {filepath}")
        print(f"Resuming from epoch {self.epoch}, step {self.global_step}")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Calculate how many batches to skip if resuming mid-epoch
        batches_per_step = self.config['training']['gradient_accumulation_steps']
        start_batch = (self.global_step % (len(self.train_loader) // batches_per_step)) * batches_per_step
        
        pbar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.epoch} (Step {self.global_step}/{self.config['training']['max_steps']})",
            initial=start_batch,
            total=len(self.train_loader)
        )
        
        for batch_idx, (input_ids, targets) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                logits, loss = self.model(input_ids, targets)
                loss = loss / self.config['training']['gradient_accumulation_steps']
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad(set_to_none=True)
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config['training']['log_interval'] == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    self._log_metrics({
                        'train/loss': loss.item() * self.config['training']['gradient_accumulation_steps'],
                        'train/lr': lr,
                        'train/step': self.global_step
                    })
                
                # Validation
                if self.global_step % self.config['training']['eval_interval'] == 0:
                    val_loss = self.validate()
                    self._log_metrics({'val/loss': val_loss})
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(is_best=True)
                    
                    self.model.train()
                
                # Checkpointing
                if self.global_step % self.config['training']['save_interval'] == 0:
                    self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
                
                # Max steps check
                if self.global_step >= self.config['training']['max_steps']:
                    avg_loss = total_loss / num_batches if num_batches > 0 else 0
                    return avg_loss
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'step': f"{self.global_step}/{self.config['training']['max_steps']}"
            })
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    @torch.no_grad()
    def validate(self):
        """Run validation."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        if self.val_loader is None:
            return float('inf')
        
        for input_ids, targets in tqdm(self.val_loader, desc="Validation"):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            with autocast(enabled=self.use_amp):
                logits, loss = self.model(input_ids, targets)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        # Generate samples
        self._generate_samples()
        
        return avg_loss
    
    @torch.no_grad()
    def _generate_samples(self):
        """Generate text samples for qualitative evaluation."""
        from .data import get_sample_prompts
        from .tokenizer_utils import encode_to_tensor, decode_from_tensor
        
        # Get the actual model (unwrap DataParallel if needed)
        model_for_generation = self.model.module if hasattr(self.model, 'module') else self.model
        model_for_generation.eval()
        
        prompts = get_sample_prompts()
        
        print("\n" + "="*80)
        print("Generated Samples:")
        print("="*80)
        
        for prompt in prompts[:self.config['generation']['num_samples']]:
            # Tokenize prompt
            input_ids = encode_to_tensor(self.tokenizer, prompt, self.device)
            
            # Generate
            output_ids = model_for_generation.generate(
                input_ids,
                max_new_tokens=self.config['generation']['max_new_tokens'],
                temperature=self.config['generation']['temperature'],
                top_k=self.config['generation']['top_k'],
                top_p=self.config['generation']['top_p']
            )
            
            # Decode
            generated_text = decode_from_tensor(self.tokenizer, output_ids[0], skip_special_tokens=True)
            
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated_text}")
            print("-"*80)
        
        print("="*80 + "\n")
    
    def _log_metrics(self, metrics):
        """Log metrics to tensorboard and wandb."""
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, self.global_step)
        
        if self.use_wandb:
            self.wandb.log(metrics, step=self.global_step)
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80)
        print(f"Total epochs: {self.config['training']['max_epochs']}")
        print(f"Max steps: {self.config['training']['max_steps']}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Gradient accumulation steps: {self.config['training']['gradient_accumulation_steps']}")
        print(f"Effective batch size: {self.config['training']['batch_size'] * self.config['training']['gradient_accumulation_steps']}")
        print("="*80 + "\n")
        
        # Resume from checkpoint if specified
        if self.config['checkpoint']['resume_from']:
            self.load_checkpoint(self.config['checkpoint']['resume_from'])
        
        start_time = time.time()
        
        for epoch in range(self.epoch, self.config['training']['max_epochs']):
            self.epoch = epoch
            
            epoch_loss = self.train_epoch()
            
            if epoch_loss is not None:
                print(f"\nEpoch {epoch} completed. Average loss: {epoch_loss:.4f}")
            
            # Check if max steps reached
            if self.global_step >= self.config['training']['max_steps']:
                print(f"\nReached max steps ({self.config['training']['max_steps']}). Stopping training.")
                break
        
        # Final validation
        print("\nRunning final validation...")
        final_val_loss = self.validate()
        print(f"Final validation loss: {final_val_loss:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint('final_model.pt')
        
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time/3600:.2f} hours")
        
        if self.writer is not None:
            self.writer.close()
        
        if self.use_wandb:
            self.wandb.finish()
