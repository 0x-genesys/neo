"""
TPU-specific training wrapper for PyTorch XLA.

Implements proper TPU training patterns for Kaggle single-VM TPU:
- Single-process training (no xmp.spawn for single VM)
- Proper device handling with xm.xla_device()
- Efficient data loading with ParallelLoader
- XLA-optimized gradient updates with mark_step()
- Memory-optimized checkpointing
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
import time

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.runtime as xr
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False
    xm = None
    pl = None
    xr = None


def get_world_size():
    """Get TPU world size (number of cores) - compatible with old and new API."""
    if not TPU_AVAILABLE:
        return 1
    try:
        # Try new API first (torch_xla 2.0+)
        if hasattr(xr, 'world_size'):
            return xr.world_size()
        # Fall back to old API
        elif hasattr(xm, 'xrt_world_size'):
            return xm.xrt_world_size()
        else:
            return 8  # Default for TPU v3-8
    except:
        return 8


def get_ordinal():
    """Get TPU ordinal (rank) - compatible with old and new API."""
    if not TPU_AVAILABLE:
        return 0
    try:
        # Try new API first (torch_xla 2.0+)
        if hasattr(xr, 'global_ordinal'):
            return xr.global_ordinal()
        # Fall back to old API
        elif hasattr(xm, 'get_ordinal'):
            return xm.get_ordinal()
        else:
            return 0
    except:
        return 0


class TPUTrainer:
    """
    TPU-optimized trainer using PyTorch XLA patterns for single-VM TPU.
    
    Follows Kaggle TPU best practices for single VM (8 cores):
    1. Single-process training (no xmp.spawn)
    2. Use ParallelLoader to distribute data across cores
    3. Send model and data to TPU device
    4. Use xm.mark_step() for gradient updates
    5. Use xser.save for memory-optimized checkpointing
    """
    
    def __init__(self, model, train_loader, val_loader, tokenizer, config):
        """Initialize TPU trainer."""
        if not TPU_AVAILABLE:
            raise ImportError(
                "torch_xla not installed. Install with:\n"
                "  curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py\n"
                "  python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev"
            )
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config
        
        # Ensure required config sections exist with defaults
        if 'logging' not in config:
            config['logging'] = {}
        if 'checkpoint' not in config:
            config['checkpoint'] = {}
        if 'training' not in config:
            config['training'] = {}
        
        # Set defaults for missing keys
        config['logging'].setdefault('log_interval', 10)
        config['logging'].setdefault('log_dir', 'logs')
        config['logging'].setdefault('use_wandb', False)
        config['checkpoint'].setdefault('save_interval', 1000)
        config['checkpoint'].setdefault('save_dir', 'checkpoints')
        config['training'].setdefault('gradient_accumulation_steps', 1)
        config['training'].setdefault('max_grad_norm', 1.0)
        config['training'].setdefault('eval_interval', 500)
        
        # TPU configuration
        self.num_cores = 8  # Kaggle TPU v3-8 has 8 cores
        
        # Training state
        self.device = None
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Checkpoint to resume from (if any)
        self.resume_checkpoint = None
        
        # Logging
        self.writer = None
        
        # Weights & Biases
        self.use_wandb = config['logging']['use_wandb']
        self.wandb = None
        self.wandb_config = None
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                self.wandb_config = {
                    'project': config['logging']['wandb_project'],
                    'entity': config['logging']['wandb_entity'],
                    'config': config
                }
                print("Weights & Biases will be initialized during training")
            except ImportError:
                print("wandb not installed, disabling W&B logging")
                self.use_wandb = False
        
        print("="*80)
        print("🚀 TPU Trainer Initialized (Single-Process Mode)")
        print("="*80)
        print(f"TPU cores: {self.num_cores}")
        print(f"torch_xla version: {torch_xla.__version__}")
        print("="*80)
        
        # Load checkpoint if resuming
        self._load_checkpoint_if_needed()
    
    def train(self):
        """Start single-process TPU training with ParallelLoader."""
        print("\n🚀 Starting TPU training (single-process, multi-core)...")
        
        # Single-process training - no xmp.spawn needed for single VM
        # ParallelLoader will handle data distribution across 8 cores
        self._train_single_process()
        
        print("\n✅ TPU training complete!")
    
    def _train_single_process(self):
        """
        Single-process training function for single-VM TPU.
        Uses ParallelLoader to distribute data across 8 cores automatically.
        No xmp.spawn needed - XLA handles multi-core parallelism internally.
        """
        # Get TPU device
        self.device = xm.xla_device()
        
        print(f"📍 Using TPU device: {self.device}")
        print(f"📍 World size (cores): {get_world_size()}")
        print(f"📍 Ordinal (rank): {get_ordinal()}")
        
        # Check if curriculum learning is enabled
        curriculum_config = self.config.get('training', {}).get('curriculum_learning', {})
        curriculum_enabled = curriculum_config.get('enabled', False)
        
        if curriculum_enabled:
            print(f"\n🎓 Curriculum Learning: ENABLED")
            print(f"   Phases: Pattern Discovery → Foundation → Bridge → Refinement")
        else:
            print(f"\n📚 Curriculum Learning: DISABLED (standard training)")
        print()
        
        # Initialize TensorBoard writer
        if self.config['logging']['log_dir']:
            log_dir = Path(self.config['logging']['log_dir'])
            log_dir.mkdir(parents=True, exist_ok=True)
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir, flush_secs=300)
        
        # Initialize W&B
        if self.use_wandb and self.wandb_config:
            try:
                self.wandb.init(**self.wandb_config)
                print("✅ Weights & Biases initialized")
            except Exception as e:
                print(f"⚠️  Failed to initialize W&B: {e}")
                self.use_wandb = False
        
        # Move model to TPU device (no MpModelWrapper needed for single-process)
        model = self.model.to(self.device)
        
        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer(model)
        self.scheduler = self._create_scheduler()
        
        # Load optimizer and scheduler state from checkpoint if resuming
        if self.resume_checkpoint is not None:
            if 'optimizer_state_dict' in self.resume_checkpoint:
                try:
                    self.optimizer.load_state_dict(self.resume_checkpoint['optimizer_state_dict'])
                    print("   ✅ Optimizer state loaded")
                except Exception as e:
                    print(f"   ⚠️  Could not load optimizer state: {e}")
            
            if 'scheduler_state_dict' in self.resume_checkpoint and self.scheduler is not None:
                try:
                    self.scheduler.load_state_dict(self.resume_checkpoint['scheduler_state_dict'])
                    print("   ✅ Scheduler state loaded")
                except Exception as e:
                    print(f"   ⚠️  Could not load scheduler state: {e}")
        
        # Training loop
        max_epochs = self.config['training'].get('max_epochs', 10)
        max_steps = self.config['training'].get('max_steps', None)
        
        # Start from the epoch we left off at (if resuming)
        start_epoch = self.epoch
        
        # Check if curriculum learning is enabled
        curriculum_config = self.config.get('training', {}).get('curriculum_learning', {})
        curriculum_enabled = curriculum_config.get('enabled', False)
        
        # If resuming mid-training, update curriculum to match current epoch
        if start_epoch > 0 and curriculum_enabled and hasattr(self.train_loader.dataset, 'update_distribution'):
            epoch_distributions = curriculum_config.get('epoch_distributions', {})
            epoch_num = start_epoch + 1  # 1-indexed for config
            
            if epoch_num in epoch_distributions:
                new_distribution = epoch_distributions[epoch_num]
                sources = curriculum_config.get('sources', ['wikitext', 'stack', 'ultrachat'])
                
                print(f"\n{'='*80}")
                print(f"🎓 RESUMING WITH CURRICULUM - Epoch {epoch_num}")
                print(f"{'='*80}")
                print(f"Restoring dataset distribution for current epoch:")
                for source, pct in zip(sources, new_distribution):
                    print(f"  {source:12s}: {pct:3d}%")
                print(f"{'='*80}\n")
                
                # Update the dataset distribution
                self.train_loader.dataset.update_distribution(new_distribution)
        
        for epoch in range(start_epoch, max_epochs):
            self.epoch = epoch
            
            # Update curriculum distribution if enabled (only at epoch boundaries)
            if curriculum_enabled and hasattr(self.train_loader.dataset, 'update_distribution'):
                epoch_distributions = curriculum_config.get('epoch_distributions', {})
                # Epochs are 0-indexed in code but 1-indexed in config
                epoch_num = epoch + 1
                
                # Only update if this is a new epoch (not resuming mid-epoch)
                if epoch > start_epoch and epoch_num in epoch_distributions:
                    new_distribution = epoch_distributions[epoch_num]
                    sources = curriculum_config.get('sources', ['wikitext', 'stack', 'ultrachat'])
                    
                    print(f"\n{'='*80}")
                    print(f"🎓 CURRICULUM UPDATE - Epoch {epoch_num}")
                    print(f"{'='*80}")
                    
                    # Determine phase name
                    if epoch_num == 1:
                        phase = "Current: Pattern Discovery (detect leakage)"
                    elif epoch_num in [2, 3]:
                        phase = "Foundation: Knowledge/Logic Hardening"
                    elif epoch_num == 4:
                        phase = "Bridge A: Balanced Contextualization"
                    elif epoch_num == 5:
                        phase = "Bridge B: Priority Shift"
                    elif epoch_num == 6:
                        phase = "Bridge C: Instruction Emergence"
                    elif epoch_num in [7, 8]:
                        phase = "Refinement: Behavior & Formatting (The 'Flip')"
                    else:
                        phase = "Custom Distribution"
                    
                    print(f"Phase: {phase}")
                    print(f"\nDataset Distribution:")
                    for source, pct in zip(sources, new_distribution):
                        print(f"  {source:12s}: {pct:3d}%")
                    print(f"{'='*80}\n")
                    
                    # Update the dataset distribution
                    self.train_loader.dataset.update_distribution(new_distribution)
            
            # Create parallel loader for efficient TPU data loading
            # This automatically distributes data across all 8 TPU cores
            para_loader = pl.ParallelLoader(self.train_loader, [self.device])
            
            # Train for one epoch
            self._train_epoch(model, para_loader.per_device_loader(self.device))
            
            # Validate
            if self.val_loader is not None:
                val_loss = self._validate(model)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(model, f"best_model_step_{self.global_step}.pt")
            
            # Check if we've reached max steps
            if max_steps and self.global_step >= max_steps:
                print(f"\n✅ Reached max steps: {max_steps}")
                break
    
    def _load_checkpoint_if_needed(self):
        """Load checkpoint if resume_from is specified in config."""
        resume_path = self.config.get('checkpoint', {}).get('resume_from')
        
        if not resume_path:
            return
        
        resume_path = Path(resume_path)
        if not resume_path.exists():
            print(f"⚠️  Checkpoint not found: {resume_path}")
            print("   Starting training from scratch")
            return
        
        print(f"\n📥 Loading checkpoint from: {resume_path}")
        
        try:
            # Load checkpoint on CPU first
            checkpoint = torch.load(resume_path, map_location='cpu')
            
            # Store checkpoint data for later use
            self.resume_checkpoint = checkpoint
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                # Handle DataParallel wrapper if present
                state_dict = checkpoint['model_state_dict']
                if any(k.startswith('module.') for k in state_dict.keys()):
                    # Remove 'module.' prefix from DataParallel
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                self.model.load_state_dict(state_dict)
                print("   ✅ Model weights loaded")
            
            # Extract training state
            if 'global_step' in checkpoint:
                self.global_step = checkpoint['global_step']
                print(f"   ✅ Resuming from step: {self.global_step}")
            
            if 'epoch' in checkpoint:
                self.epoch = checkpoint['epoch']
                print(f"   ✅ Resuming from epoch: {self.epoch}")
            else:
                # Calculate epoch from global_step if not in checkpoint
                # Estimate: steps_per_epoch = len(train_loader) / grad_accum_steps
                if self.global_step > 0 and hasattr(self.train_loader, 'dataset'):
                    batch_size = self.config['training'].get('batch_size', 16)
                    grad_accum = self.config['training'].get('gradient_accumulation_steps', 1)
                    dataset_size = len(self.train_loader.dataset)
                    steps_per_epoch = (dataset_size // batch_size) // grad_accum
                    if steps_per_epoch > 0:
                        self.epoch = self.global_step // steps_per_epoch
                        print(f"   ✅ Calculated epoch from step: {self.epoch}")
            
            if 'best_val_loss' in checkpoint:
                self.best_val_loss = checkpoint['best_val_loss']
                print(f"   ✅ Best validation loss: {self.best_val_loss:.4f}")
            
            print(f"✅ Checkpoint loaded successfully!")
            print(f"   Continuing training from epoch {self.epoch}, step {self.global_step}")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("   Starting training from scratch")
            import traceback
            traceback.print_exc()
            self.resume_checkpoint = None
    
    def _train_epoch(self, model, train_loader):
        """Train for one epoch on TPU."""
        model.train()
        
        grad_accum_steps = self.config['training']['gradient_accumulation_steps']
        log_interval = self.config['logging']['log_interval']
        save_interval = self.config['checkpoint']['save_interval']
        eval_interval = self.config['training']['eval_interval']
        
        epoch_loss = 0.0
        step_count = 0
        
        # Calculate how many batches to skip if resuming mid-epoch
        # Use self.train_loader (original DataLoader) not train_loader (PerDeviceLoader)
        batch_size = self.config['training']['batch_size']
        dataset_size = len(self.train_loader.dataset)
        steps_per_epoch = (dataset_size // batch_size) // grad_accum_steps
        batches_to_skip = 0
        
        if self.global_step > 0:
            # Calculate which batch we should be at
            completed_steps_this_epoch = self.global_step % steps_per_epoch
            batches_to_skip = completed_steps_this_epoch * grad_accum_steps
            if batches_to_skip > 0:
                print(f"⏭️  Skipping {batches_to_skip} batches to resume from step {self.global_step}")
        
        # Add tqdm progress bar
        try:
            from tqdm import tqdm
            pbar = tqdm(enumerate(train_loader), total=len(self.train_loader), 
                       desc=f"Epoch {self.epoch}", 
                       disable=False,
                       initial=batches_to_skip)
        except ImportError:
            # Fallback if tqdm not available
            pbar = enumerate(train_loader)
        
        for batch_idx, batch in pbar:
            # Skip batches if resuming mid-epoch
            if batch_idx < batches_to_skip:
                continue
            # Move data to TPU device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                inputs = batch['input_ids']
                targets = batch['labels']
            else:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            
            # Forward pass
            outputs = model(inputs, targets)
            
            # Handle model output - can be tuple (logits, loss) or dict
            if isinstance(outputs, tuple):
                logits, loss = outputs
                # If model computed loss, use it; otherwise compute manually
                if loss is None:
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )
            elif isinstance(outputs, dict):
                logits = outputs['logits']
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
            else:
                # outputs is just logits tensor
                logits = outputs
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
            
            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
            
            # Backward pass
            loss.backward()
            
            # Accumulate loss
            epoch_loss += loss.item() * grad_accum_steps
            step_count += 1
            
            # Gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0:
                # Gradient clipping
                max_grad_norm = self.config['training']['max_grad_norm']
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                xm.mark_step()  # XLA optimization barrier - critical for TPU
                self.optimizer.zero_grad()
                
                # Update learning rate
                if self.scheduler:
                    self.scheduler.step()
                
                self.global_step += 1
                
                # Logging - always print, but only log to TB/W&B at intervals
                avg_loss = epoch_loss / step_count if step_count > 0 else 0.0
                lr = self.optimizer.param_groups[0]['lr']
                
                # Print every step (or every N steps if you want less output)
                if step_count > 0 and (self.global_step % log_interval == 0 or step_count >= log_interval):
                    log_msg = (
                        f"Epoch {self.epoch} | Step {self.global_step} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                    )
                    print(log_msg)
                    
                    # Update tqdm if available
                    if hasattr(pbar, 'set_postfix'):
                        pbar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'lr': f'{lr:.2e}',
                            'step': self.global_step
                        })
                    
                    # Log to TensorBoard and W&B
                    self._log_metrics({
                        'train/loss': avg_loss,
                        'train/learning_rate': lr,
                        'train/epoch': self.epoch
                    })
                    
                    epoch_loss = 0.0
                    step_count = 0
                
                # Validation at eval_interval
                if self.global_step % eval_interval == 0:
                    print(f"\n{'='*80}")
                    print(f"🔍 VALIDATION at step {self.global_step}")
                    print(f"{'='*80}")
                    val_loss = self._validate(model)
                    
                    print(f"Validation loss: {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        print(f"✅ New best validation loss! Saving best model...")
                        self._save_checkpoint(model, f"best_model_step_{self.global_step}.pt")
                    else:
                        print(f"Current best: {self.best_val_loss:.4f}")
                    
                    model.train()
                    print(f"{'='*80}\n")
                
                # Checkpointing
                if self.global_step % save_interval == 0:
                    self._save_checkpoint(model, f"checkpoint_step_{self.global_step}.pt")
    
    def _validate(self, model):
        """Validate on TPU."""
        model.eval()
        
        # Create parallel loader for validation
        para_loader = pl.ParallelLoader(self.val_loader, [self.device])
        
        total_loss = 0.0
        total_samples = 0
        
        # Add tqdm progress bar
        try:
            from tqdm import tqdm
            val_iter = tqdm(para_loader.per_device_loader(self.device), 
                           desc="Validation", 
                           total=len(self.val_loader),
                           disable=False)
        except ImportError:
            val_iter = para_loader.per_device_loader(self.device)
        
        with torch.no_grad():
            for batch in val_iter:
                # Move data to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    inputs = batch['input_ids']
                    targets = batch['labels']
                else:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                
                # Forward pass
                outputs = model(inputs, targets)
                
                # Handle model output - can be tuple (logits, loss) or dict
                if isinstance(outputs, tuple):
                    logits, loss = outputs
                    # If model computed loss, use it; otherwise compute manually
                    if loss is None:
                        loss = nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            targets.view(-1)
                        )
                elif isinstance(outputs, dict):
                    logits = outputs['logits']
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )
                else:
                    # outputs is just logits tensor
                    logits = outputs
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )
                
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        avg_loss = total_loss / total_samples
        
        print(f"\n📊 Validation Loss: {avg_loss:.4f}\n")
        
        # Log to TensorBoard and W&B
        self._log_metrics({
            'val/loss': avg_loss,
            'val/epoch': self.epoch
        })
        
        return avg_loss
    
    def save_checkpoint(self, filename='checkpoint.pt'):
        """
        Public method to save checkpoint (for error handling in train.py).
        Delegates to internal _save_checkpoint method.
        """
        # Create a minimal checkpoint if training hasn't started yet
        checkpoint_path = Path(self.config['checkpoint']['save_dir']) / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save using standard PyTorch
        torch.save(checkpoint, str(checkpoint_path))
        print(f"💾 Emergency checkpoint saved: {checkpoint_path}")
    
    def _log_metrics(self, metrics):
        """Log metrics to TensorBoard and W&B."""
        # TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, self.global_step)
        
        # Weights & Biases
        if self.use_wandb and hasattr(self, 'wandb'):
            self.wandb.log(metrics, step=self.global_step)
    
    def _save_checkpoint(self, model, filename):
        """Save checkpoint (memory-optimized)."""
        import torch_xla.utils.serialization as xser
        
        checkpoint_path = Path(self.config['checkpoint']['save_dir']) / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Memory-optimized save
        xser.save(checkpoint, str(checkpoint_path))
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Upload to HuggingFace Hub if enabled
        if self.config.get('huggingface_hub', {}).get('enabled', False):
            self._upload_to_hub(checkpoint_path)
    
    def _upload_to_hub(self, checkpoint_path):
        """Upload checkpoint to HuggingFace Hub and clean up local file."""
        try:
            from huggingface_hub import HfApi
            
            repo_id = self.config['huggingface_hub']['repo_id']
            api = HfApi()
            
            api.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=checkpoint_path.name,
                repo_id=repo_id,
                repo_type="model"
            )
            
            print(f"☁️  Uploaded to HuggingFace Hub: {repo_id}/{checkpoint_path.name}")
            
            # Clean up local checkpoint after successful upload to save disk space
            # Keep only best_model checkpoints locally
            if 'best_model' not in checkpoint_path.name:
                try:
                    checkpoint_path.unlink()
                    print(f"🗑️  Deleted local checkpoint (saved to Hub): {checkpoint_path.name}")
                except Exception as e:
                    print(f"⚠️  Could not delete local checkpoint: {e}")
            else:
                print(f"💾 Keeping best model checkpoint locally: {checkpoint_path.name}")
                
        except Exception as e:
            print(f"⚠️  Failed to upload to HuggingFace Hub: {e}")
    
    def _create_optimizer(self, model):
        """Create optimizer."""
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training'].get('weight_decay', 0.01)
        betas = self.config['training'].get('betas', (0.9, 0.999))
        
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_type = self.config['training'].get('scheduler', 'cosine')
        warmup_steps = self.config['training'].get('warmup_steps', 1000)
        max_steps = self.config['training'].get('max_steps', 10000)
        
        if scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(self.optimizer, T_max=max_steps - warmup_steps)
        elif scheduler_type == 'linear':
            from torch.optim.lr_scheduler import LinearLR
            return LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=max_steps)
        else:
            return None


def is_tpu_available():
    """Check if TPU is available."""
    return TPU_AVAILABLE


def get_tpu_info():
    """Get TPU information."""
    if not TPU_AVAILABLE:
        return None
    
    try:
        info = {
            'available': True,
            'version': torch_xla.__version__,
            'num_cores': get_world_size(),
            'ordinal': get_ordinal(),
            'device': str(xm.xla_device())
        }
        return info
    except Exception as e:
        return {'available': False, 'error': str(e)}
