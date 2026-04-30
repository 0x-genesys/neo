"""
TPU-specific training wrapper for PyTorch XLA.

Implements proper TPU training patterns for Kaggle, Colab, and GCP:
- Multi-core training with xmp.spawn
- Proper device handling with xm.xla_device()
- Efficient data loading with ParallelLoader
- XLA-optimized gradient updates with mark_step()
- Master-only operations for logging and checkpointing
"""

import torch
import torch.nn as nn
from pathlib import Path
import time

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.runtime as xr
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False
    xm = None
    pl = None
    xmp = None
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
    TPU-optimized trainer using PyTorch XLA patterns.
    
    Follows Kaggle TPU best practices:
    1. Use xmp.spawn for multi-core training
    2. Instantiate model outside mp_fn and use MpModelWrapper
    3. Send model and data to TPU device
    4. Use ParallelLoader for efficient data loading
    5. Use xm.master_print for logging
    6. Use xm.mark_step() for gradient updates
    7. Use xser.save for memory-optimized checkpointing
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
        
        # TPU configuration
        self.num_cores = 8  # Kaggle TPU v3-8 has 8 cores
        
        # Training state (will be set in mp_fn)
        self.device = None
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Checkpoint to resume from (if any)
        self.resume_checkpoint = None
        
        # Logging - will be initialized in each spawned process
        self.writer = None
        
        # Weights & Biases
        self.use_wandb = config['logging']['use_wandb']
        self.wandb = None
        self.wandb_config = None
        if self.use_wandb:
            try:
                import wandb
                # Store wandb module and config for later initialization in mp_fn
                self.wandb = wandb
                self.wandb_config = {
                    'project': config['logging']['wandb_project'],
                    'entity': config['logging']['wandb_entity'],
                    'config': config
                }
                print("Weights & Biases will be initialized in master process")
            except ImportError:
                print("wandb not installed, disabling W&B logging")
                self.use_wandb = False
        
        print("="*80)
        print("🚀 TPU Trainer Initialized")
        print("="*80)
        print(f"TPU cores: {self.num_cores}")
        print(f"torch_xla version: {torch_xla.__version__}")
        print("="*80)
        
        # Load checkpoint if resuming
        self._load_checkpoint_if_needed()
    
    def train(self):
        """Start multi-core TPU training with PJRT runtime."""
        print("\n🚀 Starting TPU training on all available TPU cores...")
        
        # PJRT-compatible spawn: no nprocs, auto-discover cores
        # Use 'fork' as standard for Kaggle TPU VM slices
        xmp.spawn(self._mp_fn, args=(), nprocs=None, start_method='fork')
        
        print("\n✅ TPU training complete!")
    
    def _mp_fn(self, rank):
        """
        Multi-processing function that runs on each TPU core.
        PJRT-compatible: receives rank, accesses self directly.
        
        Args:
            rank: Core rank (0-7 for TPU v3-8)
        """
        # Get TPU device for this core
        self.device = xm.xla_device()
        
        # Initialize TensorBoard writer in this process (master only)
        if self.config['logging']['log_dir'] and xm.is_master_ordinal():
            log_dir = Path(self.config['logging']['log_dir'])
            log_dir.mkdir(parents=True, exist_ok=True)
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir, flush_secs=300)
        
        # Initialize W&B on master process only
        if self.use_wandb and xm.is_master_ordinal() and self.wandb_config:
            try:
                self.wandb.init(**self.wandb_config)
                xm.master_print("✅ Weights & Biases initialized")
            except Exception as e:
                xm.master_print(f"⚠️  Failed to initialize W&B: {e}")
                self.use_wandb = False
        
        # Wrap model for multi-core training
        model = xmp.MpModelWrapper(self.model)
        model = model.to(self.device)
    
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
            # Load checkpoint on CPU first (before TPU spawn)
            checkpoint = torch.load(resume_path, map_location='cpu')
            
            # Store checkpoint data for later use in mp_fn
            self.resume_checkpoint = checkpoint
            
            # Load model state (on CPU, will be moved to TPU in mp_fn)
            if 'model_state_dict' in checkpoint:
                # Handle DataParallel wrapper if present
                state_dict = checkpoint['model_state_dict']
                if any(k.startswith('module.') for k in state_dict.keys()):
                    # Remove 'module.' prefix from DataParallel
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                self.model.load_state_dict(state_dict)
                print("   ✅ Model weights loaded")
            
            # Extract training state (will be applied in mp_fn)
            if 'global_step' in checkpoint:
                self.global_step = checkpoint['global_step']
                print(f"   ✅ Resuming from step: {self.global_step}")
            
            if 'epoch' in checkpoint:
                self.epoch = checkpoint['epoch']
                print(f"   ✅ Resuming from epoch: {self.epoch}")
            
            if 'best_val_loss' in checkpoint:
                self.best_val_loss = checkpoint['best_val_loss']
                print(f"   ✅ Best validation loss: {self.best_val_loss:.4f}")
            
            print(f"✅ Checkpoint loaded successfully!")
            print(f"   Continuing training from step {self.global_step}")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("   Starting training from scratch")
            import traceback
            traceback.print_exc()
            self.resume_checkpoint = None
    
    def _mp_fn(self, rank):
        """
        Multi-processing function that runs on each TPU core.
        PJRT-compatible: receives rank, accesses self directly.
        
        Args:
            rank: Core rank (0-7 for TPU v3-8)
        """
        # Get TPU device for this core
        self.device = xm.xla_device()
        
        # Initialize TensorBoard writer in this process (master only)
        if self.config['logging']['log_dir'] and xm.is_master_ordinal():
            log_dir = Path(self.config['logging']['log_dir'])
            log_dir.mkdir(parents=True, exist_ok=True)
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir, flush_secs=300)
        
        # Initialize W&B on master process only
        if self.use_wandb and xm.is_master_ordinal() and self.wandb_config:
            try:
                self.wandb.init(**self.wandb_config)
                xm.master_print("✅ Weights & Biases initialized")
            except Exception as e:
                xm.master_print(f"⚠️  Failed to initialize W&B: {e}")
                self.use_wandb = False
        
        # Wrap model for multi-core training
        model = xmp.MpModelWrapper(self.model)
        model = model.to(self.device)
        
        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer(model)
        self.scheduler = self._create_scheduler()
        
        # Load optimizer and scheduler state from checkpoint if resuming
        if self.resume_checkpoint is not None:
            if 'optimizer_state_dict' in self.resume_checkpoint:
                try:
                    self.optimizer.load_state_dict(self.resume_checkpoint['optimizer_state_dict'])
                    xm.master_print("   ✅ Optimizer state loaded")
                except Exception as e:
                    xm.master_print(f"   ⚠️  Could not load optimizer state: {e}")
            
            if 'scheduler_state_dict' in self.resume_checkpoint and self.scheduler is not None:
                try:
                    self.scheduler.load_state_dict(self.resume_checkpoint['scheduler_state_dict'])
                    xm.master_print("   ✅ Scheduler state loaded")
                except Exception as e:
                    xm.master_print(f"   ⚠️  Could not load scheduler state: {e}")
        
        # Create distributed sampler for this core
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_loader.dataset,
            num_replicas=get_world_size(),
            rank=get_ordinal(),
            shuffle=True
        )
        
        # Create data loader with distributed sampler
        train_loader = torch.utils.data.DataLoader(
            self.train_loader.dataset,
            batch_size=self.config['training']['batch_size'],
            sampler=train_sampler,
            num_workers=self.config['data'].get('num_workers', 4),
            drop_last=True
        )
        
        # Training loop
        max_epochs = self.config['training'].get('max_epochs', 10)
        max_steps = self.config['training'].get('max_steps', None)
        
        # Start from the epoch we left off at (if resuming)
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, max_epochs):
            self.epoch = epoch
            
            # Set epoch for sampler (ensures different shuffle each epoch)
            train_sampler.set_epoch(epoch)
            
            # Create parallel loader for efficient TPU data loading
            para_loader = pl.ParallelLoader(train_loader, [self.device])
            
            # Train for one epoch
            self._train_epoch(model, para_loader.per_device_loader(self.device))
            
            # Validate
            if self.val_loader is not None:
                val_loss = self._validate(model)
                
                # Save best model (master only)
                if xm.is_master_ordinal():
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(model, f"best_model_step_{self.global_step}.pt")
            
            # Check if we've reached max steps
            if max_steps and self.global_step >= max_steps:
                xm.master_print(f"\n✅ Reached max steps: {max_steps}")
                break
    
    def _train_epoch(self, model, train_loader):
        """Train for one epoch on TPU."""
        model.train()
        
        grad_accum_steps = self.config['training']['gradient_accumulation_steps']
        log_interval = self.config['logging']['log_interval']
        save_interval = self.config['checkpoint']['save_interval']
        
        epoch_loss = 0.0
        step_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
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
            outputs = model(inputs)
            
            # Compute loss
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
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
                max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)
                xm.reduce_gradients(self.optimizer)  # Sync gradients across cores
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                xm.mark_step()  # XLA optimization barrier
                self.optimizer.zero_grad()
                
                # Update learning rate
                if self.scheduler:
                    self.scheduler.step()
                
                self.global_step += 1
                
                # Logging (master only)
                if self.global_step % log_interval == 0:
                    avg_loss = epoch_loss / step_count
                    lr = self.optimizer.param_groups[0]['lr']
                    
                    xm.master_print(
                        f"Epoch {self.epoch} | Step {self.global_step} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                    )
                    
                    # Log to TensorBoard and W&B (master only)
                    if xm.is_master_ordinal():
                        self._log_metrics({
                            'train/loss': avg_loss,
                            'train/learning_rate': lr,
                            'train/epoch': self.epoch
                        })
                    
                    epoch_loss = 0.0
                    step_count = 0
                
                # Checkpointing (master only)
                if self.global_step % save_interval == 0:
                    if xm.is_master_ordinal():
                        self._save_checkpoint(model, f"checkpoint_step_{self.global_step}.pt")
    
    def _validate(self, model):
        """Validate on TPU."""
        model.eval()
        
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            self.val_loader.dataset,
            num_replicas=get_world_size(),
            rank=get_ordinal(),
            shuffle=False
        )
        
        val_loader = torch.utils.data.DataLoader(
            self.val_loader.dataset,
            batch_size=self.config['training']['batch_size'],
            sampler=val_sampler,
            num_workers=self.config['data'].get('num_workers', 4),
            drop_last=False
        )
        
        para_loader = pl.ParallelLoader(val_loader, [self.device])
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in para_loader.per_device_loader(self.device):
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
                outputs = model(inputs)
                
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        # Reduce across all cores
        total_loss = xm.mesh_reduce('val_loss', total_loss, lambda x: sum(x))
        total_samples = xm.mesh_reduce('val_samples', total_samples, lambda x: sum(x))
        
        avg_loss = total_loss / total_samples
        
        xm.master_print(f"\n📊 Validation Loss: {avg_loss:.4f}\n")
        
        # Log to TensorBoard and W&B (master only)
        if xm.is_master_ordinal():
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
        
        # Save using standard PyTorch (xser might not be available yet)
        torch.save(checkpoint, str(checkpoint_path))
        print(f"💾 Emergency checkpoint saved: {checkpoint_path}")
    
    def _log_metrics(self, metrics):
        """Log metrics to TensorBoard and W&B (master only)."""
        if not xm.is_master_ordinal():
            return
        
        # TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, self.global_step)
        
        # Weights & Biases
        if self.use_wandb and hasattr(self, 'wandb'):
            self.wandb.log(metrics, step=self.global_step)
    
    def _save_checkpoint(self, model, filename):
        """Save checkpoint (master only, memory-optimized)."""
        if not xm.is_master_ordinal():
            return
        
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
        
        # Memory-optimized save (master only)
        xser.save(checkpoint, str(checkpoint_path), master_only=True)
        
        xm.master_print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Upload to HuggingFace Hub if enabled
        if self.config.get('huggingface_hub', {}).get('enabled', False):
            self._upload_to_hub(checkpoint_path)
    
    def _upload_to_hub(self, checkpoint_path):
        """Upload checkpoint to HuggingFace Hub."""
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
            
            xm.master_print(f"☁️  Uploaded to HuggingFace Hub: {repo_id}/{checkpoint_path.name}")
        except Exception as e:
            xm.master_print(f"⚠️  Failed to upload to HuggingFace Hub: {e}")
    
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
