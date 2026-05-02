"""
Base Fine-Tuning Trainer with LoRA and Chain-of-Thought Support
Shared core logic using peft and transformers for hardware-adaptive fine-tuning.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import time


class LoRAFineTuner:
    """
    Base trainer for fine-tuning with LoRA (Low-Rank Adaptation).
    
    Specifications:
    - LoRA rank: 16, alpha: 32
    - Target modules: All linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
    - Dropout: 0.1
    - Frozen embeddings (100k token embedding layer)
    - AdamW optimizer with betas=(0.9, 0.999)
    - Fixed learning rate: 2.0e-5 (prevents catastrophic forgetting)
    - Cosine scheduler with 10% warmup
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        train_dataset,
        val_dataset,
        output_dir: str = "finetuned_model",
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 2.0e-5,
        weight_decay: float = 0.05,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        logging_steps: int = 10,
        eval_steps: int = 100,
        save_steps: int = 500,
        device: str = "auto",
        use_amp: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        resume_from_checkpoint: str = None,
        upload_to_hub: bool = False,
        hub_repo_id: str = "0x-genesys/neo_weights_checkpoints",
        hub_path_prefix: str = "finetune/",
    ):
        """
        Initialize the LoRA fine-tuner.
        
        Args:
            model: Base transformer model (117M parameters)
            tokenizer: Tokenizer for the model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Directory to save checkpoints
            batch_size: Training batch size
            gradient_accumulation_steps: Number of steps to accumulate gradients
            num_epochs: Number of training epochs
            learning_rate: Fixed learning rate (2.0e-5 recommended)
            weight_decay: Weight decay for AdamW
            warmup_ratio: Warmup ratio (0.1 = 10% warmup)
            max_grad_norm: Maximum gradient norm for clipping
            logging_steps: Log every N steps
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
            use_amp: Use automatic mixed precision (FP16)
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            resume_from_checkpoint: Path to checkpoint to resume from
            upload_to_hub: Upload checkpoints to HuggingFace Hub
            hub_repo_id: HuggingFace repository ID
            hub_path_prefix: Path prefix in repository (e.g., "finetune/")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training hyperparameters
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.use_amp = use_amp
        
        # LoRA configuration
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # HuggingFace Hub configuration
        self.upload_to_hub = upload_to_hub
        self.hub_repo_id = hub_repo_id
        self.hub_path_prefix = hub_path_prefix
        
        # Device setup
        self.device = self._setup_device(device)
        
        # Add config attribute to model for PEFT compatibility
        # This is required by PEFT but won't interfere with regular training/inference
        if not hasattr(self.model, 'config'):
            from types import SimpleNamespace
            # Extract model parameters
            vocab_size = getattr(self.model, 'token_embedding', None)
            if vocab_size is not None:
                vocab_size = vocab_size.num_embeddings
            else:
                vocab_size = 100277  # Default
            
            d_model = getattr(self.model, 'd_model', 768)
            context_length = getattr(self.model, 'context_length', 512)
            
            # Count layers
            num_layers = len(self.model.blocks) if hasattr(self.model, 'blocks') else 12
            
            # Infer num_heads from first attention layer
            num_heads = 12  # Default
            if hasattr(self.model, 'blocks') and len(self.model.blocks) > 0:
                first_block = self.model.blocks[0]
                if hasattr(first_block, 'attn') and hasattr(first_block.attn, 'num_heads'):
                    num_heads = first_block.attn.num_heads
            
            self.model.config = SimpleNamespace(
                vocab_size=vocab_size,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                context_length=context_length,
                model_type="gpt",  # PEFT uses this to determine model type
            )
        
        # Apply LoRA to model
        self.model = self._apply_lora()
        
        # Freeze embeddings explicitly
        self._freeze_embeddings()
        
        # Move model to device
        self.model.to(self.device)
        
        # Create data loaders
        self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(val_dataset, shuffle=False) if val_dataset else None
        
        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
        
        # Print configuration
        self._print_config()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup training device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _apply_lora(self) -> nn.Module:
        """
        Apply LoRA to the model.
        
        Target modules: Specific linear layers in attention and output
        - Attention: c_attn (combined QKV), c_proj (output projection)
        - Output: lm_head (language model head)
        
        Note: We avoid targeting MLP layers with numeric names ('0', '2') as they
        can cause PEFT to incorrectly match parent modules.
        """
        print("\n" + "="*80)
        print("🔧 Applying LoRA Configuration")
        print("="*80)
        
        # Target only the attention and output projection layers
        # Avoid numeric module names which can cause PEFT matching issues
        target_modules = ["c_attn", "c_proj", "lm_head"]
        
        print(f"Target modules identified: {target_modules}")
        print(f"Note: Targeting attention and output layers only to avoid PEFT matching issues")
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=None,  # Don't save any additional modules
        )
        
        # Apply LoRA
        model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"\n✅ LoRA Applied Successfully")
        print(f"   Rank (r): {self.lora_r}")
        print(f"   Alpha (α): {self.lora_alpha}")
        print(f"   Dropout: {self.lora_dropout}")
        print(f"   Target modules: {target_modules}")
        print(f"\n📊 Parameter Statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Trainable %: {100 * trainable_params / total_params:.2f}%")
        print("="*80 + "\n")
        
        return model
    
    def _freeze_embeddings(self):
        """Explicitly freeze the 100k token embedding layer."""
        print("🔒 Freezing embedding layers...")
        
        frozen_params = 0
        for name, param in self.model.named_parameters():
            if 'embedding' in name.lower():
                param.requires_grad = False
                frozen_params += param.numel()
                print(f"   Frozen: {name} ({param.numel():,} parameters)")
        
        print(f"✅ Frozen {frozen_params:,} embedding parameters\n")
    
    def _create_dataloader(self, dataset, shuffle: bool) -> DataLoader:
        """Create data loader."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Set to 0 for compatibility
            pin_memory=True if self.device.type == "cuda" else False,
        )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create AdamW optimizer with proper weight decay.
        
        Optimizer: AdamW with betas=(0.9, 0.999)
        Learning Rate: Fixed at 2.0e-5 (crucial to prevent catastrophic forgetting)
        Weight Decay: 0.05
        """
        # Separate parameters with and without weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # No weight decay for biases and layer norms
            if 'bias' in name or 'ln' in name or 'layernorm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """
        Create cosine learning rate scheduler with 10% warmup.
        """
        num_training_steps = len(self.train_loader) * self.num_epochs // self.gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        print(f"📅 Scheduler Configuration:")
        print(f"   Total training steps: {num_training_steps:,}")
        print(f"   Warmup steps: {num_warmup_steps:,} ({self.warmup_ratio*100:.0f}%)")
        print(f"   Learning rate: {self.learning_rate:.2e} (fixed)")
        print()
        
        return scheduler
    
    def _print_config(self):
        """Print training configuration."""
        print("\n" + "="*80)
        print("🚀 Fine-Tuning Configuration")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Mixed Precision (FP16): {self.use_amp}")
        print(f"\n📦 Data:")
        print(f"   Training samples: {len(self.train_dataset):,}")
        if self.val_dataset:
            print(f"   Validation samples: {len(self.val_dataset):,}")
        print(f"\n🎯 Training:")
        print(f"   Epochs: {self.num_epochs}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"   Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
        print(f"   Learning rate: {self.learning_rate:.2e}")
        print(f"   Weight decay: {self.weight_decay}")
        print(f"   Max gradient norm: {self.max_grad_norm}")
        print(f"\n💾 Checkpointing:")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Save every: {self.save_steps} steps")
        print(f"   Evaluate every: {self.eval_steps} steps")
        print(f"   Log every: {self.logging_steps} steps")
        print("="*80 + "\n")
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}",
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp and self.device.type == "cuda":
                from torch.amp import autocast
                with autocast('cuda'):
                    # Call model with input_ids for PEFT compatibility
                    outputs = self.model(input_ids=input_ids, targets=labels)
                    logits, loss = outputs
            else:
                outputs = self.model(input_ids=input_ids, targets=labels)
                logits, loss = outputs
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'step': self.global_step
                    })
                
                # Evaluation
                if self.val_loader and self.global_step % self.eval_steps == 0:
                    val_loss = self.evaluate()
                    print(f"\n📊 Step {self.global_step} | Val Loss: {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best_model", is_best=True)
                        print(f"✅ New best model saved! (loss: {val_loss:.4f})")
                    
                    self.model.train()
                
                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}")
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        if not self.val_loader:
            return float('inf')
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            if self.use_amp and self.device.type == "cuda":
                from torch.amp import autocast
                with autocast('cuda'):
                    outputs = self.model(input_ids=input_ids, targets=labels)
                    logits, loss = outputs
            else:
                outputs = self.model(input_ids=input_ids, targets=labels)
                logits, loss = outputs
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*80)
        print("🎓 Starting Fine-Tuning")
        print("="*80 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"{'='*80}")
            
            train_loss = self.train_epoch()
            
            print(f"\n📊 Epoch {epoch + 1} Summary:")
            print(f"   Training Loss: {train_loss:.4f}")
            
            if self.val_loader:
                val_loss = self.evaluate()
                print(f"   Validation Loss: {val_loss:.4f}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model", is_best=True)
                    print(f"   ✅ New best model saved!")
            
            # Save epoch checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}")
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("🎉 Fine-Tuning Complete!")
        print("="*80)
        print(f"Total time: {elapsed_time / 3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final model saved to: {self.output_dir}")
        print("="*80 + "\n")
    
    def save_checkpoint(self, name: str, is_best: bool = False):
        """Save model checkpoint and optionally upload to HuggingFace Hub."""
        save_path = self.output_dir / name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights
        self.model.save_pretrained(save_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        torch.save(state, save_path / "training_state.pt")
        
        print(f"💾 Checkpoint saved: {save_path}")
        
        # Upload to HuggingFace Hub if enabled
        # Upload best model always, and epoch checkpoints if requested
        if self.upload_to_hub:
            if is_best or name.startswith("checkpoint_epoch_"):
                self._upload_checkpoint_to_hub(save_path, name)
    
    def _upload_checkpoint_to_hub(self, checkpoint_path: Path, name: str):
        """Upload checkpoint to HuggingFace Hub."""
        try:
            from huggingface_hub import HfApi, create_repo
            
            print(f"\n📤 Uploading to HuggingFace Hub...")
            print(f"   Repository: {self.hub_repo_id}")
            
            api = HfApi()
            
            # Create repo if it doesn't exist
            try:
                create_repo(self.hub_repo_id, repo_type="model", exist_ok=True)
            except Exception:
                pass  # Repo already exists
            
            # Upload all files in checkpoint directory
            files_to_upload = [
                "adapter_config.json",
                "adapter_model.bin",
                "training_state.pt",
                "tokenizer_config.json",
                "tokenizer.json",
            ]
            
            # Use chat_adapter as the name for best_model
            upload_name = "chat_adapter" if name == "best_model" else name
            
            for filename in files_to_upload:
                file_path = checkpoint_path / filename
                if file_path.exists():
                    repo_path = f"{self.hub_path_prefix}{upload_name}/{filename}"
                    print(f"   Uploading {filename}...")
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=repo_path,
                        repo_id=self.hub_repo_id,
                        repo_type="model",
                    )
            
            print(f"✅ Upload complete!")
            print(f"   URL: https://huggingface.co/{self.hub_repo_id}/tree/main/{self.hub_path_prefix}{upload_name}")
            
        except ImportError:
            print("⚠️  huggingface_hub not installed. Skipping upload.")
        except Exception as e:
            print(f"⚠️  Upload failed: {e}")
            print("   Continuing training...")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        load_path = Path(path)
        
        # Load LoRA weights
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, load_path)
        
        # Load training state
        state_path = load_path / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            self.global_step = state['global_step']
            self.current_epoch = state['current_epoch']
            self.best_val_loss = state['best_val_loss']
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
        
        print(f"✅ Checkpoint loaded: {load_path}")
    
    def _load_checkpoint(self, path: str):
        """Internal method to load checkpoint during initialization."""
        print(f"\n📥 Resuming from checkpoint: {path}")
        self.load_checkpoint(path)


# System prompt for Chain-of-Thought reasoning
SYSTEM_PROMPT = """You are a helpful, creative, and clever AI assistant. When a user asks a question, provide a clear and concise answer. If the question involves logic, think through it step-by-step using a 'thought' block. If the user asks for code, provide clean examples in Markdown. Admit if you are unsure of a fact."""
