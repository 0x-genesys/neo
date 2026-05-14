"""
GPU Fine-Tuning Script (CUDA/MPS Auto-Detection)
Supports NVIDIA CUDA and Apple MPS with Mixed Precision (FP16).
"""
import torch
import sys
import argparse
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model import create_model
from src.tokenizer_utils import load_tokenizer
from src.finetuning.base_trainer import LoRAFineTuner, SYSTEM_PROMPT
from src.finetuning.data_utils import create_cot_dataset, prepare_tokenizer


def detect_gpu_device():
    """
    Auto-detect GPU device (CUDA or MPS).
    
    Returns:
        str: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        device = 'cuda'
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ CUDA GPU Detected: {device_name}")
        print(f"   Memory: {memory_gb:.2f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print(f"✅ Apple MPS Detected")
        print(f"   Device: Apple Silicon GPU")
    else:
        device = 'cpu'
        print(f"⚠️  No GPU detected, falling back to CPU")
        print(f"   Warning: Training will be significantly slower")
    
    return device


def load_config_from_yaml(config_path: str) -> dict:
    """
    Load configuration from YAML file and merge with CLI defaults.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Merged configuration dictionary
    """
    config = {}
    config_path_obj = Path(config_path)
    
    if config_path_obj.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        print(f"✅ Loaded config from: {config_path}")
    else:
        print(f"⚠️  Config file not found: {config_path}")
        print("   Using built-in defaults")
    
    return config


def get_config_value(config: dict, cli_value, cli_default, *keys):
    """
    Get a value from config, CLI args, or default.
    
    Priority: CLI arg > YAML config > CLI default
    
    Args:
        config: YAML config dictionary
        cli_value: Value from CLI args (None if not provided)
        cli_default: Default value from CLI parser
        *keys: Path to value in nested config dict
        
    Returns:
        Value from highest priority source
    """
    if cli_value is not None:
        return cli_value
    
    # Navigate nested config
    value = config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return cli_default
    
    if value is not None:
        return value
    
    return cli_default


def main():
    """Main GPU fine-tuning script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='GPU Fine-Tuning for 117M Transformer with LoRA + CoT')
    
    # Model arguments
    parser.add_argument('--config', type=str, default='config/finetuning_config.yaml',
                        help='Path to model config YAML file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to local pre-trained model checkpoint')
    parser.add_argument('--model-remote', type=str, default=None,
                        help='Checkpoint filename from HuggingFace Hub (e.g., "best_model.pt")')
    parser.add_argument('--model-repo', type=str, default=None,
                        help='HuggingFace model repository ID')
    
    # Data arguments
    parser.add_argument('--train-data', type=str, default=None,
                        help='Path to training data (JSONL)')
    parser.add_argument('--val-data', type=str, default=None,
                        help='Path to validation data (JSONL)')
    
    # Training arguments
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for fine-tuned model')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    
    # Resume arguments
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to local checkpoint to resume from')
    parser.add_argument('--resume-remote', type=str, default=None,
                        help='Checkpoint filename from HuggingFace Hub to resume from')
    
    # Upload arguments
    parser.add_argument('--upload', action='store_true', default=None,
                        help='Upload best model to HuggingFace Hub (default: True)')
    parser.add_argument('--no-upload', dest='upload', action='store_false',
                        help='Disable upload to HuggingFace Hub')
    parser.add_argument('--upload-repo', type=str, default=None,
                        help='HuggingFace repository for upload')
    parser.add_argument('--upload-path', type=str, default=None,
                        help='Path prefix in repository (e.g., "finetune/")')
    
    # Evaluation arguments
    parser.add_argument('--eval-steps', type=int, default=None,
                        help='Evaluate every N steps (default: 1000 or once per epoch, whichever is smaller)')
    parser.add_argument('--save-steps', type=int, default=None,
                        help='Save checkpoint every N steps (default: 1000)')
    
    # Multi-GPU arguments
    parser.add_argument('--multi-gpu', action='store_true',
                        help='Use DataParallel for multi-GPU training')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🚀 GPU Fine-Tuning for Transformer with LoRA + CoT")
    print("="*80 + "\n")
    
    # ============================================================================
    # Load Configuration from YAML
    # ============================================================================
    
    config = load_config_from_yaml(args.config)
    
    # Extract model config
    MODEL_CONFIG = config.get('model', {})
    if not MODEL_CONFIG:
        MODEL_CONFIG = {
            'vocab_size': 100277,
            'd_model': 768,
            'num_heads': 12,
            'num_layers': 12,
            'context_length': 512,
            'dropout': 0.1,
        }
    
    # Fill in missing model config values
    MODEL_CONFIG['vocab_size'] = MODEL_CONFIG.get('vocab_size', 100277)
    MODEL_CONFIG['d_model'] = MODEL_CONFIG.get('d_model', 768)
    MODEL_CONFIG['num_heads'] = MODEL_CONFIG.get('num_heads', 12)
    MODEL_CONFIG['num_layers'] = MODEL_CONFIG.get('num_layers', 12)
    MODEL_CONFIG['context_length'] = MODEL_CONFIG.get('context_length', 512)
    MODEL_CONFIG['dropout'] = MODEL_CONFIG.get('dropout', 0.1)
    
    # Extract training config
    TRAIN_YAML = config.get('training', {})
    
    # Extract LoRA config
    LORA_YAML = config.get('lora', {})
    
    # Extract data config
    DATA_YAML = config.get('data', {})
    
    # Extract checkpoint config
    CHECKPOINT_YAML = config.get('checkpoint', {})
    
    # Extract generation config
    GENERATION_YAML = config.get('generation', {})
    
    # Extract device config
    DEVICE_YAML = config.get('device', {})
    
    # Extract logging config
    LOGGING_YAML = config.get('logging', {})
    
    # Extract advanced config
    ADVANCED_YAML = config.get('advanced', {})
    
    print(f"\n📊 Model Configuration:")
    print(f"   Vocabulary: {MODEL_CONFIG['vocab_size']:,}")
    print(f"   Dimensions: {MODEL_CONFIG['d_model']}")
    print(f"   Heads: {MODEL_CONFIG['num_heads']}")
    print(f"   Layers: {MODEL_CONFIG['num_layers']}")
    print(f"   Context: {MODEL_CONFIG['context_length']}")
    print(f"   Dropout: {MODEL_CONFIG['dropout']}")
    
    # Calculate smart eval_steps default: 1000 or once per epoch, whichever is smaller
    # We'll calculate this after loading the dataset
    eval_steps_arg = args.eval_steps
    save_steps_arg = args.save_steps
    
    # Training configuration - use YAML values when CLI args not provided
    TRAIN_CONFIG = {
        'batch_size': get_config_value(TRAIN_YAML, args.batch_size, 8, 'batch_size'),
        'gradient_accumulation_steps': get_config_value(TRAIN_YAML, None, 4, 'gradient_accumulation_steps'),
        'num_epochs': get_config_value(TRAIN_YAML, args.epochs, 3, 'num_epochs'),
        'learning_rate': get_config_value(TRAIN_YAML, args.lr, 2.0e-5, 'learning_rate'),
        'weight_decay': get_config_value(TRAIN_YAML, None, 0.05, 'weight_decay'),
        'warmup_ratio': get_config_value(TRAIN_YAML, None, 0.1, 'warmup_ratio'),
        'max_grad_norm': get_config_value(TRAIN_YAML, None, 1.0, 'max_grad_norm'),
        'logging_steps': get_config_value(TRAIN_YAML, None, 10, 'logging_steps'),
        'eval_steps': get_config_value(TRAIN_YAML, None, 1000, 'eval_steps'),
        'save_steps': get_config_value(TRAIN_YAML, save_steps_arg, 500, 'save_steps'),
    }
    
    # LoRA configuration - use YAML values when CLI args not provided
    LORA_CONFIG = {
        'lora_r': get_config_value(LORA_YAML, None, 16, 'r'),
        'lora_alpha': get_config_value(LORA_YAML, None, 32, 'alpha'),
        'lora_dropout': get_config_value(LORA_YAML, None, 0.1, 'dropout'),
        'target_modules': get_config_value(LORA_YAML, None, None, 'target_modules'),
    }
    
    # Data paths - use YAML values when CLI args not provided
    DATA_CONFIG = {
        'train_path': get_config_value(DATA_YAML, args.train_data, 'data/factual/factual_train.jsonl', 'train_path'),
        'val_path': get_config_value(DATA_YAML, args.val_data, 'data/factual/factual_val.jsonl', 'val_path'),
        'max_length': get_config_value(DATA_YAML, None, MODEL_CONFIG['context_length'], 'max_length'),
    }
    
    # Checkpoint paths - use YAML values when CLI args not provided
    CHECKPOINT_CONFIG = {
        'pretrained_model': get_config_value(CHECKPOINT_YAML, args.model, 'checkpoints/best_model.pt', 'pretrained_model'),
        'output_dir': get_config_value(CHECKPOINT_YAML, args.output_dir, 'finetuned_model', 'output_dir'),
    }
    
    # ============================================================================
    # Device Setup
    # ============================================================================
    
    device = detect_gpu_device()
    
    # Adjust settings based on device (YAML overrides device-specific settings)
    device_config = DEVICE_YAML.get(device, {}) if DEVICE_YAML else {}
    
    if device == 'mps':
        print("\n⚙️  Adjusting settings for MPS:")
        # Use YAML values if available, otherwise defaults
        mps_batch_size = get_config_value(device_config, None, 4, 'batch_size')
        mps_grad_accum = get_config_value(device_config, None, 8, 'gradient_accumulation_steps')
        mps_use_amp = get_config_value(device_config, None, False, 'use_amp')
        
        TRAIN_CONFIG['batch_size'] = mps_batch_size
        TRAIN_CONFIG['gradient_accumulation_steps'] = mps_grad_accum
        use_amp = mps_use_amp
        print(f"   Batch size: {TRAIN_CONFIG['batch_size']}")
        print(f"   Gradient accumulation: {TRAIN_CONFIG['gradient_accumulation_steps']}")
        print(f"   Mixed precision: {use_amp}")
    elif device == 'cuda':
        # Use YAML values if available, otherwise defaults
        cuda_batch_size = get_config_value(device_config, None, 8, 'batch_size')
        cuda_grad_accum = get_config_value(device_config, None, 4, 'gradient_accumulation_steps')
        cuda_use_amp = get_config_value(device_config, None, True, 'use_amp')
        
        TRAIN_CONFIG['batch_size'] = cuda_batch_size
        TRAIN_CONFIG['gradient_accumulation_steps'] = cuda_grad_accum
        use_amp = cuda_use_amp
        print(f"\n⚙️  CUDA settings:")
        print(f"   Mixed precision (FP16): {use_amp}")
        print(f"   Batch size: {TRAIN_CONFIG['batch_size']}")
    else:
        print("\n⚠️  CPU training not recommended for fine-tuning")
        print("   Please use gpu_finetune.py on a GPU or cpu_finetune.py for CPU")
        return
    
    # ============================================================================
    # Load Tokenizer
    # ============================================================================
    
    print("\n" + "="*80)
    print("📚 Loading Tokenizer")
    print("="*80)
    
    tokenizer = load_tokenizer()
    tokenizer = prepare_tokenizer(tokenizer)
    
    # CRITICAL: Verify special tokens are encoded as single tokens
    print(f"\n🔍 Verifying Special Token Encoding...")
    from src.finetuning.data_utils import SPECIAL_TOKENS
    
    im_start_tokens = tokenizer.encode(SPECIAL_TOKENS['im_start'])
    im_end_tokens = tokenizer.encode(SPECIAL_TOKENS['im_end'])
    
    print(f"   <|im_start|> encoded as: {im_start_tokens} ({len(im_start_tokens)} token{'s' if len(im_start_tokens) != 1 else ''})")
    print(f"   <|im_end|> encoded as: {im_end_tokens} ({len(im_end_tokens)} token{'s' if len(im_end_tokens) != 1 else ''})")
    
    if len(im_start_tokens) != 1 or len(im_end_tokens) != 1:
        print(f"\n❌ CRITICAL ERROR: Special tokens are being split!")
        print(f"   <|im_start|> should be 1 token, got {len(im_start_tokens)}")
        print(f"   <|im_end|> should be 1 token, got {len(im_end_tokens)}")
        print(f"\n   This will cause training to fail.")
        print(f"   The tokenizer must use allowed_special='all' when encoding.")
        print(f"   Check src/tokenizer_utils.py TiktokenWrapper.encode() method.")
        raise ValueError("Special tokens are being split into multiple tokens")
    
    print(f"✅ Special tokens verified as single tokens")
    print(f"   <|im_start|> = token ID {im_start_tokens[0]}")
    print(f"   <|im_end|> = token ID {im_end_tokens[0]}")
    
    # Get actual tokenizer vocab size
    tokenizer_vocab_size = len(tokenizer)
    model_vocab_size = MODEL_CONFIG['vocab_size']
    
    print(f"\n📊 Vocabulary Size Check:")
    print(f"   Tokenizer vocab size: {tokenizer_vocab_size}")
    print(f"   Model config vocab size: {model_vocab_size}")
    
    if tokenizer_vocab_size != model_vocab_size:
        print(f"\n❌ CRITICAL: Vocab size mismatch!")
        print(f"   The model was trained with vocab_size={model_vocab_size}")
        print(f"   But tokenizer has vocab_size={tokenizer_vocab_size}")
        print(f"\n   This will cause CUDA assertion errors during training.")
        print(f"   The checkpoint must match the tokenizer vocab size.")
        print(f"\n   Options:")
        print(f"   1. Use a checkpoint trained with vocab_size={tokenizer_vocab_size}")
        print(f"   2. Or update the tokenizer to match the checkpoint")
        return
    
    print(f"✅ Vocab sizes match!")
    
    # ============================================================================
    # Load Pre-trained Model
    # ============================================================================
    
    print("\n" + "="*80)
    print("🔧 Loading Pre-trained Model")
    print("="*80)
    
    # Create model using the factory function from model.py
    model = create_model(config)
    
    # Determine checkpoint path (local or remote)
    if args.model_remote:
        # Download from HuggingFace Hub
        from src.remote_model_loader import get_remote_checkpoint_path
        print(f"\n📥 Loading model from HuggingFace Hub...")
        print(f"   Repository: {args.model_repo}")
        print(f"   File: {args.model_remote}")
        checkpoint_path = Path(get_remote_checkpoint_path(args.model_remote, args.model_repo))
    else:
        checkpoint_path = Path(CHECKPOINT_CONFIG['pretrained_model'])
    
    # Load pre-trained weights
    if checkpoint_path.exists():
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"✅ Loaded pre-trained weights (loss: {checkpoint.get('best_val_loss', 'N/A')})")
        else:
            state_dict = checkpoint
            print(f"✅ Loaded pre-trained weights")
        
        # Check vocab size in checkpoint
        if 'token_embedding.weight' in state_dict:
            checkpoint_vocab_size = state_dict['token_embedding.weight'].shape[0]
            print(f"\n📊 Checkpoint vocab size: {checkpoint_vocab_size}")
            print(f"   Model config vocab size: {MODEL_CONFIG['vocab_size']}")
            print(f"   Tokenizer vocab size: {tokenizer_vocab_size}")
            
            if checkpoint_vocab_size != MODEL_CONFIG['vocab_size']:
                print(f"\n⚠️  Checkpoint vocab size ({checkpoint_vocab_size}) != config ({MODEL_CONFIG['vocab_size']})")
                print(f"   Updating model config to match checkpoint...")
                MODEL_CONFIG['vocab_size'] = checkpoint_vocab_size
                
                # Recreate model with correct vocab size
                model = create_model(config)
        
        # Load state dict
        model.load_state_dict(state_dict)
        print(f"✅ Model weights loaded successfully")
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print(f"   Starting from random initialization")
    
    # ============================================================================
    # Load Data
    # ============================================================================
    
    print("\n" + "="*80)
    print("📊 Loading Training Data")
    print("="*80)
    
    # Check if data files exist
    train_path = Path(DATA_CONFIG['train_path'])
    val_path = Path(DATA_CONFIG['val_path'])
    
    if not train_path.exists():
        print(f"❌ Training data not found: {train_path}")
        print(f"\n💡 To create sample data, run:")
        print(f"   python -c \"from src.finetuning.data_utils import create_sample_cot_data; create_sample_cot_data('{train_path}', 1000)\"")
        return
    
    # Create datasets
    train_dataset, val_dataset = create_cot_dataset(
        train_path=str(train_path),
        val_path=str(val_path) if val_path.exists() else None,
        tokenizer=tokenizer,
        max_length=DATA_CONFIG['max_length'],
        system_prompt=SYSTEM_PROMPT,
    )
    
    # Calculate smart eval_steps: 1000 or once per epoch, whichever is smaller
    steps_per_epoch = len(train_dataset) // (TRAIN_CONFIG['batch_size'] * TRAIN_CONFIG['gradient_accumulation_steps'])
    if eval_steps_arg is not None:
        # User specified eval_steps
        TRAIN_CONFIG['eval_steps'] = eval_steps_arg
    else:
        # Smart default: min(1000, steps_per_epoch)
        TRAIN_CONFIG['eval_steps'] = min(1000, max(steps_per_epoch, 1))
    
    print(f"\n📊 Training Schedule:")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Evaluate every: {TRAIN_CONFIG['eval_steps']} steps")
    print(f"   Save checkpoint every: {TRAIN_CONFIG['save_steps']} steps")
    
    # ============================================================================
    # Initialize Trainer
    # ============================================================================
    
    print("\n" + "="*80)
    print("🎯 Initializing LoRA Fine-Tuner")
    print("="*80)
    
    # Handle resume from checkpoint
    resume_from = None
    if args.resume:
        resume_from = args.resume
        print(f"📥 Resuming from local checkpoint: {resume_from}")
    elif args.resume_remote:
        from src.remote_model_loader import get_remote_checkpoint_path
        print(f"\n📥 Resuming from remote checkpoint: {args.resume_remote}")
        resume_from = get_remote_checkpoint_path(args.resume_remote, args.model_repo)
        print(f"✅ Downloaded to: {resume_from}")
    
    # Handle upload settings from YAML
    logging_config = LOGGING_YAML if LOGGING_YAML else {}
    upload_to_hub = get_config_value(logging_config, args.upload, False, 'use_wandb')
    hub_repo_id = get_config_value(logging_config, args.upload_repo, '0x-genesys/neo_weights_checkpoints', 'wandb_project')
    hub_path_prefix = get_config_value(logging_config, args.upload_path, 'finetune/', 'log_dir')
    
    # CLI args override YAML
    if args.upload is not None:
        upload_to_hub = args.upload
    if args.upload_repo is not None:
        hub_repo_id = args.upload_repo
    if args.upload_path is not None:
        hub_path_prefix = args.upload_path
    
    trainer = LoRAFineTuner(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=CHECKPOINT_CONFIG['output_dir'],
        device=device,
        use_amp=use_amp,
        use_multi_gpu=args.multi_gpu,
        resume_from_checkpoint=resume_from,
        upload_to_hub=upload_to_hub,
        hub_repo_id=hub_repo_id,
        hub_path_prefix=hub_path_prefix,
        **TRAIN_CONFIG,
        **LORA_CONFIG,
    )
    
    # ============================================================================
    # Train
    # ============================================================================
    
    trainer.train()
    
    # ============================================================================
    # Summary
    # ============================================================================
    
    print("\n" + "="*80)
    print("✅ Fine-Tuning Complete!")
    print("="*80)
    print(f"\n📁 Model saved to: {CHECKPOINT_CONFIG['output_dir']}")
    print(f"\n🚀 To use the fine-tuned model:")
    print(f"   1. Load base model:")
    print(f"      from src.model import create_model")
    print(f"      import yaml")
    print(f"      with open('{args.config}', 'r') as f:")
    print(f"          config = yaml.safe_load(f)")
    print(f"      model = create_model(config)")
    print(f"   2. Load LoRA weights:")
    print(f"      from peft import PeftModel")
    print(f"      model = PeftModel.from_pretrained(model, '{CHECKPOINT_CONFIG['output_dir']}/best_model')")
    print(f"   3. Generate: model.generate(...)")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
