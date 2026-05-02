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


def main():
    """Main GPU fine-tuning script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='GPU Fine-Tuning for 117M Transformer with LoRA + CoT')
    
    # Model arguments
    parser.add_argument('--config', type=str, default='config/auto_training_117m_balanced.yaml',
                        help='Path to model config YAML file')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt',
                        help='Path to local pre-trained model checkpoint')
    parser.add_argument('--model-remote', type=str, default=None,
                        help='Checkpoint filename from HuggingFace Hub (e.g., "best_model.pt")')
    parser.add_argument('--model-repo', type=str, default='0x-genesys/neo_weights_checkpoints',
                        help='HuggingFace model repository ID')
    
    # Data arguments
    parser.add_argument('--train-data', type=str, default='data/cot_train.jsonl',
                        help='Path to training data (JSONL)')
    parser.add_argument('--val-data', type=str, default='data/cot_val.jsonl',
                        help='Path to validation data (JSONL)')
    
    # Training arguments
    parser.add_argument('--output-dir', type=str, default='finetuned_model_gpu',
                        help='Output directory for fine-tuned model')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2.0e-5,
                        help='Learning rate')
    
    # Resume arguments
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to local checkpoint to resume from')
    parser.add_argument('--resume-remote', type=str, default=None,
                        help='Checkpoint filename from HuggingFace Hub to resume from')
    
    # Upload arguments
    parser.add_argument('--upload', action='store_true', default=True,
                        help='Upload best model to HuggingFace Hub (default: True)')
    parser.add_argument('--no-upload', dest='upload', action='store_false',
                        help='Disable upload to HuggingFace Hub')
    parser.add_argument('--upload-repo', type=str, default='0x-genesys/neo_weights_checkpoints',
                        help='HuggingFace repository for upload')
    parser.add_argument('--upload-path', type=str, default='finetune/',
                        help='Path prefix in repository (e.g., "finetune/")')
    
    # Evaluation arguments
    parser.add_argument('--eval-steps', type=int, default=None,
                        help='Evaluate every N steps (default: 1000 or once per epoch, whichever is smaller)')
    parser.add_argument('--save-steps', type=int, default=1000,
                        help='Save checkpoint every N steps (default: 1000)')
    
    # Multi-GPU arguments
    parser.add_argument('--multi-gpu', action='store_true',
                        help='Use DataParallel for multi-GPU training')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🚀 GPU Fine-Tuning for 117M Transformer with LoRA + CoT")
    print("="*80 + "\n")
    
    # ============================================================================
    # Load Configuration from YAML
    # ============================================================================
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print(f"   Using default 117M configuration")
        config = {
            'model': {
                'vocab_size': 100277,
                'd_model': 768,
                'num_heads': 12,
                'num_layers': 12,
                'context_length': 512,
                'dropout': 0.1,
            }
        }
    else:
        print(f"📄 Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Config loaded successfully")
    
    # Extract model config
    MODEL_CONFIG = config['model']
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
    
    # Training configuration
    TRAIN_CONFIG = {
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': 4,  # Effective batch = batch_size * 4
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': 0.05,
        'warmup_ratio': 0.1,  # 10% warmup
        'max_grad_norm': 1.0,
        'logging_steps': 10,
        'eval_steps': 1000,  # Will be updated after loading dataset
        'save_steps': save_steps_arg,
    }
    
    # LoRA configuration
    LORA_CONFIG = {
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
    }
    
    # Data paths
    DATA_CONFIG = {
        'train_path': args.train_data,
        'val_path': args.val_data,
        'max_length': MODEL_CONFIG['context_length'],
    }
    
    # Checkpoint paths
    CHECKPOINT_CONFIG = {
        'pretrained_model': args.model,
        'output_dir': args.output_dir,
    }
    
    # ============================================================================
    # Device Setup
    # ============================================================================
    
    device = detect_gpu_device()
    
    # Adjust settings based on device
    if device == 'mps':
        print("\n⚙️  Adjusting settings for MPS:")
        TRAIN_CONFIG['batch_size'] = 4  # Smaller batch for MPS
        TRAIN_CONFIG['gradient_accumulation_steps'] = 8  # Maintain effective batch
        use_amp = False  # Disable AMP for MPS stability
        print(f"   Batch size: {TRAIN_CONFIG['batch_size']}")
        print(f"   Gradient accumulation: {TRAIN_CONFIG['gradient_accumulation_steps']}")
        print(f"   Mixed precision: {use_amp}")
    elif device == 'cuda':
        use_amp = True  # Enable FP16 for CUDA
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
        upload_to_hub=args.upload,
        hub_repo_id=args.upload_repo,
        hub_path_prefix=args.upload_path,
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
