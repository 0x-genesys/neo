"""
GPU Fine-Tuning Script (CUDA/MPS Auto-Detection)
Supports NVIDIA CUDA and Apple MPS with Mixed Precision (FP16).
"""
import torch
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model import DecoderOnlyTransformer
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
    parser.add_argument('--upload', action='store_true',
                        help='Upload best model to HuggingFace Hub')
    parser.add_argument('--upload-repo', type=str, default='0x-genesys/neo_weights_checkpoints',
                        help='HuggingFace repository for upload')
    parser.add_argument('--upload-path', type=str, default='finetune/',
                        help='Path prefix in repository (e.g., "finetune/")')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🚀 GPU Fine-Tuning for 117M Transformer with LoRA + CoT")
    print("="*80 + "\n")
    
    # ============================================================================
    # Configuration
    # ============================================================================
    
    # Model configuration (117M parameters)
    MODEL_CONFIG = {
        'vocab_size': 100277,  # tiktoken vocabulary
        'd_model': 768,
        'num_heads': 12,
        'num_layers': 12,
        'context_length': 512,
        'dropout': 0.1,
    }
    
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
        'eval_steps': 100,
        'save_steps': 500,
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
        'max_length': 512,
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
    
    # Update vocab size if tokenizer was extended
    MODEL_CONFIG['vocab_size'] = len(tokenizer)
    
    # ============================================================================
    # Load Pre-trained Model
    # ============================================================================
    
    print("\n" + "="*80)
    print("🔧 Loading Pre-trained Model")
    print("="*80)
    
    # Create model
    model = DecoderOnlyTransformer(
        vocab_size=MODEL_CONFIG['vocab_size'],
        d_model=MODEL_CONFIG['d_model'],
        num_heads=MODEL_CONFIG['num_heads'],
        num_layers=MODEL_CONFIG['num_layers'],
        context_length=MODEL_CONFIG['context_length'],
        dropout=MODEL_CONFIG['dropout'],
    )
    
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
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded pre-trained weights (loss: {checkpoint.get('best_val_loss', 'N/A')})")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded pre-trained weights")
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
    print(f"   1. Load base model: model = DecoderOnlyTransformer(...)")
    print(f"   2. Load LoRA weights: from peft import PeftModel")
    print(f"   3. model = PeftModel.from_pretrained(model, '{CHECKPOINT_CONFIG['output_dir']}/best_model')")
    print(f"   4. Generate: model.generate(...)")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
