"""
CPU Fine-Tuning Script
Optimized for CPU training with reduced batch sizes and memory usage.
"""
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model import DecoderOnlyTransformer
from src.tokenizer_utils import load_tokenizer
from src.finetuning.base_trainer import LoRAFineTuner, SYSTEM_PROMPT
from src.finetuning.data_utils import create_cot_dataset, prepare_tokenizer


def main():
    """Main CPU fine-tuning script."""
    print("\n" + "="*80)
    print("🚀 CPU Fine-Tuning for 117M Transformer with LoRA + CoT")
    print("="*80)
    print("\n⚠️  CPU Training Warning:")
    print("   CPU training is significantly slower than GPU training.")
    print("   Consider using a cloud GPU (Colab, Kaggle, AWS, etc.) for faster training.")
    print("   Estimated time: 10-50x slower than GPU\n")
    
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
    
    # Training configuration (CPU-optimized)
    TRAIN_CONFIG = {
        'batch_size': 2,  # Small batch for CPU
        'gradient_accumulation_steps': 16,  # Effective batch = 32
        'num_epochs': 3,
        'learning_rate': 2.0e-5,  # Fixed LR to prevent catastrophic forgetting
        'weight_decay': 0.05,
        'warmup_ratio': 0.1,  # 10% warmup
        'max_grad_norm': 1.0,
        'logging_steps': 5,  # More frequent logging for CPU
        'eval_steps': 200,  # Less frequent eval to save time
        'save_steps': 1000,
    }
    
    # LoRA configuration
    LORA_CONFIG = {
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
    }
    
    # Data paths
    DATA_CONFIG = {
        'train_path': 'data/cot_train.jsonl',
        'val_path': 'data/cot_val.jsonl',
        'max_length': 512,
    }
    
    # Checkpoint paths
    CHECKPOINT_CONFIG = {
        'pretrained_model': 'checkpoints/best_model.pt',  # Pre-trained base model
        'output_dir': 'finetuned_model_cpu',
    }
    
    # ============================================================================
    # Device Setup
    # ============================================================================
    
    device = 'cpu'
    use_amp = False  # No mixed precision for CPU
    
    print(f"\n⚙️  CPU settings:")
    print(f"   Device: CPU")
    print(f"   Threads: {torch.get_num_threads()}")
    print(f"   Mixed precision: {use_amp}")
    print(f"   Batch size: {TRAIN_CONFIG['batch_size']}")
    print(f"   Gradient accumulation: {TRAIN_CONFIG['gradient_accumulation_steps']}")
    print(f"   Effective batch: {TRAIN_CONFIG['batch_size'] * TRAIN_CONFIG['gradient_accumulation_steps']}")
    
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
    
    # Load pre-trained weights
    checkpoint_path = Path(CHECKPOINT_CONFIG['pretrained_model'])
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
    
    trainer = LoRAFineTuner(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=CHECKPOINT_CONFIG['output_dir'],
        device=device,
        use_amp=use_amp,
        **TRAIN_CONFIG,
        **LORA_CONFIG,
    )
    
    # ============================================================================
    # Train
    # ============================================================================
    
    print("\n💡 CPU Training Tips:")
    print("   - Training will take significantly longer than GPU")
    print("   - Consider reducing num_epochs or dataset size for testing")
    print("   - Monitor CPU usage and temperature")
    print("   - Use tmux/screen for long-running sessions")
    print()
    
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
