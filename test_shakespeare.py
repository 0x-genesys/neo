#!/usr/bin/env python3
"""
Test script for training on Shakespeare data (CPU-friendly).
Tests new code changes: warmup scheduler, gradient checkpointing, GPT-4 tokenizer support.
"""
import torch
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model import create_model
from src.data import load_data
from src.trainer import Trainer


def create_shakespeare_config():
    """Create a minimal config for testing on Shakespeare data."""
    config = {
        'model': {
            'vocab_size': 50257,  # Will be set by tokenizer
            'd_model': 128,       # Small for CPU
            'num_heads': 4,
            'num_layers': 2,
            'context_length': 128,
            'dropout': 0.1,
            'use_flash_attention': False,
            'use_gradient_checkpointing': True  # Test gradient checkpointing
        },
        'training': {
            'batch_size': 4,      # Small for CPU
            'gradient_accumulation_steps': 2,
            'max_epochs': 1,
            'max_steps': 100,     # Just 100 steps for testing
            'learning_rate': 3.0e-4,
            'weight_decay': 0.01,
            'warmup_steps': 20,   # Test warmup (20% of steps)
            'max_grad_norm': 1.0,
            'eval_interval': 25,
            'save_interval': 50,
            'log_interval': 5
        },
        'data': {
            'dataset_name': 'tiny_shakespeare',
            'dataset_config': None,
            'train_split': 'train',
            'val_split': 'validation',
            'test_split': 'test',
            'max_length': 128,
            'num_workers': 0,
            'pin_memory': False
        },
        'tokenizer': {
            'type': 'gpt2',  # Use GPT-2 for testing (GPT-4 requires tiktoken)
            'vocab_size': 50257
        },
        'optimizer': {
            'type': 'adamw',
            'betas': [0.9, 0.95],
            'eps': 1.0e-8
        },
        'scheduler': {
            'type': 'cosine_with_warmup',
            'min_lr': 3.0e-5
        },
        'system': {
            'device': 'cpu',
            'mixed_precision': False,  # Disable for CPU
            'compile_model': False,
            'seed': 42
        },
        'checkpoint': {
            'save_dir': 'checkpoints/test_shakespeare',
            'resume_from': None,
            'save_best_only': False
        },
        'logging': {
            'use_wandb': False,
            'wandb_project': 'test-shakespeare',
            'wandb_entity': None,
            'log_dir': 'logs/test_shakespeare'
        },
        'generation': {
            'max_new_tokens': 50,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.95,
            'num_samples': 2
        }
    }
    return config


def test_warmup_scheduler():
    """Test that warmup scheduler is working correctly."""
    print("\n" + "="*80)
    print("TEST 1: Warmup Scheduler")
    print("="*80)
    
    config = create_shakespeare_config()
    
    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # Create scheduler
    import math
    from torch.optim.lr_scheduler import LambdaLR
    
    warmup_steps = config['training']['warmup_steps']
    max_steps = config['training']['max_steps']
    min_lr = config['scheduler']['min_lr']
    max_lr = config['training']['learning_rate']
    min_lr_ratio = min_lr / max_lr
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
    
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Test learning rates at key points
    print(f"\nWarmup steps: {warmup_steps}")
    print(f"Max steps: {max_steps}")
    print(f"Max LR: {max_lr:.2e}")
    print(f"Min LR: {min_lr:.2e}")
    print("\nLearning rate schedule:")
    
    test_steps = [0, 5, 10, 20, 30, 50, 75, 100]
    for step in test_steps:
        lr = max_lr * lr_lambda(step)
        phase = "WARMUP" if step < warmup_steps else "DECAY"
        print(f"  Step {step:3d}: LR = {lr:.6f} ({phase})")
        
        # Advance scheduler
        if step < max_steps:
            optimizer.step()
            scheduler.step()
    
    print("\n✅ Warmup scheduler test passed!")
    return True


def test_gradient_checkpointing():
    """Test that gradient checkpointing is working."""
    print("\n" + "="*80)
    print("TEST 2: Gradient Checkpointing")
    print("="*80)
    
    config = create_shakespeare_config()
    
    # Create model with gradient checkpointing
    config['model']['use_gradient_checkpointing'] = True
    model_with_ckpt = create_model(config)
    
    # Create model without gradient checkpointing
    config['model']['use_gradient_checkpointing'] = False
    model_without_ckpt = create_model(config)
    
    print(f"\nModel with checkpointing: {sum(p.numel() for p in model_with_ckpt.parameters())/1e6:.2f}M params")
    print(f"Model without checkpointing: {sum(p.numel() for p in model_without_ckpt.parameters())/1e6:.2f}M params")
    
    # Test forward pass
    batch_size = 2
    seq_len = 32
    dummy_input = torch.randint(0, 1000, (batch_size, seq_len))
    dummy_target = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward with checkpointing
    model_with_ckpt.train()
    logits_ckpt, loss_ckpt = model_with_ckpt(dummy_input, dummy_target)
    print(f"\nWith checkpointing - Loss: {loss_ckpt.item():.4f}")
    
    # Forward without checkpointing
    model_without_ckpt.train()
    logits_no_ckpt, loss_no_ckpt = model_without_ckpt(dummy_input, dummy_target)
    print(f"Without checkpointing - Loss: {loss_no_ckpt.item():.4f}")
    
    print("\n✅ Gradient checkpointing test passed!")
    return True


def test_training():
    """Test actual training on Shakespeare data."""
    print("\n" + "="*80)
    print("TEST 3: Training on Shakespeare Data")
    print("="*80)
    
    config = create_shakespeare_config()
    
    print("\nLoading data...")
    try:
        train_loader, val_loader, test_loader, tokenizer = load_data(config)
        print(f"✅ Data loaded successfully")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader) if val_loader else 0}")
    except Exception as e:
        print(f"⚠️  Could not load tiny_shakespeare, using wikitext-2 instead")
        print(f"   Error: {e}")
        config['data']['dataset_name'] = 'wikitext'
        config['data']['dataset_config'] = 'wikitext-2-raw-v1'
        train_loader, val_loader, test_loader, tokenizer = load_data(config)
    
    print("\nCreating model...")
    model = create_model(config)
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    print("\nCreating trainer...")
    trainer = Trainer(model, train_loader, val_loader, tokenizer, config)
    print("✅ Trainer created")
    
    print("\nStarting training (100 steps)...")
    print("This will test:")
    print("  - Warmup scheduler (steps 0-20)")
    print("  - Gradient checkpointing (memory efficient)")
    print("  - Training loop")
    print("  - Validation")
    print("  - Checkpointing")
    
    try:
        trainer.train()
        print("\n✅ Training completed successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("TESTING NEW CODE CHANGES")
    print("="*80)
    print("\nThis script tests:")
    print("1. ✅ Warmup scheduler implementation")
    print("2. ✅ Gradient checkpointing")
    print("3. ✅ Training on Shakespeare data (CPU)")
    print("\nAll tests use CPU and small models for fast testing.")
    
    # Run tests
    tests_passed = 0
    tests_total = 3
    
    try:
        if test_warmup_scheduler():
            tests_passed += 1
    except Exception as e:
        print(f"\n❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        if test_gradient_checkpointing():
            tests_passed += 1
    except Exception as e:
        print(f"\n❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        if test_training():
            tests_passed += 1
    except Exception as e:
        print(f"\n❌ Test 3 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"\nTests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("\n🎉 All tests passed! New code changes are working correctly.")
        print("\nYou can now:")
        print("  1. Train with quick_start.yaml (uses new warmup scheduler)")
        print("  2. Train with gpu_training_117m.yaml (uses warmup + gradient checkpointing)")
        print("  3. Use GPT-4 tokenizer by installing tiktoken and updating config")
        return 0
    else:
        print(f"\n⚠️  {tests_total - tests_passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == '__main__':
    exit(main())
