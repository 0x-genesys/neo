#!/usr/bin/env python3
"""
Quick test to verify training setup works.
"""

import sys
import yaml

print("Testing training setup...")
print("="*80)

# Test 1: Load config
print("\n1. Testing config loading...")
try:
    with open('config/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Config loaded")
except Exception as e:
    print(f"❌ Config loading failed: {e}")
    sys.exit(1)

# Test 2: Import modules
print("\n2. Testing imports...")
try:
    import torch
    from src.model import create_model
    from src.data import load_data
    from src.trainer import Trainer
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 3: Load small dataset
print("\n3. Testing dataset loading...")
try:
    from datasets import load_dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:10]')
    print(f"✅ Dataset loaded: {len(dataset)} examples")
except Exception as e:
    print(f"❌ Dataset loading failed: {e}")
    sys.exit(1)

# Test 4: Create model
print("\n4. Testing model creation...")
try:
    # Use tiny config for testing
    test_config = {
        'model': {
            'vocab_size': 50257,
            'd_model': 128,
            'num_heads': 4,
            'num_layers': 2,
            'context_length': 128,
            'dropout': 0.1
        }
    }
    model = create_model(test_config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {num_params/1e6:.2f}M parameters")
except Exception as e:
    print(f"❌ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Device detection
print("\n5. Testing device detection...")
try:
    from src.device_utils import select_device
    device = select_device('auto', verbose=False)
    print(f"✅ Device selected: {device}")
except Exception as e:
    print(f"❌ Device detection failed: {e}")
    sys.exit(1)

# Test 6: Forward pass
print("\n6. Testing forward pass...")
try:
    model.eval()
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    with torch.no_grad():
        logits, loss = model(input_ids, input_ids)
    print(f"✅ Forward pass successful")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output shape: {logits.shape}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\nYour training setup is working correctly.")
print("\nTo start training:")
print("  bash quick_start.sh")
print("\nOr:")
print("  python train.py --config config/model_config.yaml")
