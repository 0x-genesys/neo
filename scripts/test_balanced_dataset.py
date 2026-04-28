"""
Test script to verify the balanced dataset preparation.
"""

import numpy as np
import tiktoken
from pathlib import Path
import json


def test_binary_dataset(data_dir: str = "data/balanced_300m"):
    """Test the prepared binary dataset."""
    data_dir = Path(data_dir)
    
    print("="*80)
    print("TESTING BALANCED DATASET")
    print("="*80)
    
    # Check files exist
    train_file = data_dir / "train.bin"
    val_file = data_dir / "val.bin"
    stats_file = data_dir / "dataset_stats.json"
    
    print("\n1. Checking files...")
    if not train_file.exists():
        print(f"❌ Training file not found: {train_file}")
        return False
    print(f"✅ Training file exists: {train_file}")
    
    if not val_file.exists():
        print(f"❌ Validation file not found: {val_file}")
        return False
    print(f"✅ Validation file exists: {val_file}")
    
    if not stats_file.exists():
        print(f"⚠️  Statistics file not found: {stats_file}")
    else:
        print(f"✅ Statistics file exists: {stats_file}")
    
    # Load statistics
    print("\n2. Loading statistics...")
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        print("\nDataset Statistics:")
        print("-" * 80)
        for source, data in stats['sources'].items():
            docs = data['docs']
            tokens = data['tokens']
            pct = (tokens / stats['totals']['tokens'] * 100)
            print(f"{source:15s}: {docs:8,} docs | {tokens:12,} tokens ({pct:5.1f}%)")
        print("-" * 80)
        print(f"{'TOTAL':15s}: {stats['totals']['documents']:8,} docs | {stats['totals']['tokens']:12,} tokens")
        print("-" * 80)
    
    # Load binary files
    print("\n3. Loading binary files...")
    train_tokens = np.memmap(train_file, dtype=np.uint32, mode='r')
    val_tokens = np.memmap(val_file, dtype=np.uint32, mode='r')
    
    print(f"✅ Training tokens: {len(train_tokens):,}")
    print(f"✅ Validation tokens: {len(val_tokens):,}")
    
    # Check file sizes
    train_size_gb = train_file.stat().st_size / 1e9
    val_size_gb = val_file.stat().st_size / 1e9
    print(f"\nFile sizes:")
    print(f"  Training: {train_size_gb:.2f} GB")
    print(f"  Validation: {val_size_gb:.2f} GB")
    
    # Validate token ranges
    print("\n4. Validating token ranges...")
    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab
    
    train_min = train_tokens.min()
    train_max = train_tokens.max()
    val_min = val_tokens.min()
    val_max = val_tokens.max()
    
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Training tokens range: [{train_min}, {train_max}]")
    print(f"Validation tokens range: [{val_min}, {val_max}]")
    
    if train_max >= vocab_size:
        print(f"❌ Training tokens exceed vocab size!")
        return False
    if val_max >= vocab_size:
        print(f"❌ Validation tokens exceed vocab size!")
        return False
    
    print("✅ All tokens within valid range")
    
    # Sample and decode
    print("\n5. Sampling and decoding...")
    sample_size = 256
    
    # Sample from training
    train_sample = train_tokens[:sample_size]
    train_text = tokenizer.decode(train_sample.tolist())
    
    print("\nTraining sample (first 256 tokens):")
    print("-" * 80)
    print(train_text[:500] + "..." if len(train_text) > 500 else train_text)
    print("-" * 80)
    
    # Sample from validation
    val_sample = val_tokens[:sample_size]
    val_text = tokenizer.decode(val_sample.tolist())
    
    print("\nValidation sample (first 256 tokens):")
    print("-" * 80)
    print(val_text[:500] + "..." if len(val_text) > 500 else val_text)
    print("-" * 80)
    
    # Calculate sequences
    print("\n6. Calculating training sequences...")
    context_length = 256
    train_sequences = len(train_tokens) // context_length
    val_sequences = len(val_tokens) // context_length
    
    print(f"Context length: {context_length}")
    print(f"Training sequences: {train_sequences:,}")
    print(f"Validation sequences: {val_sequences:,}")
    
    # Calculate training steps
    batch_size = 16
    grad_accum = 8
    epochs = 8
    
    steps_per_epoch = train_sequences // (batch_size * grad_accum)
    total_steps = steps_per_epoch * epochs
    
    print(f"\nTraining configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {grad_accum}")
    print(f"  Effective batch: {batch_size * grad_accum}")
    print(f"  Epochs: {epochs}")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Total steps: {total_steps:,}")
    
    # Verify against expected
    expected_steps = 36621
    if abs(total_steps - expected_steps) > 100:
        print(f"⚠️  Steps mismatch: expected ~{expected_steps:,}, got {total_steps:,}")
    else:
        print(f"✅ Steps match expected: ~{expected_steps:,}")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    
    print("\n💡 Next steps:")
    print("  1. Update config to use this dataset:")
    print("     data:")
    print("       dataset_type: 'binary'")
    print("       train_file: 'data/balanced_300m/train.bin'")
    print("       val_file: 'data/balanced_300m/val.bin'")
    print("  2. Start training:")
    print("     python train.py --config config/gpu_training_117m_1.5gb.yaml")
    
    return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test balanced dataset preparation'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/balanced_300m',
        help='Dataset directory'
    )
    
    args = parser.parse_args()
    
    try:
        success = test_binary_dataset(args.data_dir)
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()
