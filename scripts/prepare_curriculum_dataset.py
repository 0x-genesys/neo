"""
Prepare curriculum learning dataset by splitting sources into separate binary files.

This script takes the combined balanced dataset and splits it into separate
binary files for each source (wikitext, stack, ultrachat) to enable curriculum learning.
"""

import numpy as np
import json
from pathlib import Path
import argparse


def split_dataset_by_source(data_dir, output_dir):
    """
    Split combined dataset into separate source files.
    
    Args:
        data_dir: Directory containing the combined dataset
        output_dir: Directory to save split datasets
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Curriculum Dataset Preparation")
    print("="*80)
    
    # Load dataset stats
    stats_file = data_dir / "dataset_stats.json"
    if not stats_file.exists():
        raise FileNotFoundError(f"Dataset stats not found: {stats_file}")
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    print(f"\nDataset Statistics:")
    for source, info in stats['sources'].items():
        if info['tokens'] > 0:
            print(f"  {source:12s}: {info['tokens']:,} tokens, {info['docs']:,} docs")
    
    # Load combined train file
    train_file = data_dir / "train.bin"
    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")
    
    print(f"\nLoading combined dataset: {train_file}")
    tokens = np.memmap(train_file, dtype=np.uint32, mode='r')
    total_tokens = len(tokens)
    print(f"Total tokens: {total_tokens:,}")
    
    # Calculate token ranges for each source based on stats
    sources = stats['sources']
    source_ranges = {}
    current_pos = 0
    
    for source, info in sources.items():
        if info['tokens'] > 0:
            source_tokens = info['tokens']
            source_ranges[source] = (current_pos, current_pos + source_tokens)
            current_pos += source_tokens
            print(f"\n{source}:")
            print(f"  Range: {source_ranges[source][0]:,} - {source_ranges[source][1]:,}")
            print(f"  Tokens: {source_tokens:,}")
    
    # Split and save each source
    print(f"\nSplitting dataset into separate source files...")
    
    for source, (start, end) in source_ranges.items():
        output_file = output_dir / f"{source}_train.bin"
        
        print(f"\nProcessing {source}...")
        print(f"  Output: {output_file}")
        
        # Extract source tokens
        source_tokens = tokens[start:end]
        
        # Save as new binary file
        with open(output_file, 'wb') as f:
            source_tokens.tofile(f)
        
        # Verify
        verify_tokens = np.memmap(output_file, dtype=np.uint32, mode='r')
        print(f"  Saved: {len(verify_tokens):,} tokens")
        print(f"  ✅ Verified")
    
    # Copy validation file (not split by source)
    val_file = data_dir / "val.bin"
    if val_file.exists():
        output_val = output_dir / "val.bin"
        print(f"\nCopying validation file...")
        print(f"  From: {val_file}")
        print(f"  To: {output_val}")
        
        val_tokens = np.memmap(val_file, dtype=np.uint32, mode='r')
        with open(output_val, 'wb') as f:
            val_tokens.tofile(f)
        
        print(f"  ✅ Copied {len(val_tokens):,} tokens")
    
    # Save updated stats
    output_stats = output_dir / "dataset_stats.json"
    with open(output_stats, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n✅ Saved stats: {output_stats}")
    
    print(f"\n{'='*80}")
    print(f"✅ Curriculum dataset preparation complete!")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Files created:")
    for source in source_ranges.keys():
        print(f"  - {source}_train.bin")
    if val_file.exists():
        print(f"  - val.bin")
    print(f"  - dataset_stats.json")
    
    print(f"\nTo use curriculum learning:")
    print(f"1. Update config to point to: {output_dir}")
    print(f"2. Enable curriculum_learning in training config")
    print(f"3. Run: python train.py --config config/gpu_training_117m_balanced.yaml")


def main():
    parser = argparse.ArgumentParser(description='Prepare curriculum learning dataset')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/balanced_300m',
        help='Directory containing combined dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/balanced_300m_curriculum',
        help='Output directory for split datasets'
    )
    
    args = parser.parse_args()
    
    split_dataset_by_source(args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()
