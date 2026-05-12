"""
Prepare curriculum learning dataset by splitting sources into separate binary files.

This script takes the combined balanced dataset and splits it into separate
binary files for each source (wikitext, ultrachat) to enable curriculum learning.
"""

import numpy as np
import json
from pathlib import Path
import argparse


CURRICULUM_SOURCES = ("wikitext", "ultrachat")


def filter_stats_for_curriculum(stats):
    """Keep only the two Conversational Scholar curriculum sources."""
    sources = {
        source: stats.get('sources', {}).get(source, {'docs': 0, 'tokens': 0})
        for source in CURRICULUM_SOURCES
        if stats.get('sources', {}).get(source, {}).get('tokens', 0) > 0
    }
    total_docs = sum(info.get('docs', 0) for info in sources.values())
    total_tokens = sum(info.get('tokens', 0) for info in sources.values())

    filtered = dict(stats)
    filtered['sources'] = sources
    filtered['totals'] = {
        'documents': total_docs,
        'tokens': total_tokens,
    }
    filtered['targets'] = {
        'wikitext': 240_000_000,
        'ultrachat': 60_000_000,
    }
    return filtered


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
    
    stats = filter_stats_for_curriculum(stats)

    print(f"\nDataset Statistics:")
    for source, info in stats['sources'].items():
        if info['tokens'] > 0:
            print(f"  {source:12s}: {info['tokens']:,} tokens, {info['docs']:,} docs")

    source_names = list(stats['sources'].keys())
    if not source_names:
        raise ValueError("No wikitext or ultrachat sources found in dataset_stats.json")

    source_ranges = {}
    source_files_exist = all((data_dir / f"{source}_train.bin").exists() for source in source_names)

    if source_files_exist:
        print(f"\nUsing existing per-source training files...")
    else:
        train_file = data_dir / "train.bin"
        if not train_file.exists():
            raise FileNotFoundError(f"Train file not found: {train_file}")

        print(f"\nLoading combined dataset: {train_file}")
        tokens = np.memmap(train_file, dtype=np.uint32, mode='r')
        total_tokens = len(tokens)
        print(f"Total tokens: {total_tokens:,}")

        current_pos = 0
        for source in source_names:
            source_tokens = stats['sources'][source]['tokens']
            source_ranges[source] = (current_pos, current_pos + source_tokens)
            current_pos += source_tokens
            print(f"\n{source}:")
            print(f"  Range: {source_ranges[source][0]:,} - {source_ranges[source][1]:,}")
            print(f"  Tokens: {source_tokens:,}")

    print(f"\nWriting curriculum source files...")

    for source in source_names:
        output_file = output_dir / f"{source}_train.bin"
        
        print(f"\nProcessing {source}...")
        print(f"  Output: {output_file}")
        
        source_file = data_dir / f"{source}_train.bin"
        if source_file.exists():
            source_tokens = np.memmap(source_file, dtype=np.uint32, mode='r')
        else:
            start, end = source_ranges[source]
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
    for source in source_names:
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
