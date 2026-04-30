#!/usr/bin/env python3
"""
Verify curriculum dataset files are properly downloaded and accessible.

This script checks:
1. Dataset directory exists
2. All required curriculum source files exist
3. Files are not empty
4. Files can be read as memory-mapped arrays
5. Dataset statistics are available

Usage:
    python scripts/verify_curriculum_dataset.py [dataset_dir]
    
Example:
    python scripts/verify_curriculum_dataset.py data/balanced_300m_curriculum
"""

import sys
import json
from pathlib import Path
import numpy as np


def format_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def verify_curriculum_dataset(dataset_dir: str = "data/balanced_300m_curriculum"):
    """Verify curriculum dataset is properly set up."""
    
    dataset_path = Path(dataset_dir)
    
    print("="*80)
    print("CURRICULUM DATASET VERIFICATION")
    print("="*80)
    print(f"Dataset directory: {dataset_path.absolute()}")
    print()
    
    # Check if directory exists
    if not dataset_path.exists():
        print(f"❌ ERROR: Dataset directory does not exist: {dataset_path}")
        print(f"\nTo download the dataset:")
        print(f"  python -c \"from src.dataset_downloader import DatasetDownloader; ")
        print(f"  d = DatasetDownloader(); ")
        print(f"  d.download_dataset('0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum', ")
        print(f"  'balanced_300m_curriculum', is_curriculum=True)\"")
        return False
    
    print(f"✅ Dataset directory exists")
    print()
    
    # Check for required files
    required_files = {
        'wikitext_train.bin': 'WikiText training data',
        'stack_train.bin': 'Stack Overflow training data',
        'ultrachat_train.bin': 'UltraChat training data',
        'val.bin': 'Validation data',
        'dataset_stats.json': 'Dataset statistics'
    }
    
    all_files_ok = True
    file_info = {}
    
    print("📋 Checking required files:")
    print("-" * 80)
    
    for filename, description in required_files.items():
        file_path = dataset_path / filename
        
        if not file_path.exists():
            print(f"❌ MISSING: {filename:25s} - {description}")
            all_files_ok = False
            continue
        
        file_size = file_path.stat().st_size
        
        if file_size == 0:
            print(f"❌ EMPTY:   {filename:25s} - {description}")
            all_files_ok = False
            continue
        
        print(f"✅ OK:      {filename:25s} - {format_size(file_size):>10s} - {description}")
        file_info[filename] = file_size
    
    print()
    
    if not all_files_ok:
        print("❌ Some files are missing or empty!")
        print("\nTo fix this issue:")
        print("1. Delete the dataset directory:")
        print(f"   rm -rf {dataset_path}")
        print("2. Re-download the dataset:")
        print(f"   python train.py --config config/auto_training_117m_balanced.yaml")
        return False
    
    # Try to load binary files
    print("🔍 Verifying binary file integrity:")
    print("-" * 80)
    
    binary_files = ['wikitext_train.bin', 'stack_train.bin', 'ultrachat_train.bin', 'val.bin']
    
    for filename in binary_files:
        file_path = dataset_path / filename
        
        try:
            # Try to memory-map the file
            tokens = np.memmap(file_path, dtype=np.uint32, mode='r')
            num_tokens = len(tokens)
            
            # Try to read first and last token
            first_token = tokens[0]
            last_token = tokens[-1]
            
            print(f"✅ {filename:25s} - {num_tokens:,} tokens (first={first_token}, last={last_token})")
            
        except Exception as e:
            print(f"❌ {filename:25s} - ERROR: {e}")
            all_files_ok = False
    
    print()
    
    # Load and display statistics
    stats_file = dataset_path / "dataset_stats.json"
    if stats_file.exists():
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            print("📊 Dataset Statistics:")
            print("-" * 80)
            
            # Print totals
            totals = stats.get('totals', {})
            if totals:
                print(f"Total documents: {totals.get('documents', 'Unknown'):,}")
                print(f"Total tokens:    {totals.get('tokens', 'Unknown'):,}")
            
            # Print tokenizer info
            tokenizer = stats.get('tokenizer', 'Unknown')
            vocab_size = stats.get('vocab_size', 'Unknown')
            print(f"Tokenizer:       {tokenizer}")
            if isinstance(vocab_size, int):
                print(f"Vocabulary size: {vocab_size:,}")
            else:
                print(f"Vocabulary size: {vocab_size}")
            
            # Print source breakdown
            sources = stats.get('sources', {})
            if sources:
                print(f"\n📚 Source Composition:")
                total_tokens = totals.get('tokens', 1)
                for source, source_stats in sources.items():
                    docs = source_stats.get('docs', 0)
                    tokens = source_stats.get('tokens', 0)
                    if tokens > 0:
                        pct = (tokens / total_tokens * 100) if total_tokens > 0 else 0
                        print(f"  {source:15s}: {docs:8,} docs | {tokens:12,} tokens ({pct:5.1f}%)")
            
            print()
            
        except Exception as e:
            print(f"⚠️  Could not load statistics: {e}")
            print()
    
    # Final verdict
    if all_files_ok:
        print("="*80)
        print("✅ VERIFICATION PASSED - Dataset is ready for training!")
        print("="*80)
        print()
        print("You can now start training with:")
        print(f"  python train.py --config config/auto_training_117m_balanced.yaml")
        print()
        return True
    else:
        print("="*80)
        print("❌ VERIFICATION FAILED - Please fix the issues above")
        print("="*80)
        return False


def main():
    """Main entry point."""
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "data/balanced_300m_curriculum"
    
    success = verify_curriculum_dataset(dataset_dir)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
