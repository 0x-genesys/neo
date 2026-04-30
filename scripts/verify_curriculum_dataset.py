#!/usr/bin/env python3
"""
Verify curriculum dataset files are in the correct location.
"""
from pathlib import Path
import sys

def verify_curriculum_dataset(data_dir="data/balanced_300m_curriculum"):
    """Verify all curriculum dataset files exist."""
    data_dir = Path(data_dir)
    
    print(f"\n{'='*80}")
    print(f"Verifying Curriculum Dataset")
    print(f"{'='*80}")
    print(f"Directory: {data_dir}")
    print()
    
    required_files = [
        "wikitext_train.bin",
        "stack_train.bin",
        "ultrachat_train.bin",
        "val.bin",
        "dataset_stats.json"
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / 1e6
            print(f"✅ {filename:25s} ({size_mb:8.1f} MB)")
        else:
            print(f"❌ {filename:25s} (MISSING)")
            all_exist = False
    
    print(f"{'='*80}")
    
    if all_exist:
        print(f"✅ All curriculum dataset files present!")
        print(f"\nYou can now run training:")
        print(f"   python train.py --config config/auto_training_117m_balanced.yaml")
        return 0
    else:
        print(f"❌ Some files are missing!")
        print(f"\nTo download:")
        print(f"   python -m src.dataset_downloader \\")
        print(f"       0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum \\")
        print(f"       --dataset-name balanced_300m_curriculum")
        return 1

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/balanced_300m_curriculum"
    sys.exit(verify_curriculum_dataset(data_dir))
