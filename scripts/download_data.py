"""
Script to download and cache datasets from HuggingFace.
"""
from datasets import load_dataset
import argparse


def download_dataset(dataset_name, config_name=None, cache_dir=None):
    """Download and cache a dataset."""
    print(f"Downloading dataset: {dataset_name}")
    if config_name:
        print(f"Config: {config_name}")
    
    try:
        if config_name:
            dataset = load_dataset(dataset_name, config_name, cache_dir=cache_dir)
        else:
            dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        
        print("\nDataset downloaded successfully!")
        print(f"Splits: {list(dataset.keys())}")
        
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data)} examples")
        
        # Show sample
        if 'train' in dataset:
            print("\nSample from train split:")
            print(dataset['train'][0])
        
        return dataset
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Download HuggingFace datasets')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--config', type=str, help='Dataset config name')
    parser.add_argument('--cache-dir', type=str, help='Cache directory')
    
    args = parser.parse_args()
    
    download_dataset(args.dataset, args.config, args.cache_dir)


if __name__ == '__main__':
    # Popular datasets for language modeling
    print("="*80)
    print("Popular Datasets for Language Modeling:")
    print("="*80)
    print("\n1. WikiText (Small to Medium)")
    print("   - wikitext-2-raw-v1: ~2M tokens")
    print("   - wikitext-103-raw-v1: ~100M tokens")
    print("   Command: python scripts/download_data.py --dataset wikitext --config wikitext-2-raw-v1")
    
    print("\n2. OpenWebText (Large)")
    print("   - ~8M documents, ~40GB")
    print("   Command: python scripts/download_data.py --dataset openwebtext")
    
    print("\n3. BookCorpus (Large)")
    print("   - ~11K books")
    print("   Command: python scripts/download_data.py --dataset bookcorpus")
    
    print("\n4. Tiny Shakespeare (Tiny - for testing)")
    print("   - ~1MB, Shakespeare's works")
    print("   Command: python scripts/download_data.py --dataset tiny_shakespeare")
    
    print("\n5. C4 (Colossal Clean Crawled Corpus)")
    print("   - Massive web corpus")
    print("   Command: python scripts/download_data.py --dataset c4 --config en")
    
    print("\n6. The Pile (Very Large)")
    print("   - 825GB diverse text")
    print("   Command: python scripts/download_data.py --dataset EleutherAI/pile")
    
    print("\n" + "="*80)
    print("\nNote: Some datasets require authentication.")
    print("Set your HuggingFace token: huggingface-cli login")
    print("="*80 + "\n")
    
    main()
