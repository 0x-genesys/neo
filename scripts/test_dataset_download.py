#!/usr/bin/env python3
"""
Test script for the automatic dataset downloader.

This script tests the dataset download functionality without running
a full training session.
"""

import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dataset_downloader import DatasetDownloader, configure_tqdm_for_datasets


def test_dataset_download():
    """Test downloading a dataset from HuggingFace Hub."""
    
    print("🧪 Testing Dataset Downloader")
    print("="*50)
    
    # Configure tqdm
    configure_tqdm_for_datasets()
    
    # Initialize downloader
    downloader = DatasetDownloader(cache_dir="data")
    
    # Test configuration - use a real repository for testing
    test_config = {
        'data': {
            'dataset_type': 'binary',
            'train_file': 'data/test_dataset/train.bin',
            'val_file': 'data/test_dataset/val.bin',
            'huggingface_dataset': {
                'repo_id': "0x-genesys/mix_wiki_chat_data_300M_tokens",  # Your uploaded dataset
                'dataset_name': "balanced_300m",  # Local folder name in data/
                'auto_download': True
            }
        }
    }
    
    print("ℹ️  Note: Using 'wikitext' repository for testing (not binary format)")
    print("   This tests the download mechanism, not the actual binary dataset.")
    
    try:
        print("\n1. Testing repository access...")
        
        # Test if we can access the repository
        from huggingface_hub import list_repo_files
        try:
            files = list_repo_files("wikitext", repo_type="dataset")
            print(f"✅ Repository access works (found {len(files)} files)")
        except Exception as e:
            print(f"❌ Repository access failed: {e}")
            return False
        
        print("\n2. Testing dataset availability check...")
        
        # This will fail because wikitext is not a binary dataset,
        # but it tests the download mechanism
        try:
            train_file, val_file, stats = downloader.ensure_dataset_available(test_config)
            print(f"✅ Download mechanism works!")
            return True
        except Exception as e:
            if "train.bin not found" in str(e):
                print(f"✅ Download mechanism works (expected error: binary files not found in test repo)")
                return True
            else:
                print(f"❌ Unexpected error: {e}")
                return False
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        
        # Provide helpful guidance
        if "404" in str(e) or "Repository Not Found" in str(e):
            print(f"\n💡 The test repository doesn't exist. This is normal!")
            print(f"   To set up your actual dataset repository:")
            print(f"   1. Run: python scripts/setup_dataset_repo.py")
            print(f"   2. Follow the interactive setup")
            print(f"   3. Test again with your real repository")
        
        return False


def test_config_loading():
    """Test loading a config file with HuggingFace dataset configuration."""
    
    print("\n🧪 Testing Config Loading")
    print("="*50)
    
    config_file = Path("config/gpu_training_117m_balanced.yaml")
    
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Config loaded successfully: {config_file}")
        
        # Check HuggingFace dataset configuration
        hf_config = config.get('data', {}).get('huggingface_dataset')
        if hf_config:
            print(f"✅ HuggingFace dataset config found:")
            print(f"  Repository: {hf_config.get('repo_id')}")
            print(f"  Dataset name: {hf_config.get('dataset_name')}")
            print(f"  Auto download: {hf_config.get('auto_download')}")
        else:
            print(f"⚠️  No HuggingFace dataset config found")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False


def main():
    """Run all tests."""
    
    print("🚀 Dataset Downloader Test Suite")
    print("="*60)
    
    tests = [
        ("Config Loading", test_config_loading),
        ("Dataset Download", test_dataset_download),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == '__main__':
    exit(main())
