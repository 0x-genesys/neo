"""
Test remote model loading from HuggingFace Hub.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.remote_model_loader import RemoteModelLoader


def test_list_checkpoints():
    """Test listing available checkpoints."""
    print("\n" + "="*80)
    print("TEST: List Available Checkpoints")
    print("="*80)
    
    loader = RemoteModelLoader(repo_id="0x-genesys/neo_weights_checkpoints")
    checkpoints = loader.list_available_checkpoints()
    
    if checkpoints:
        print(f"✅ Found {len(checkpoints)} checkpoints:")
        for cp in checkpoints:
            print(f"   - {cp}")
        return True
    else:
        print("❌ No checkpoints found")
        return False


def test_download_checkpoint():
    """Test downloading a checkpoint."""
    print("\n" + "="*80)
    print("TEST: Download Checkpoint")
    print("="*80)
    
    loader = RemoteModelLoader(repo_id="0x-genesys/neo_weights_checkpoints")
    
    # List available checkpoints first
    checkpoints = loader.list_available_checkpoints()
    if not checkpoints:
        print("❌ No checkpoints available to test")
        return False
    
    # Try to download the first checkpoint
    test_checkpoint = checkpoints[0]
    print(f"\nTesting download of: {test_checkpoint}")
    
    try:
        local_path = loader.download_checkpoint(test_checkpoint)
        print(f"✅ Successfully downloaded to: {local_path}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def test_load_checkpoint():
    """Test loading a checkpoint."""
    print("\n" + "="*80)
    print("TEST: Load Checkpoint")
    print("="*80)
    
    loader = RemoteModelLoader(repo_id="0x-genesys/neo_weights_checkpoints")
    
    # List available checkpoints first
    checkpoints = loader.list_available_checkpoints()
    if not checkpoints:
        print("❌ No checkpoints available to test")
        return False
    
    # Try to load the first checkpoint
    test_checkpoint = checkpoints[0]
    print(f"\nTesting load of: {test_checkpoint}")
    
    try:
        checkpoint = loader.load_checkpoint(test_checkpoint, map_location='cpu')
        print(f"✅ Successfully loaded checkpoint")
        
        if isinstance(checkpoint, dict):
            print(f"   Checkpoint keys: {list(checkpoint.keys())}")
            if 'epoch' in checkpoint:
                print(f"   Epoch: {checkpoint['epoch']}")
            if 'global_step' in checkpoint:
                print(f"   Global step: {checkpoint['global_step']}")
        
        return True
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("REMOTE MODEL LOADING TESTS")
    print("="*80)
    print("\nTesting HuggingFace Hub integration...")
    print("Repository: 0x-genesys/neo_weights_checkpoints")
    
    results = []
    
    # Test 1: List checkpoints
    results.append(("List Checkpoints", test_list_checkpoints()))
    
    # Test 2: Download checkpoint
    results.append(("Download Checkpoint", test_download_checkpoint()))
    
    # Test 3: Load checkpoint
    results.append(("Load Checkpoint", test_load_checkpoint()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
