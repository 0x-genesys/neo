"""
Test script to verify HuggingFace Hub upload integration.
Creates a small test file and uploads it to verify credentials and permissions.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

def test_hf_hub_upload():
    """Test HuggingFace Hub upload functionality."""
    
    print("="*80)
    print("HUGGINGFACE HUB UPLOAD TEST")
    print("="*80)
    print()
    
    # Step 1: Check if huggingface_hub is installed
    print("Step 1: Checking huggingface_hub installation...")
    try:
        from huggingface_hub import HfApi
        print("✅ huggingface_hub is installed")
    except ImportError:
        print("❌ huggingface_hub not installed")
        print("   Install with: pip install huggingface_hub")
        return False
    
    print()
    
    # Step 2: Check authentication
    print("Step 2: Checking authentication...")
    try:
        api = HfApi()
        user_info = api.whoami()
        username = user_info['name']
        print(f"✅ Authenticated as: {username}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("   Login with: huggingface-cli login")
        return False
    
    print()
    
    # Step 3: Check repository access
    print("Step 3: Checking repository access...")
    repo_id = "0x-genesys/neo_weights_checkpoints"
    
    try:
        # Try to list files in the repo
        files = api.list_repo_files(repo_id, repo_type="model")
        print(f"✅ Repository accessible: {repo_id}")
        print(f"   Current files: {len(files)}")
        if files:
            print(f"   Latest files:")
            for f in files[:5]:
                print(f"     - {f}")
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower():
            print(f"⚠️  Repository not found: {repo_id}")
            print(f"   Attempting to create repository...")
            
            try:
                api.create_repo(repo_id, repo_type="model", exist_ok=True)
                print(f"✅ Repository created: {repo_id}")
            except Exception as create_error:
                print(f"❌ Failed to create repository: {create_error}")
                return False
        else:
            print(f"❌ Error accessing repository: {e}")
            return False
    
    print()
    
    # Step 4: Create test file
    print("Step 4: Creating test file...")
    test_dir = Path("test_hf_upload")
    test_dir.mkdir(exist_ok=True)
    
    test_file = test_dir / "test_upload.json"
    test_data = {
        "test": "HuggingFace Hub upload test",
        "timestamp": datetime.now().isoformat(),
        "repo_id": repo_id,
        "status": "success"
    }
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"✅ Test file created: {test_file}")
    print(f"   Size: {test_file.stat().st_size} bytes")
    
    print()
    
    # Step 5: Upload test file
    print("Step 5: Uploading test file to HuggingFace Hub...")
    try:
        api.upload_file(
            path_or_fileobj=str(test_file),
            path_in_repo="test_upload.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Test upload from integration script"
        )
        print(f"✅ Upload successful!")
        print(f"   View at: https://huggingface.co/{repo_id}/blob/main/test_upload.json")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False
    
    print()
    
    # Step 6: Verify upload
    print("Step 6: Verifying upload...")
    try:
        files = api.list_repo_files(repo_id, repo_type="model")
        if "test_upload.json" in files:
            print(f"✅ File verified in repository")
        else:
            print(f"⚠️  File not found in repository (may take a moment)")
    except Exception as e:
        print(f"⚠️  Could not verify: {e}")
    
    print()
    
    # Step 7: Cleanup
    print("Step 7: Cleaning up local test file...")
    try:
        test_file.unlink()
        test_dir.rmdir()
        print(f"✅ Local test file removed")
    except Exception as e:
        print(f"⚠️  Could not remove test file: {e}")
    
    print()
    print("="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print()
    print("Your HuggingFace Hub integration is working correctly!")
    print()
    print("Next steps:")
    print("  1. Start training:")
    print("     python train.py --config config/gpu_training_117m_1.5gb.yaml --multi-gpu")
    print()
    print("  2. Checkpoints will automatically upload to:")
    print(f"     https://huggingface.co/{repo_id}")
    print()
    print("  3. Monitor uploads during training:")
    print("     Look for '📤 Uploading to HuggingFace Hub' messages")
    print()
    
    return True


def main():
    """Main entry point."""
    try:
        success = test_hf_hub_upload()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
