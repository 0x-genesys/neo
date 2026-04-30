#!/usr/bin/env python3
"""
Test script for uploading checkpoints to HuggingFace Hub.

This script tests the checkpoint upload functionality using an existing checkpoint.
"""

import os
import sys
import argparse
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import HfApi, create_repo, upload_file


def check_checkpoint(checkpoint_path):
    """Check if checkpoint file is valid."""
    print(f"📋 Checking checkpoint: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        # Load checkpoint to verify it's valid
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"✅ Checkpoint is valid")
        print(f"   File size: {Path(checkpoint_path).stat().st_size / (1024*1024):.1f} MB")
        
        # Show checkpoint contents
        if isinstance(checkpoint, dict):
            print(f"   Keys: {list(checkpoint.keys())}")
            
            if 'epoch' in checkpoint:
                print(f"   Epoch: {checkpoint['epoch']}")
            if 'global_step' in checkpoint:
                print(f"   Step: {checkpoint['global_step']}")
            if 'best_val_loss' in checkpoint:
                print(f"   Best val loss: {checkpoint['best_val_loss']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False


def upload_checkpoint_to_hf(
    checkpoint_path: str,
    repo_id: str,
    token: str = None,
    private: bool = False,
    commit_message: str = None
):
    """
    Upload a checkpoint to HuggingFace Hub.
    
    Args:
        checkpoint_path: Path to checkpoint file
        repo_id: HuggingFace repository ID (e.g., "username/model-name")
        token: HuggingFace token (or set HF_TOKEN environment variable)
        private: Whether to create a private repository
        commit_message: Custom commit message
    """
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"\n📤 Uploading checkpoint to HuggingFace Hub")
    print(f"{'='*60}")
    print(f"Local file: {checkpoint_path}")
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")
    
    # Initialize HuggingFace API
    api = HfApi(token=token)
    
    try:
        # Create repository (model repo, not dataset)
        print(f"\n📋 Creating/verifying repository...")
        create_repo(
            repo_id=repo_id,
            repo_type="model",  # This is a model checkpoint
            private=private,
            token=token,
            exist_ok=True
        )
        print(f"✅ Repository ready: {repo_id}")
        
        # Prepare commit message
        if commit_message is None:
            commit_message = f"Upload checkpoint: {checkpoint_path.name}"
        
        # Upload checkpoint
        print(f"\n📤 Uploading {checkpoint_path.name}...")
        print(f"   Size: {checkpoint_path.stat().st_size / (1024*1024):.1f} MB")
        
        url = upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=checkpoint_path.name,
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=commit_message
        )
        
        print(f"\n✅ Upload complete!")
        print(f"🔗 Repository: https://huggingface.co/{repo_id}")
        print(f"🔗 File: {url}")
        
        # Show download instructions
        print(f"\n💡 To download this checkpoint:")
        print(f"   from huggingface_hub import hf_hub_download")
        print(f"   checkpoint = hf_hub_download(")
        print(f"       repo_id='{repo_id}',")
        print(f"       filename='{checkpoint_path.name}'")
        print(f"   )")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        return False


def test_checkpoint_upload():
    """Interactive test for checkpoint upload."""
    
    print("🧪 Checkpoint Upload Test")
    print("="*60)
    
    # Check for HF token
    token = os.environ.get('HF_TOKEN')
    if not token:
        print("❌ HuggingFace token not found")
        print("   Set your token: export HF_TOKEN=your_token_here")
        print("   Get token from: https://huggingface.co/settings/tokens")
        return False
    
    # Get username
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        username = user_info['name']
        print(f"✅ Authenticated as: {username}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False
    
    # Find available checkpoints
    print(f"\n📁 Looking for checkpoints...")
    checkpoint_dirs = [
        "checkpoints/production",
        "checkpoints/quick_start",
        "checkpoints/gpu_training_117m_balanced"
    ]
    
    available_checkpoints = []
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_path = Path(checkpoint_dir)
        if checkpoint_path.exists():
            for file in checkpoint_path.glob("*.pt"):
                available_checkpoints.append(file)
    
    if not available_checkpoints:
        print("❌ No checkpoints found")
        print("   Train a model first to create checkpoints")
        return False
    
    print(f"✅ Found {len(available_checkpoints)} checkpoint(s):")
    for i, checkpoint in enumerate(available_checkpoints, 1):
        size_mb = checkpoint.stat().st_size / (1024*1024)
        print(f"   {i}. {checkpoint} ({size_mb:.1f} MB)")
    
    # Select checkpoint
    if len(available_checkpoints) == 1:
        selected_checkpoint = available_checkpoints[0]
        print(f"\n📦 Using: {selected_checkpoint}")
    else:
        try:
            choice = int(input(f"\nSelect checkpoint (1-{len(available_checkpoints)}): ")) - 1
            selected_checkpoint = available_checkpoints[choice]
        except (ValueError, IndexError):
            print("❌ Invalid selection")
            return False
    
    # Verify checkpoint
    if not check_checkpoint(selected_checkpoint):
        return False
    
    # Suggest repository name
    suggested_repo = f"{username}/transformer-checkpoint-test"
    print(f"\n💡 Suggested repository: {suggested_repo}")
    
    repo_id = input(f"Enter repository ID (or press Enter for default): ").strip()
    if not repo_id:
        repo_id = suggested_repo
    
    # Confirm upload
    print(f"\n📤 Ready to upload:")
    print(f"   Checkpoint: {selected_checkpoint}")
    print(f"   Repository: {repo_id}")
    
    confirm = input("Proceed with upload? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Upload cancelled")
        return False
    
    # Upload
    return upload_checkpoint_to_hf(
        checkpoint_path=str(selected_checkpoint),
        repo_id=repo_id,
        token=token,
        private=False
    )


def main():
    """CLI interface for checkpoint upload."""
    
    parser = argparse.ArgumentParser(
        description='Upload model checkpoints to HuggingFace Hub'
    )
    parser.add_argument(
        'checkpoint',
        nargs='?',
        help='Path to checkpoint file (e.g., checkpoints/production/checkpoint.pt)'
    )
    parser.add_argument(
        'repo_id',
        nargs='?',
        help='HuggingFace repository ID (e.g., username/model-name)'
    )
    parser.add_argument(
        '--token',
        help='HuggingFace token (or set HF_TOKEN environment variable)'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Create private repository'
    )
    parser.add_argument(
        '--message',
        help='Custom commit message'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run interactive test mode'
    )
    
    args = parser.parse_args()
    
    # Interactive test mode
    if args.test or (not args.checkpoint and not args.repo_id):
        return 0 if test_checkpoint_upload() else 1
    
    # Direct upload mode
    if not args.checkpoint or not args.repo_id:
        print("❌ Both checkpoint and repo_id are required")
        print("   Or use --test for interactive mode")
        parser.print_help()
        return 1
    
    # Get token
    token = args.token or os.environ.get('HF_TOKEN')
    if not token:
        print("❌ HuggingFace token required")
        print("   Set HF_TOKEN environment variable or use --token")
        print("   Get token from: https://huggingface.co/settings/tokens")
        return 1
    
    # Check checkpoint
    if not check_checkpoint(args.checkpoint):
        return 1
    
    # Upload
    try:
        success = upload_checkpoint_to_hf(
            checkpoint_path=args.checkpoint,
            repo_id=args.repo_id,
            token=token,
            private=args.private,
            commit_message=args.message
        )
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())