#!/usr/bin/env python3
"""
Cleanup old checkpoints to save disk space.

This script:
1. Deletes regular checkpoint files (checkpoint_step_*.pt)
2. Keeps best model checkpoints (best_model_*.pt)
3. Keeps error/interrupted checkpoints for recovery
4. Optionally verifies files exist on HuggingFace Hub before deletion
"""

import argparse
from pathlib import Path
import sys


def cleanup_checkpoints(checkpoint_dir, verify_hub=False, repo_id=None, dry_run=False):
    """
    Clean up checkpoint files.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        verify_hub: If True, verify files exist on HuggingFace Hub before deleting
        repo_id: HuggingFace Hub repository ID (required if verify_hub=True)
        dry_run: If True, only print what would be deleted without actually deleting
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Find all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return
    
    print(f"\n📁 Found {len(checkpoint_files)} checkpoint files in {checkpoint_dir}")
    print("="*80)
    
    # Categorize checkpoints
    regular_checkpoints = []
    best_checkpoints = []
    special_checkpoints = []
    
    for ckpt in checkpoint_files:
        if 'best_model' in ckpt.name:
            best_checkpoints.append(ckpt)
        elif any(x in ckpt.name for x in ['error', 'interrupted', 'emergency']):
            special_checkpoints.append(ckpt)
        elif 'checkpoint_step' in ckpt.name:
            regular_checkpoints.append(ckpt)
        else:
            special_checkpoints.append(ckpt)
    
    print(f"\n📊 Checkpoint breakdown:")
    print(f"   Regular checkpoints: {len(regular_checkpoints)}")
    print(f"   Best model checkpoints: {len(best_checkpoints)}")
    print(f"   Special checkpoints (error/interrupted): {len(special_checkpoints)}")
    
    # Verify on HuggingFace Hub if requested
    hub_files = set()
    if verify_hub:
        if not repo_id:
            print("\n⚠️  --verify-hub requires --repo-id")
            return
        
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            print(f"\n🔍 Checking HuggingFace Hub: {repo_id}")
            repo_files = api.list_repo_files(repo_id, repo_type="model")
            hub_files = {f for f in repo_files if f.endswith('.pt')}
            print(f"   Found {len(hub_files)} checkpoint files on Hub")
            
        except Exception as e:
            print(f"\n⚠️  Could not verify Hub files: {e}")
            print("   Proceeding without Hub verification")
    
    # Delete regular checkpoints
    deleted_count = 0
    deleted_size = 0
    skipped_count = 0
    
    print(f"\n🗑️  Processing regular checkpoints...")
    print("="*80)
    
    for ckpt in regular_checkpoints:
        file_size = ckpt.stat().st_size / (1024**3)  # Size in GB
        
        # Check if file exists on Hub
        on_hub = ckpt.name in hub_files if verify_hub else False
        hub_status = "✅ On Hub" if on_hub else ("❓ Not verified" if not verify_hub else "❌ Not on Hub")
        
        if verify_hub and not on_hub:
            print(f"   ⏭️  Skipping {ckpt.name} ({file_size:.2f} GB) - not found on Hub")
            skipped_count += 1
            continue
        
        if dry_run:
            print(f"   [DRY RUN] Would delete: {ckpt.name} ({file_size:.2f} GB) - {hub_status}")
            deleted_count += 1
            deleted_size += file_size
        else:
            try:
                ckpt.unlink()
                print(f"   ✅ Deleted: {ckpt.name} ({file_size:.2f} GB) - {hub_status}")
                deleted_count += 1
                deleted_size += file_size
            except Exception as e:
                print(f"   ❌ Failed to delete {ckpt.name}: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 Cleanup Summary:")
    print("="*80)
    
    if dry_run:
        print(f"[DRY RUN] Would delete: {deleted_count} files ({deleted_size:.2f} GB)")
    else:
        print(f"✅ Deleted: {deleted_count} files ({deleted_size:.2f} GB)")
    
    if skipped_count > 0:
        print(f"⏭️  Skipped: {skipped_count} files (not on Hub)")
    
    print(f"💾 Kept: {len(best_checkpoints)} best model checkpoints")
    print(f"💾 Kept: {len(special_checkpoints)} special checkpoints (error/interrupted)")
    
    if best_checkpoints:
        print(f"\n📌 Best model checkpoints kept:")
        for ckpt in sorted(best_checkpoints):
            file_size = ckpt.stat().st_size / (1024**3)
            print(f"   - {ckpt.name} ({file_size:.2f} GB)")
    
    if special_checkpoints:
        print(f"\n📌 Special checkpoints kept:")
        for ckpt in sorted(special_checkpoints):
            file_size = ckpt.stat().st_size / (1024**3)
            print(f"   - {ckpt.name} ({file_size:.2f} GB)")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Clean up old checkpoint files to save disk space',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (see what would be deleted)
  python scripts/cleanup_checkpoints.py --checkpoint-dir checkpoints/my_run --dry-run
  
  # Delete checkpoints
  python scripts/cleanup_checkpoints.py --checkpoint-dir checkpoints/my_run
  
  # Verify on HuggingFace Hub before deleting
  python scripts/cleanup_checkpoints.py --checkpoint-dir checkpoints/my_run \\
      --verify-hub --repo-id username/model-repo
        """
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        required=True,
        help='Directory containing checkpoint files'
    )
    parser.add_argument(
        '--verify-hub',
        action='store_true',
        help='Verify files exist on HuggingFace Hub before deleting'
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        help='HuggingFace Hub repository ID (required with --verify-hub)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    
    args = parser.parse_args()
    
    cleanup_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        verify_hub=args.verify_hub,
        repo_id=args.repo_id,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
