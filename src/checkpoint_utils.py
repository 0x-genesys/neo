"""
Shared checkpoint utilities for uploading to HuggingFace Hub.
"""

from pathlib import Path


def upload_checkpoint_to_hub(
    checkpoint_path,
    config,
    global_step,
    epoch,
    is_best=False,
    delete_after_upload=False
):
    """
    Upload checkpoint to HuggingFace Hub with optional local cleanup.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Training config dict with huggingface_hub settings
        global_step: Current training step
        epoch: Current epoch
        is_best: Whether this is the best model
        delete_after_upload: Whether to delete local file after upload (saves disk space)
    
    Returns:
        bool: True if upload succeeded, False otherwise
    """
    # Check if HF Hub upload is enabled
    hf_hub_config = config.get('huggingface_hub', {})
    if not hf_hub_config.get('enabled', False):
        return False
    
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        repo_id = hf_hub_config.get('repo_id')
        
        if not repo_id:
            print("⚠️  HuggingFace Hub repo_id not configured, skipping upload")
            return False
        
        # Determine remote path
        checkpoint_path = Path(checkpoint_path)
        if is_best:
            path_in_repo = f"best_model_step_{global_step}.pt"
        else:
            path_in_repo = checkpoint_path.name
        
        print(f"📤 Uploading to HuggingFace Hub: {repo_id}/{path_in_repo}")
        
        # Upload file
        api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload checkpoint at step {global_step} (epoch {epoch})"
        )
        
        print(f"✅ Uploaded to HuggingFace Hub: https://huggingface.co/{repo_id}/tree/main")
        
        # Clean up local checkpoint after successful upload if requested
        if delete_after_upload:
            # Keep best_model checkpoints locally
            if 'best_model' not in checkpoint_path.name:
                try:
                    checkpoint_path.unlink()
                    print(f"🗑️  Deleted local checkpoint (saved to Hub): {checkpoint_path.name}")
                except Exception as e:
                    print(f"⚠️  Could not delete local checkpoint: {e}")
            else:
                print(f"💾 Keeping best model checkpoint locally: {checkpoint_path.name}")
        
        return True
        
    except ImportError:
        print("⚠️  huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"⚠️  Failed to upload to HuggingFace Hub: {e}")
        print(f"   Continuing training...")
        return False
