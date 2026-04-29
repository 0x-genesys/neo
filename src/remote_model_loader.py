"""
Remote model and checkpoint loader for HuggingFace Hub.

Enables loading checkpoints and models directly from HuggingFace Hub
without manual download.
"""

import os
from pathlib import Path
from typing import Optional, Dict
import torch
from huggingface_hub import hf_hub_download, list_repo_files


class RemoteModelLoader:
    """Load models and checkpoints from HuggingFace Hub."""
    
    def __init__(
        self,
        repo_id: str = "0x-genesys/neo_weights_checkpoints",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize remote model loader.
        
        Args:
            repo_id: HuggingFace repository ID
            cache_dir: Local cache directory (default: ~/.cache/huggingface)
        """
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        
    def list_available_checkpoints(self) -> list:
        """List all available checkpoint files in the repository."""
        try:
            files = list_repo_files(self.repo_id, repo_type="model")
            checkpoints = [f for f in files if f.endswith('.pt')]
            return sorted(checkpoints)
        except Exception as e:
            print(f"❌ Error listing checkpoints: {e}")
            return []
    
    def download_checkpoint(
        self,
        filename: str,
        force_download: bool = False
    ) -> str:
        """
        Download a checkpoint from HuggingFace Hub.
        
        Args:
            filename: Checkpoint filename (e.g., "checkpoint.pt")
            force_download: Force re-download even if cached
            
        Returns:
            Local path to downloaded checkpoint
        """
        print(f"📥 Downloading checkpoint from HuggingFace Hub...")
        print(f"   Repository: {self.repo_id}")
        print(f"   File: {filename}")
        
        try:
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                repo_type="model",
                cache_dir=self.cache_dir,
                force_download=force_download
            )
            
            print(f"✅ Downloaded to: {local_path}")
            return local_path
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print(f"\n💡 Available checkpoints:")
            checkpoints = self.list_available_checkpoints()
            if checkpoints:
                for cp in checkpoints:
                    print(f"   - {cp}")
            else:
                print(f"   Could not list checkpoints")
            raise
    
    def load_checkpoint(
        self,
        filename: str,
        map_location: str = "cpu",
        force_download: bool = False
    ) -> Dict:
        """
        Download and load a checkpoint from HuggingFace Hub.
        
        Args:
            filename: Checkpoint filename
            map_location: Device to load checkpoint to
            force_download: Force re-download
            
        Returns:
            Loaded checkpoint dictionary
        """
        local_path = self.download_checkpoint(filename, force_download)
        
        print(f"📂 Loading checkpoint...")
        checkpoint = torch.load(local_path, map_location=map_location)
        
        print(f"✅ Checkpoint loaded successfully")
        if isinstance(checkpoint, dict):
            print(f"   Keys: {list(checkpoint.keys())}")
            if 'epoch' in checkpoint:
                print(f"   Epoch: {checkpoint['epoch']}")
            if 'global_step' in checkpoint:
                print(f"   Step: {checkpoint['global_step']}")
            if 'best_val_loss' in checkpoint:
                print(f"   Best val loss: {checkpoint['best_val_loss']:.4f}")
        
        return checkpoint


def get_remote_checkpoint_path(
    filename: str,
    repo_id: str = "0x-genesys/neo_weights_checkpoints"
) -> str:
    """
    Helper function to download and get path to remote checkpoint.
    
    Args:
        filename: Checkpoint filename
        repo_id: HuggingFace repository ID
        
    Returns:
        Local path to downloaded checkpoint
    """
    loader = RemoteModelLoader(repo_id=repo_id)
    return loader.download_checkpoint(filename)


def load_remote_checkpoint(
    filename: str,
    repo_id: str = "0x-genesys/neo_weights_checkpoints",
    map_location: str = "cpu"
) -> Dict:
    """
    Helper function to load checkpoint from HuggingFace Hub.
    
    Args:
        filename: Checkpoint filename
        repo_id: HuggingFace repository ID
        map_location: Device to load to
        
    Returns:
        Loaded checkpoint dictionary
    """
    loader = RemoteModelLoader(repo_id=repo_id)
    return loader.load_checkpoint(filename, map_location=map_location)