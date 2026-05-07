"""
Automatic dataset downloader for pre-processed HuggingFace datasets.

This module handles downloading and caching of pre-processed binary datasets
from HuggingFace Hub, eliminating the need to build datasets locally.

Features:
- Automatic download when dataset is missing
- Progress bars with proper tqdm configuration
- Configurable dataset sources via YAML config
- Support for multiple dataset sizes (117M, 300M, etc.)
- Efficient binary format with memory mapping
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import requests
from tqdm import tqdm
import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
import yaml


class DatasetDownloader:
    """Download and manage pre-processed binary datasets from HuggingFace Hub."""
    
    def __init__(self, cache_dir: str = "data"):
        """
        Initialize dataset downloader.
        
        Args:
            cache_dir: Local directory to store downloaded datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure tqdm to avoid newline issues
        tqdm.pandas()
        
    def download_dataset(
        self, 
        repo_id: str, 
        dataset_name: str,
        force_download: bool = False,
        is_curriculum: bool = False
    ) -> Tuple[str, str, Dict]:
        """
        Download a pre-processed dataset from HuggingFace Hub.
        
        Args:
            repo_id: HuggingFace repository ID (e.g., "0x-genesys/mix_wiki_code_chat_data_300M_tokens")
            dataset_name: Local dataset name (e.g., "balanced_300m")
            force_download: Force re-download even if files exist
            is_curriculum: Whether this is a curriculum dataset (separate source files)
            
        Returns:
            Tuple of (train_file_path, val_file_path, dataset_stats)
        """
        dataset_dir = self.cache_dir / dataset_name
        train_file = dataset_dir / "train.bin"
        val_file = dataset_dir / "val.bin"
        stats_file = dataset_dir / "dataset_stats.json"
        
        # Check if dataset already exists
        if not force_download and self._dataset_exists(dataset_dir, is_curriculum):
            print(f"✅ Dataset '{dataset_name}' already exists at {dataset_dir}")
            stats = self._load_stats(stats_file)
            return str(train_file), str(val_file), stats
        
        print(f"\n{'='*80}")
        print(f"DOWNLOADING DATASET: {dataset_name}")
        print(f"{'='*80}")
        print(f"Repository: {repo_id}")
        print(f"Local path: {dataset_dir}")
        
        # Create dataset directory
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # List available files in the repository
            print(f"\n📋 Checking repository contents...")
            try:
                repo_files = list_repo_files(repo_id, repo_type="dataset")
            except Exception as e:
                if "404" in str(e) or "Repository Not Found" in str(e):
                    raise FileNotFoundError(
                        f"Repository '{repo_id}' not found on HuggingFace Hub.\n"
                        f"Please check:\n"
                        f"1. Repository exists: https://huggingface.co/datasets/{repo_id}\n"
                        f"2. Repository is public or you have access\n"
                        f"3. Repository ID is correct (should be 'username/dataset-name')\n"
                        f"\nTo create this repository:\n"
                        f"1. Build dataset locally: python scripts/prepare_balanced_dataset.py\n"
                        f"2. Upload to HF Hub: python scripts/upload_dataset_to_hf.py data/balanced_300m {repo_id}"
                    )
                else:
                    raise
            
            # Filter for dataset files
            dataset_files = [f for f in repo_files if f.endswith(('.bin', '.json'))]
            print(f"Found {len(dataset_files)} dataset files:")
            for file in dataset_files:
                print(f"  - {file}")
            
            # Download each file with progress bar
            downloaded_files = {}
            
            for filename in dataset_files:
                print(f"\n📥 Downloading {filename}...")
                
                # Download to HF cache first (more reliable)
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    resume_download=True,  # Resume partial downloads
                    repo_type="dataset"  # Specify this is a dataset repository
                )
                
                downloaded_files[filename] = downloaded_path
                
                # Show file size
                file_size = Path(downloaded_path).stat().st_size
                print(f"✅ Downloaded {filename}: {self._format_size(file_size)}")
                
                # Copy file from HF cache to target directory
                expected_path = dataset_dir / filename
                if downloaded_path != str(expected_path):
                    import shutil
                    print(f"   📋 Copying to {expected_path}...")
                    shutil.copy2(downloaded_path, expected_path)
                    print(f"   ✅ Copied to target directory")
            
            # Verify required files exist in target directory
            print(f"\n🔍 Verifying files in target directory...")
            if is_curriculum:
                # For curriculum datasets, check for source files
                required_files = ['wikitext_train.bin', 'stack_train.bin', 'ultrachat_train.bin', 'val.bin']
                missing_files = []
                for f in required_files:
                    file_path = dataset_dir / f
                    if not file_path.exists():
                        missing_files.append(f)
                        print(f"   ❌ Missing: {f}")
                    else:
                        print(f"   ✅ Found: {f} ({self._format_size(file_path.stat().st_size)})")
                        
                if missing_files:
                    raise FileNotFoundError(
                        f"Missing curriculum source files in {dataset_dir}: {missing_files}\n"
                        f"Files were downloaded but not copied correctly.\n"
                        f"Please check the target directory: {dataset_dir}"
                    )
            else:
                # For standard datasets, check for train.bin
                train_path = dataset_dir / "train.bin"
                val_path = dataset_dir / "val.bin"
                
                if not train_path.exists():
                    raise FileNotFoundError(
                        f"train.bin not found in {dataset_dir}\n"
                        f"Repository does not contain train.bin (might be a curriculum dataset)\n"
                        f"If this is a curriculum dataset, set curriculum_learning.enabled: true in config"
                    )
                    
                print(f"   ✅ Found: train.bin ({self._format_size(train_path.stat().st_size)})")
                if val_path.exists():
                    print(f"   ✅ Found: val.bin ({self._format_size(val_path.stat().st_size)})")
            
            # Load and display statistics
            stats = {}
            if "dataset_stats.json" in downloaded_files or (dataset_dir / "dataset_stats.json").exists():
                stats = self._load_stats(dataset_dir / "dataset_stats.json")
                self._print_dataset_info(stats)
            
            print(f"\n✅ Dataset download complete!")
            print(f"📁 Files saved to: {dataset_dir}")
            
            # List downloaded files
            print(f"\n📋 Downloaded files:")
            for filename in dataset_files:
                file_path = dataset_dir / filename
                if file_path.exists():
                    print(f"   ✅ {filename} ({self._format_size(file_path.stat().st_size)})")
                else:
                    print(f"   ❌ {filename} (missing)")
            
            return str(train_file), str(val_file), stats
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            # Clean up partial downloads
            if dataset_dir.exists():
                import shutil
                shutil.rmtree(dataset_dir)
            raise
    
    def _dataset_exists(self, dataset_dir: Path, is_curriculum: bool = False) -> bool:
        """Check if dataset files exist locally in the target directory."""
        if not dataset_dir.exists():
            return False
            
        if is_curriculum:
            # For curriculum datasets, check for source files
            stats_file = dataset_dir / "dataset_stats.json"
            if not stats_file.exists():
                return False
            
            # Load stats to get source names
            stats = self._load_stats(stats_file)
            sources = stats.get('sources', {})
            
            # Check if all source files exist
            for source in sources.keys():
                if sources[source].get('tokens', 0) > 0:
                    source_file = dataset_dir / f"{source}_train.bin"
                    if not source_file.exists() or source_file.stat().st_size == 0:
                        print(f"   ⚠️  Missing curriculum source file: {source_file}")
                        return False
            
            # Check validation file
            val_file = dataset_dir / "val.bin"
            if not val_file.exists() or val_file.stat().st_size == 0:
                print(f"   ⚠️  Missing validation file: {val_file}")
                return False
                
            print(f"   ✅ All curriculum source files present")
            return True
        else:
            # For standard datasets, check for train.bin
            train_file = dataset_dir / "train.bin"
            val_file = dataset_dir / "val.bin"
            
            if not train_file.exists() or train_file.stat().st_size == 0:
                print(f"   ⚠️  Missing train file: {train_file}")
                return False
            if not val_file.exists() or val_file.stat().st_size == 0:
                print(f"   ⚠️  Missing validation file: {val_file}")
                return False
                
            print(f"   ✅ Dataset files present")
            return True
    
    def _load_stats(self, stats_file: Path) -> Dict:
        """Load dataset statistics from JSON file."""
        if not stats_file.exists():
            return {}
        
        try:
            with open(stats_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Could not load stats file: {e}")
            return {}
    
    def _print_dataset_info(self, stats: Dict):
        """Print dataset information from stats."""
        if not stats:
            return
        
        print(f"\n📊 Dataset Information:")
        print(f"{'='*50}")
        
        # Print totals
        totals = stats.get('totals', {})
        if totals:
            print(f"Total documents: {totals.get('documents', 'Unknown'):,}")
            print(f"Total tokens: {totals.get('tokens', 'Unknown'):,}")
        
        # Print tokenizer info
        tokenizer = stats.get('tokenizer', 'Unknown')
        vocab_size = stats.get('vocab_size', 'Unknown')
        print(f"Tokenizer: {tokenizer}")
        print(f"Vocabulary size: {vocab_size:,}" if isinstance(vocab_size, int) else f"Vocabulary size: {vocab_size}")
        
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
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def get_dataset_config(self, config: Dict) -> Optional[Dict]:
        """
        Extract dataset configuration from training config.
        
        Args:
            config: Training configuration dictionary
            
        Returns:
            Dataset configuration with HuggingFace repo info, or None if not configured
        """
        data_config = config.get('data', {})
        
        # Check if HuggingFace dataset is configured
        hf_config = data_config.get('huggingface_dataset')
        if not hf_config:
            return None
        
        # Check if curriculum learning is enabled
        training_config = config.get('training', {})
        curriculum_config = training_config.get('curriculum_learning', {})
        is_curriculum = curriculum_config.get('enabled', False)
        
        return {
            'repo_id': hf_config.get('repo_id'),
            'dataset_name': hf_config.get('dataset_name', 'balanced_300m'),
            'auto_download': hf_config.get('auto_download', True),
            'is_curriculum': is_curriculum
        }
    
    def ensure_dataset_available(self, config: Dict) -> Tuple[str, str, Dict]:
        """
        Ensure dataset is available locally, downloading if necessary.
        
        Args:
            config: Training configuration dictionary
            
        Returns:
            Tuple of (train_file_path, val_file_path, dataset_stats)
            
        Raises:
            ValueError: If dataset configuration is invalid
            FileNotFoundError: If dataset cannot be found or downloaded
        """
        data_config = config.get('data', {})
        
        # Check if using binary dataset type
        if data_config.get('dataset_type') != 'binary':
            raise ValueError("Dataset downloader only supports binary dataset type")
        
        # Get file paths from config
        train_file = data_config.get('train_file')
        val_file = data_config.get('val_file')
        
        if not train_file:
            raise ValueError("train_file must be specified in data config")
        
        # Check if files exist locally
        train_path = Path(train_file)
        val_path = Path(val_file) if val_file else None
        
        if train_path.exists() and (not val_path or val_path.exists()):
            print(f"✅ Dataset files found locally:")
            print(f"  Train: {train_path} ({self._format_size(train_path.stat().st_size)})")
            if val_path and val_path.exists():
                print(f"  Val: {val_path} ({self._format_size(val_path.stat().st_size)})")
            
            # Load stats if available
            stats_file = train_path.parent / "dataset_stats.json"
            stats = self._load_stats(stats_file)
            return str(train_path), str(val_path) if val_path else None, stats
        
        # Files don't exist - check for HuggingFace config
        hf_config = self.get_dataset_config(config)
        if not hf_config:
            raise FileNotFoundError(
                f"Dataset files not found and no HuggingFace download configured.\n"
                f"Missing: {train_file}\n"
                f"To enable automatic download, add to your config:\n"
                f"data:\n"
                f"  huggingface_dataset:\n"
                f"    repo_id: '0x-genesys/mix_wiki_code_chat_data_300M_tokens'\n"
                f"    dataset_name: 'balanced_300m'\n"
                f"    auto_download: true"
            )
        
        if not hf_config['auto_download']:
            raise FileNotFoundError(
                f"Dataset files not found and auto_download is disabled.\n"
                f"Missing: {train_file}\n"
                f"Set auto_download: true in your config to enable automatic download."
            )
        
        # Download dataset
        print(f"📥 Dataset files not found locally, downloading from HuggingFace...")
        return self.download_dataset(
            repo_id=hf_config['repo_id'],
            dataset_name=hf_config['dataset_name'],
            is_curriculum=hf_config.get('is_curriculum', False)
        )


def configure_tqdm_for_datasets():
    """
    Configure tqdm to avoid newline issues in dataset loading.
    
    This fixes the problem where tqdm creates new lines on every update
    instead of updating the existing progress bar.
    """
    # Set tqdm defaults for better CLI experience
    tqdm.pandas(
        desc="Processing",
        unit="items",
        dynamic_ncols=True,  # Adjust width to terminal
        leave=True,          # Keep progress bar after completion
        position=0,          # Use position 0 (main progress bar)
        ascii=False,         # Use Unicode characters for better display
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    # Disable tqdm in environments where it causes issues
    if os.environ.get('DISABLE_TQDM', '').lower() in ('1', 'true', 'yes'):
        tqdm.disable = True
        print("ℹ️  tqdm progress bars disabled via DISABLE_TQDM environment variable")


# Configure tqdm on import
configure_tqdm_for_datasets()


def main():
    """CLI interface for dataset downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download pre-processed datasets from HuggingFace Hub')
    parser.add_argument('repo_id', help='HuggingFace repository ID')
    parser.add_argument('--dataset-name', default='balanced_300m', help='Local dataset name')
    parser.add_argument('--cache-dir', default='data', help='Local cache directory')
    parser.add_argument('--force', action='store_true', help='Force re-download')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(cache_dir=args.cache_dir)
    
    try:
        train_file, val_file, stats = downloader.download_dataset(
            repo_id=args.repo_id,
            dataset_name=args.dataset_name,
            force_download=args.force
        )
        
        print(f"\n✅ Download complete!")
        print(f"Train file: {train_file}")
        print(f"Val file: {val_file}")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        exit(1)


if __name__ == '__main__':
    main()