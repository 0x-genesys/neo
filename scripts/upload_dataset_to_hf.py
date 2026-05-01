#!/usr/bin/env python3
"""
Upload pre-processed binary datasets to HuggingFace Hub.

This script uploads locally built datasets to HuggingFace Hub so they can
be downloaded automatically by other users.
"""

import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import os


def upload_dataset(
    local_dataset_dir: str,
    repo_id: str, 
    token: str = None,
    private: bool = False
):
    """
    Upload a local dataset directory to HuggingFace Hub.
    
    Args:
        local_dataset_dir: Path to local dataset directory (e.g., "data/balanced_300m")
        repo_id: HuggingFace repository ID (e.g., "username/dataset-name")
        token: HuggingFace token (or set HF_TOKEN environment variable)
        private: Whether to create a private repository
    """
    
    dataset_path = Path(local_dataset_dir)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    
    # Check for required files
    train_file = dataset_path / "train.bin"
    if not train_file.exists():
        raise FileNotFoundError(f"train.bin not found in {dataset_path}")
    
    print(f"📤 Uploading dataset to HuggingFace Hub")
    print(f"{'='*60}")
    print(f"Local path: {dataset_path}")
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")
    
    # Initialize HuggingFace API
    api = HfApi(token=token)
    
    try:
        # Create repository
        print(f"\n📋 Creating repository...")
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            token=token,
            exist_ok=True  # Don't fail if repo already exists
        )
        print(f"✅ Repository created/verified: {repo_id}")
        
        # Get list of files to upload
        files_to_upload = []
        for file_path in dataset_path.iterdir():
            if file_path.is_file() and file_path.suffix in ['.bin', '.json']:
                files_to_upload.append(file_path)
        
        print(f"\n📁 Files to upload:")
        total_size = 0
        for file_path in files_to_upload:
            size = file_path.stat().st_size
            total_size += size
            print(f"  - {file_path.name}: {format_size(size)}")
        
        print(f"Total size: {format_size(total_size)}")
        
        # Upload files
        print(f"\n📤 Uploading files...")
        for file_path in files_to_upload:
            print(f"Uploading {file_path.name}...")
            
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path.name,
                repo_id=repo_id,
                repo_type="dataset",
                token=token
            )
            
            print(f"✅ Uploaded {file_path.name}")
        
        # Create README if it doesn't exist
        readme_content = create_dataset_readme(dataset_path, repo_id)
        
        print(f"\n📝 Creating README...")
        api.upload_file(
            path_or_fileobj=readme_content.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        
        print(f"\n✅ Upload complete!")
        print(f"🔗 Repository URL: https://huggingface.co/datasets/{repo_id}")
        print(f"\n💡 To use this dataset in training:")
        print(f"   huggingface_dataset:")
        print(f"     repo_id: \"{repo_id}\"")
        print(f"     dataset_name: \"your_local_name\"")
        print(f"     auto_download: true")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        raise


def create_dataset_readme(dataset_path: Path, repo_id: str) -> str:
    """Create a README.md for the dataset repository."""
    
    # Load statistics if available
    stats_file = dataset_path / "dataset_stats.json"
    stats = {}
    if stats_file.exists():
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
        except:
            pass
    
    # Get file sizes
    train_file = dataset_path / "train.bin"
    val_file = dataset_path / "val.bin"
    
    train_size = train_file.stat().st_size if train_file.exists() else 0
    val_size = val_file.stat().st_size if val_file.exists() else 0
    
    # Extract dataset name from repo_id
    dataset_name = repo_id.split('/')[-1].replace('_', ' ').title()
    
    readme = f"""# {dataset_name}

This is a pre-processed binary dataset for language model training.

## Dataset Information

- **Format**: Binary tokenized files (.bin)
- **Tokenizer**: {stats.get('tokenizer', 'Unknown')}
- **Vocabulary Size**: {stats.get('vocab_size', 'Unknown'):,}

## Files

- `train.bin`: Training data ({format_size(train_size)})
- `val.bin`: Validation data ({format_size(val_size)})
- `dataset_stats.json`: Dataset statistics and metadata

## Statistics
"""
    
    # Add statistics if available
    totals = stats.get('totals', {})
    if totals:
        readme += f"""
- **Total Documents**: {totals.get('documents', 'Unknown'):,}
- **Total Tokens**: {totals.get('tokens', 'Unknown'):,}
"""
    
    # Add source composition
    sources = stats.get('sources', {})
    if sources:
        readme += f"""
## Source Composition

"""
        total_tokens = totals.get('tokens', 1)
        for source, source_stats in sources.items():
            docs = source_stats.get('docs', 0)
            tokens = source_stats.get('tokens', 0)
            if tokens > 0:
                pct = (tokens / total_tokens * 100) if total_tokens > 0 else 0
                readme += f"- **{source.title()}**: {docs:,} documents, {tokens:,} tokens ({pct:.1f}%)\n"
    
    readme += f"""
## Usage

This dataset is designed to be used with the automatic dataset downloader. Add this to your training configuration:

```yaml
data:
  dataset_type: "binary"
  train_file: "data/your_dataset_name/train.bin"
  val_file: "data/your_dataset_name/val.bin"
  
  huggingface_dataset:
    repo_id: "{repo_id}"
    dataset_name: "your_dataset_name"
    auto_download: true
```

The system will automatically download and cache the dataset when training starts.

## License

Please check the original data sources for licensing information.
"""
    
    return readme


def format_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def main():
    """CLI interface for dataset uploader."""
    
    parser = argparse.ArgumentParser(
        description='Upload pre-processed datasets to HuggingFace Hub'
    )
    parser.add_argument(
        'dataset_dir',
        help='Path to local dataset directory (e.g., data/balanced_300m)'
    )
    parser.add_argument(
        'repo_id',
        help='HuggingFace repository ID (e.g., username/dataset-name)'
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
    
    args = parser.parse_args()
    
    # Get token from environment if not provided
    token = args.token or os.environ.get('HF_TOKEN')
    if not token:
        print("❌ HuggingFace token required. Set HF_TOKEN environment variable or use --token")
        print("   Get your token from: https://huggingface.co/settings/tokens")
        exit(1)
    
    try:
        upload_dataset(
            local_dataset_dir=args.dataset_dir,
            repo_id=args.repo_id,
            token=token,
            private=args.private
        )
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        exit(1)


if __name__ == '__main__':
    main()