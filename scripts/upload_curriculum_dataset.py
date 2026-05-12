"""
Upload curriculum learning dataset to HuggingFace Hub.

This script:
1. Splits the combined dataset into separate source files
2. Uploads all files to HuggingFace Hub
3. Creates a dataset card with usage instructions
"""

import numpy as np
import json
from pathlib import Path
import argparse
from huggingface_hub import HfApi, create_repo


CURRICULUM_SOURCES = ("wikitext", "ultrachat")


def filter_stats_for_curriculum(stats):
    """Keep only the two Conversational Scholar curriculum sources."""
    sources = {
        source: stats.get('sources', {}).get(source, {'docs': 0, 'tokens': 0})
        for source in CURRICULUM_SOURCES
        if stats.get('sources', {}).get(source, {}).get('tokens', 0) > 0
    }
    total_docs = sum(info.get('docs', 0) for info in sources.values())
    total_tokens = sum(info.get('tokens', 0) for info in sources.values())

    filtered = dict(stats)
    filtered['sources'] = sources
    filtered['totals'] = {
        'documents': total_docs,
        'tokens': total_tokens,
    }
    filtered['targets'] = {
        'wikitext': 240_000_000,
        'ultrachat': 60_000_000,
    }
    return filtered


def prepare_and_upload_curriculum_dataset(
    data_dir: str,
    repo_id: str,
    temp_dir: str = "data/curriculum_temp",
    private: bool = False
):
    """
    Prepare curriculum dataset and upload to HuggingFace Hub.
    
    Args:
        data_dir: Directory containing combined dataset
        repo_id: HuggingFace repository ID (e.g., "username/dataset-name")
        temp_dir: Temporary directory for split files
        private: Whether to create a private repository
    """
    data_dir = Path(data_dir)
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Curriculum Dataset Upload to HuggingFace Hub")
    print("="*80)
    print(f"Source: {data_dir}")
    print(f"Destination: {repo_id}")
    print(f"Temp directory: {temp_dir}")
    print("="*80 + "\n")
    
    # Step 1: Load dataset stats
    print("Step 1: Loading dataset statistics...")
    stats_file = data_dir / "dataset_stats.json"
    if not stats_file.exists():
        raise FileNotFoundError(f"Dataset stats not found: {stats_file}")
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    stats = filter_stats_for_curriculum(stats)
    
    print(f"✅ Loaded stats from {stats_file}")
    for source, info in stats['sources'].items():
        if info['tokens'] > 0:
            print(f"  {source:12s}: {info['tokens']:,} tokens")
    
    # Step 2: Load source data
    print(f"\nStep 2: Loading source data...")
    source_names = list(stats['sources'].keys())
    if not source_names:
        raise ValueError("No wikitext or ultrachat sources found in dataset_stats.json")

    source_files_exist = all((data_dir / f"{source}_train.bin").exists() for source in source_names)
    source_ranges = {}

    if source_files_exist:
        print("✅ Found existing per-source files")
        tokens = None
    else:
        train_file = data_dir / "train.bin"
        if not train_file.exists():
            raise FileNotFoundError(f"Train file not found: {train_file}")

        tokens = np.memmap(train_file, dtype=np.uint32, mode='r')
        total_tokens = len(tokens)
        print(f"✅ Loaded {total_tokens:,} tokens from {train_file}")

        current_pos = 0
        for source in source_names:
            source_tokens = stats['sources'][source]['tokens']
            source_ranges[source] = (current_pos, current_pos + source_tokens)
            current_pos += source_tokens

    # Step 3: Write source files
    print(f"\nStep 3: Writing source files...")

    for source in source_names:
        output_file = temp_dir / f"{source}_train.bin"
        
        print(f"\n  Processing {source}...")
        print(f"    Output: {output_file}")
        
        source_file = data_dir / f"{source}_train.bin"
        if source_file.exists():
            source_tokens = np.memmap(source_file, dtype=np.uint32, mode='r')
        else:
            start, end = source_ranges[source]
            print(f"    Range: {start:,} - {end:,}")
            source_tokens = tokens[start:end]

        with open(output_file, 'wb') as f:
            source_tokens.tofile(f)
        
        # Verify
        verify_tokens = np.memmap(output_file, dtype=np.uint32, mode='r')
        print(f"    ✅ Saved {len(verify_tokens):,} tokens")
    
    # Step 4: Copy validation file
    print(f"\nStep 4: Copying validation file...")
    val_file = data_dir / "val.bin"
    if val_file.exists():
        output_val = temp_dir / "val.bin"
        val_tokens = np.memmap(val_file, dtype=np.uint32, mode='r')
        with open(output_val, 'wb') as f:
            val_tokens.tofile(f)
        print(f"✅ Copied {len(val_tokens):,} tokens to {output_val}")
    else:
        print("⚠️  No validation file found, skipping")
    
    # Step 5: Save stats
    print(f"\nStep 5: Saving dataset statistics...")
    output_stats = temp_dir / "dataset_stats.json"
    with open(output_stats, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Saved stats to {output_stats}")
    
    # Step 6: Create README
    print(f"\nStep 6: Creating README...")
    readme_content = create_readme(stats, source_ranges)
    readme_file = temp_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    print(f"✅ Created README.md")
    
    # Step 7: Upload to HuggingFace Hub
    print(f"\nStep 7: Uploading to HuggingFace Hub...")
    print(f"Repository: {repo_id}")
    
    api = HfApi()
    
    # Create repository
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
        print(f"✅ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"⚠️  Repository creation: {e}")
    
    # Upload files
    files_to_upload = [
        "dataset_stats.json",
        "README.md",
        "val.bin"
    ]
    
    # Add source files
    for source in source_names:
        files_to_upload.append(f"{source}_train.bin")
    
    print(f"\nUploading {len(files_to_upload)} files...")
    for filename in files_to_upload:
        filepath = temp_dir / filename
        if filepath.exists():
            print(f"  Uploading {filename}...")
            try:
                api.upload_file(
                    path_or_fileobj=str(filepath),
                    path_in_repo=filename,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Upload {filename}"
                )
                print(f"    ✅ Uploaded")
            except Exception as e:
                print(f"    ❌ Failed: {e}")
        else:
            print(f"  ⚠️  Skipping {filename} (not found)")
    
    print(f"\n{'='*80}")
    print(f"✅ Upload complete!")
    print(f"{'='*80}")
    print(f"\nDataset URL: https://huggingface.co/datasets/{repo_id}")
    print(f"\nFiles uploaded:")
    for filename in files_to_upload:
        print(f"  - {filename}")
    
    print(f"\nTo use in training:")
    print(f"  1. Update config:")
    print(f"     huggingface_dataset:")
    print(f"       repo_id: \"{repo_id}\"")
    print(f"       dataset_name: \"balanced_300m_curriculum\"")
    print(f"  2. Run training:")
    print(f"     python train.py --config config/gpu_training_117m_balanced.yaml")


def create_readme(stats, source_ranges):
    """Create README.md content for the dataset."""
    total_tokens = stats['totals']['tokens']
    total_docs = stats['totals']['documents']
    
    readme = f"""# Curriculum Learning Dataset - 300M Tokens

This dataset is prepared for curriculum learning with separate source files for dynamic mixing during training.

## Dataset Information

- **Total Tokens**: {total_tokens:,}
- **Total Documents**: {total_docs:,}
- **Tokenizer**: {stats.get('tokenizer', 'tiktoken cl100k_base (GPT-4)')}
- **Vocabulary Size**: {stats.get('vocab_size', 100277):,}

## Sources

"""
    
    for source, info in stats['sources'].items():
        if info['tokens'] > 0:
            pct = (info['tokens'] / total_tokens) * 100
            readme += f"### {source.title()}\n"
            readme += f"- **Tokens**: {info['tokens']:,} ({pct:.1f}%)\n"
            readme += f"- **Documents**: {info['docs']:,}\n"
            readme += f"- **File**: `{source}_train.bin`\n\n"
    
    readme += """## Files

- `wikitext_train.bin` - WikiText/factual training data
- `ultrachat_train.bin` - UltraChat conversational training data
- `val.bin` - Validation data
- `dataset_stats.json` - Dataset statistics and metadata

## Curriculum Learning Strategy

This dataset uses the Conversational Scholar mix across all pretraining epochs:

| Epoch | WikiText/Factual | UltraChat |
|-------|------------------|-----------|
| 1-8 | 80% | 20% |

## Usage

### Automatic Download (Recommended)

The dataset will be automatically downloaded when you start training:

```bash
python train.py --config config/gpu_training_117m_balanced.yaml
```

### Configuration

In your training config:

```yaml
data:
  dataset_type: "binary"
  train_file: "data/balanced_300m_curriculum/train.bin"
  val_file: "data/balanced_300m_curriculum/val.bin"
  
  huggingface_dataset:
    repo_id: "0x-genesys/mix_wiki_chat_data_300M_tokens_curriculum"
    dataset_name: "balanced_300m_curriculum"
    auto_download: true

training:
  curriculum_learning:
    enabled: true
    sources:
      - "wikitext"
      - "ultrachat"
    epoch_distributions:
      1: [80, 20]
      2: [80, 20]
      3: [80, 20]
      4: [80, 20]
      5: [80, 20]
      6: [80, 20]
      7: [80, 20]
      8: [80, 20]
```

### Manual Download

```python
from huggingface_hub import hf_hub_download

# Download all source files
for source in ['wikitext', 'ultrachat']:
    hf_hub_download(
        repo_id="0x-genesys/mix_wiki_chat_data_300M_tokens_curriculum",
        filename=f"{source}_train.bin",
        repo_type="dataset",
        local_dir="data/balanced_300m_curriculum"
    )

# Download validation file
hf_hub_download(
    repo_id="0x-genesys/mix_wiki_chat_data_300M_tokens_curriculum",
    filename="val.bin",
    repo_type="dataset",
    local_dir="data/balanced_300m_curriculum"
)
```

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{curriculum_300m_2024,
  title={{Conversational Scholar Curriculum Dataset - 300M Tokens}},
  author={{0x-genesys}},
  year={{2024}},
  publisher={{Hugging Face}},
  howpublished={{\\url{{https://huggingface.co/datasets/0x-genesys/mix_wiki_chat_data_300M_tokens_curriculum}}}}
}}
```

## License

This dataset combines data from multiple sources. Please refer to the original licenses:
- WikiText-103: Creative Commons Attribution-ShareAlike License
- UltraChat: MIT License

## Related

- **Model Repository**: [0x-genesys/neo_weights_checkpoints](https://huggingface.co/0x-genesys/neo_weights_checkpoints)
- **Combined Dataset**: [0x-genesys/mix_wiki_chat_data_300M_tokens](https://huggingface.co/datasets/0x-genesys/mix_wiki_chat_data_300M_tokens)
- **Training Code**: [GitHub Repository](https://github.com/yourusername/neo)
"""
    
    return readme


def main():
    parser = argparse.ArgumentParser(description='Upload curriculum dataset to HuggingFace Hub')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/balanced_300m',
        help='Directory containing combined dataset'
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        default='0x-genesys/mix_wiki_chat_data_300M_tokens_curriculum',
        help='HuggingFace repository ID'
    )
    parser.add_argument(
        '--temp-dir',
        type=str,
        default='data/curriculum_temp',
        help='Temporary directory for split files'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Create private repository'
    )
    
    args = parser.parse_args()
    
    prepare_and_upload_curriculum_dataset(
        data_dir=args.data_dir,
        repo_id=args.repo_id,
        temp_dir=args.temp_dir,
        private=args.private
    )


if __name__ == '__main__':
    main()
