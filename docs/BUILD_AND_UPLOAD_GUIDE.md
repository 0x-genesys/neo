# Build and Upload Curriculum Dataset - Step-by-Step Guide

## Overview

This guide walks you through building and uploading your own curriculum dataset to HuggingFace Hub.

## Prerequisites

1. **HuggingFace Account**: Create account at [huggingface.co](https://huggingface.co)
2. **HuggingFace Token**: Get from [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. **Combined Dataset**: Have `data/balanced_300m/` with `train.bin` and `dataset_stats.json`

## Step 1: Login to HuggingFace Hub

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Login with your token
huggingface-cli login
```

Enter your HuggingFace token when prompted.

## Step 2: Prepare Curriculum Dataset

This splits the combined dataset into separate source files:

```bash
python scripts/prepare_curriculum_dataset.py \
    --data-dir data/balanced_300m \
    --output-dir data/curriculum_temp
```

**What it does**:
1. Loads `data/balanced_300m/train.bin` (combined dataset)
2. Reads `dataset_stats.json` to find source boundaries
3. Splits into separate files:
   - `wikitext_train.bin` (102M tokens, 387.9 MB)
   - `stack_train.bin` (48M tokens, 182.4 MB)
   - `ultrachat_train.bin` (208M tokens, 790.5 MB)
4. Copies `val.bin` (38M tokens, 38.1 MB)
5. Creates `dataset_stats.json` and `README.md`

**Expected output**:
```
================================================================================
Curriculum Dataset Preparation
================================================================================
Source: data/balanced_300m
Output: data/curriculum_temp
================================================================================

Step 1: Loading dataset statistics...
✅ Loaded stats from data/balanced_300m/dataset_stats.json
  wikitext    : 102,000,024 tokens
  stack       : 48,000,366 tokens
  ultrachat   : 208,000,134 tokens

Step 2: Loading combined dataset...
✅ Loaded 358,000,524 tokens from data/balanced_300m/train.bin

Step 3: Splitting into source files...

  Processing wikitext...
    Range: 0 - 102,000,024
    Output: data/curriculum_temp/wikitext_train.bin
    ✅ Saved 102,000,024 tokens

  Processing stack...
    Range: 102,000,024 - 150,000,390
    Output: data/curriculum_temp/stack_train.bin
    ✅ Saved 48,000,366 tokens

  Processing ultrachat...
    Range: 150,000,390 - 358,000,524
    Output: data/curriculum_temp/ultrachat_train.bin
    ✅ Saved 208,000,134 tokens

Step 4: Copying validation file...
✅ Copied 38,000,000 tokens to data/curriculum_temp/val.bin

Step 5: Saving dataset statistics...
✅ Saved stats to data/curriculum_temp/dataset_stats.json

Step 6: Creating README...
✅ Created README.md

================================================================================
✅ Preparation complete!
================================================================================

Files created:
  - wikitext_train.bin (387.9 MB)
  - stack_train.bin (182.4 MB)
  - ultrachat_train.bin (790.5 MB)
  - val.bin (38.1 MB)
  - dataset_stats.json (1.2 KB)
  - README.md

Total size: 1.4 GB
```

## Step 3: Upload to HuggingFace Hub

This uploads the curriculum dataset to HuggingFace Hub:

```bash
python scripts/upload_curriculum_dataset.py \
    --data-dir data/balanced_300m \
    --repo-id 0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum \
    --temp-dir data/curriculum_temp
```

**What it does**:
1. Runs Step 2 (prepare dataset) if not already done
2. Creates HuggingFace repository (if doesn't exist)
3. Uploads all files:
   - `wikitext_train.bin`
   - `stack_train.bin`
   - `ultrachat_train.bin`
   - `val.bin`
   - `dataset_stats.json`
   - `README.md`
4. Displays dataset URL

**Expected output**:
```
================================================================================
Curriculum Dataset Upload to HuggingFace Hub
================================================================================
Source: data/balanced_300m
Destination: 0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum
Temp directory: data/curriculum_temp
================================================================================

[Steps 1-6 from preparation...]

Step 7: Uploading to HuggingFace Hub...
Repository: 0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum

✅ Repository created/verified: 0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum

Uploading 6 files...
  Uploading dataset_stats.json...
    ✅ Uploaded
  Uploading README.md...
    ✅ Uploaded
  Uploading val.bin...
    ✅ Uploaded
  Uploading wikitext_train.bin...
    ✅ Uploaded
  Uploading stack_train.bin...
    ✅ Uploaded
  Uploading ultrachat_train.bin...
    ✅ Uploaded

================================================================================
✅ Upload complete!
================================================================================

Dataset URL: https://huggingface.co/datasets/0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum

Files uploaded:
  - dataset_stats.json
  - README.md
  - val.bin
  - wikitext_train.bin
  - stack_train.bin
  - ultrachat_train.bin

To use in training:
  1. Update config:
     huggingface_dataset:
       repo_id: "0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum"
       dataset_name: "balanced_300m_curriculum"
  2. Run training:
     python train.py --config config/gpu_training_117m_balanced.yaml
```

## Step 4: Verify Upload

Visit your dataset page:
```
https://huggingface.co/datasets/YOUR_USERNAME/YOUR_DATASET_NAME
```

You should see:
- ✅ README with dataset information
- ✅ Files tab with all 6 files
- ✅ Dataset card with usage instructions

## Step 5: Update Configuration

The config is already set up for the default repository. If you uploaded to a different repository, update `config/gpu_training_117m_balanced.yaml`:

```yaml
data:
  huggingface_dataset:
    repo_id: "YOUR_USERNAME/YOUR_DATASET_NAME"  # Update this
    dataset_name: "balanced_300m_curriculum"
    auto_download: true
```

## Step 6: Test Download

Test that the dataset can be downloaded:

```bash
python -m src.dataset_downloader \
    YOUR_USERNAME/YOUR_DATASET_NAME \
    --dataset-name balanced_300m_curriculum
```

**Expected output**:
```
================================================================================
DOWNLOADING DATASET: balanced_300m_curriculum
================================================================================
Repository: YOUR_USERNAME/YOUR_DATASET_NAME
Local path: data/balanced_300m_curriculum

📋 Checking repository contents...
Found 6 dataset files:
  - wikitext_train.bin
  - stack_train.bin
  - ultrachat_train.bin
  - val.bin
  - dataset_stats.json
  - README.md

📥 Downloading wikitext_train.bin...
✅ Downloaded wikitext_train.bin: 387.9 MB

📥 Downloading stack_train.bin...
✅ Downloaded stack_train.bin: 182.4 MB

📥 Downloading ultrachat_train.bin...
✅ Downloaded ultrachat_train.bin: 790.5 MB

📥 Downloading val.bin...
✅ Downloaded val.bin: 38.1 MB

📥 Downloading dataset_stats.json...
✅ Downloaded dataset_stats.json: 1.2 KB

📥 Downloading README.md...
✅ Downloaded README.md: 5.2 KB

✅ Dataset download complete!
📁 Files saved to: data/balanced_300m_curriculum
```

## Step 7: Start Training

Now you can start training with curriculum learning:

```bash
python train.py --config config/gpu_training_117m_balanced.yaml
```

The dataset will be automatically downloaded if not already present.

## Troubleshooting

### Issue: Authentication Failed

**Error**: `401 Unauthorized`

**Solution**: Login to HuggingFace Hub:
```bash
huggingface-cli login
```

### Issue: Repository Already Exists

**Error**: `Repository already exists`

**Solution**: This is fine! The script will use the existing repository. To force re-upload:
```bash
# Delete repository first (careful!)
# Then re-run upload script
```

### Issue: Upload Failed

**Error**: `Failed to upload file`

**Solution**: Check:
1. Internet connection
2. HuggingFace Hub status
3. File permissions
4. Disk space

### Issue: File Not Found

**Error**: `FileNotFoundError: data/balanced_300m/train.bin not found`

**Solution**: Make sure you have the combined dataset:
```bash
# Check if files exist
ls -lh data/balanced_300m/

# If missing, download or build the dataset
python scripts/prepare_balanced_dataset.py
```

## Advanced Options

### Custom Repository Name

```bash
python scripts/upload_curriculum_dataset.py \
    --data-dir data/balanced_300m \
    --repo-id YOUR_USERNAME/custom_curriculum_dataset \
    --temp-dir data/curriculum_temp
```

### Private Repository

```bash
python scripts/upload_curriculum_dataset.py \
    --data-dir data/balanced_300m \
    --repo-id YOUR_USERNAME/YOUR_DATASET_NAME \
    --temp-dir data/curriculum_temp \
    --private
```

### Custom Temp Directory

```bash
python scripts/upload_curriculum_dataset.py \
    --data-dir data/balanced_300m \
    --repo-id YOUR_USERNAME/YOUR_DATASET_NAME \
    --temp-dir /path/to/custom/temp
```

## Script Options

### `prepare_curriculum_dataset.py`

```bash
python scripts/prepare_curriculum_dataset.py --help

Options:
  --data-dir TEXT     Directory containing combined dataset (default: data/balanced_300m)
  --output-dir TEXT   Output directory for split files (default: data/curriculum_temp)
```

### `upload_curriculum_dataset.py`

```bash
python scripts/upload_curriculum_dataset.py --help

Options:
  --data-dir TEXT     Directory containing combined dataset (default: data/balanced_300m)
  --repo-id TEXT      HuggingFace repository ID (default: 0x-genesys/mix_wiki_code_chat_data_300M_tokens_curriculum)
  --temp-dir TEXT     Temporary directory for split files (default: data/curriculum_temp)
  --private           Create private repository (default: False)
```

## File Structure

### Before (Combined Dataset)
```
data/balanced_300m/
├── train.bin              # 358M tokens (1.4 GB)
├── val.bin                # 38M tokens (38.1 MB)
└── dataset_stats.json     # Statistics
```

### After (Curriculum Dataset)
```
data/curriculum_temp/
├── wikitext_train.bin     # 102M tokens (387.9 MB)
├── stack_train.bin        # 48M tokens (182.4 MB)
├── ultrachat_train.bin    # 208M tokens (790.5 MB)
├── val.bin                # 38M tokens (38.1 MB)
├── dataset_stats.json     # Statistics (1.2 KB)
└── README.md              # Documentation (5.2 KB)
```

### On HuggingFace Hub
```
YOUR_USERNAME/YOUR_DATASET_NAME/
├── wikitext_train.bin     # 387.9 MB
├── stack_train.bin        # 182.4 MB
├── ultrachat_train.bin    # 790.5 MB
├── val.bin                # 38.1 MB
├── dataset_stats.json     # 1.2 KB
└── README.md              # 5.2 KB
```

## Summary

1. **Login**: `huggingface-cli login`
2. **Prepare**: `python scripts/prepare_curriculum_dataset.py`
3. **Upload**: `python scripts/upload_curriculum_dataset.py`
4. **Verify**: Visit HuggingFace Hub page
5. **Update Config**: Edit `config/gpu_training_117m_balanced.yaml`
6. **Test**: `python -m src.dataset_downloader YOUR_REPO_ID`
7. **Train**: `python train.py --config config/gpu_training_117m_balanced.yaml`

**Total time**: ~10-15 minutes (depending on upload speed)

## See Also

- [CURRICULUM_QUICK_REFERENCE.md](CURRICULUM_QUICK_REFERENCE.md) - Quick reference
- [CURRICULUM_LEARNING_SETUP.md](docs/CURRICULUM_LEARNING_SETUP.md) - Setup guide
- [CURRICULUM_LEARNING_GUIDE.md](docs/CURRICULUM_LEARNING_GUIDE.md) - Comprehensive guide
- [README.md](README.md) - Main documentation
