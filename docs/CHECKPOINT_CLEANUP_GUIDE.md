# Checkpoint Cleanup Guide

## Overview

This guide explains how to manage checkpoint files to save disk space, especially important on Kaggle TPU VMs with limited storage.

## Automatic Cleanup (TPU Trainer)

The TPU trainer automatically deletes regular checkpoints after uploading to HuggingFace Hub:

- ✅ **Deleted after upload**: `checkpoint_step_*.pt` files
- 💾 **Kept locally**: `best_model_*.pt` files
- 💾 **Kept locally**: Error/interrupted checkpoints

This happens automatically when `huggingface_hub.enabled: true` in your config.

## Manual Cleanup Script

Use the cleanup script to manually clean up checkpoints:

### Basic Usage

```bash
# Dry run (see what would be deleted)
python scripts/cleanup_checkpoints.py \
    --checkpoint-dir checkpoints/auto_training_117m_balanced \
    --dry-run

# Delete checkpoints
python scripts/cleanup_checkpoints.py \
    --checkpoint-dir checkpoints/auto_training_117m_balanced
```

### Verify on HuggingFace Hub

Before deleting, verify files exist on HuggingFace Hub:

```bash
python scripts/cleanup_checkpoints.py \
    --checkpoint-dir checkpoints/auto_training_117m_balanced \
    --verify-hub \
    --repo-id 0x-genesys/neo_weights_checkpoints
```

This will:
1. Check which checkpoint files exist on HuggingFace Hub
2. Only delete files that are confirmed to be on the Hub
3. Skip files not found on the Hub (with warning)

## What Gets Deleted

### ✅ Deleted (Regular Checkpoints)
- `checkpoint_step_1000.pt`
- `checkpoint_step_2000.pt`
- `checkpoint_step_3000.pt`
- etc.

### 💾 Kept (Best Models)
- `best_model_step_5000.pt`
- `best_model_step_10000.pt`
- etc.

### 💾 Kept (Special Checkpoints)
- `error_checkpoint.pt`
- `interrupted_checkpoint.pt`
- `emergency_checkpoint.pt`

## Disk Space Savings

Typical checkpoint sizes:
- **117M model**: ~500 MB per checkpoint
- **774M model**: ~3 GB per checkpoint

If you save every 1000 steps for 10K steps:
- **117M**: ~5 GB saved
- **774M**: ~30 GB saved

## Best Practices

### 1. Enable HuggingFace Hub Upload

In your config:

```yaml
huggingface_hub:
  enabled: true
  repo_id: "your-username/your-model-repo"
```

This enables automatic cleanup after upload.

### 2. Adjust Save Interval

For limited disk space, increase save interval:

```yaml
checkpoint:
  save_interval: 2000  # Save every 2000 steps instead of 1000
```

### 3. Regular Cleanup

On Kaggle, run cleanup periodically:

```bash
# Every few hours during long training
python scripts/cleanup_checkpoints.py \
    --checkpoint-dir checkpoints/auto_training_117m_balanced \
    --verify-hub \
    --repo-id 0x-genesys/neo_weights_checkpoints
```

### 4. Monitor Disk Usage

Check disk usage:

```bash
# Total disk usage
df -h

# Checkpoint directory size
du -sh checkpoints/*/

# Individual checkpoint sizes
ls -lh checkpoints/auto_training_117m_balanced/*.pt
```

## Kaggle-Specific Tips

### Disk Space Limits

Kaggle TPU VMs have limited disk space:
- **Working directory**: ~100 GB
- **Persistent storage**: None (ephemeral)

### During Training

1. Enable HuggingFace Hub upload
2. Set reasonable save intervals (1000-2000 steps)
3. Run cleanup script every few hours

### After Training

1. Upload final best model to Hub
2. Clean up all regular checkpoints
3. Download best model locally if needed

```bash
# Upload final model
python scripts/upload_to_hub.py \
    --checkpoint checkpoints/auto_training_117m_balanced/best_model_step_10000.pt \
    --repo-id 0x-genesys/neo_weights_checkpoints

# Clean up
python scripts/cleanup_checkpoints.py \
    --checkpoint-dir checkpoints/auto_training_117m_balanced \
    --verify-hub \
    --repo-id 0x-genesys/neo_weights_checkpoints
```

## Troubleshooting

### "Not found on Hub" warnings

If files aren't on Hub:
1. Check your HuggingFace token is set
2. Verify repo_id is correct
3. Check upload logs for errors
4. Don't use `--verify-hub` if uploads failed

### Permission errors

```bash
# Make script executable
chmod +x scripts/cleanup_checkpoints.py

# Or run with python
python scripts/cleanup_checkpoints.py --help
```

### Accidental deletion

Best models and special checkpoints are never deleted automatically. If you need to recover:
1. Check HuggingFace Hub for uploaded checkpoints
2. Download from Hub using `scripts/download_from_hub.py`

## Summary

- ✅ **Automatic**: TPU trainer deletes after Hub upload
- 🔧 **Manual**: Use cleanup script for more control
- 🔍 **Safe**: Verify on Hub before deleting
- 💾 **Protected**: Best models always kept
- 📊 **Transparent**: Dry run shows what will be deleted
