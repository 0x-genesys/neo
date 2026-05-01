# HuggingFace Hub Integration Guide

## Overview

The trainer now automatically uploads checkpoints and best models to your HuggingFace Hub repository during training. This provides:

- ✅ **Automatic backup**: Checkpoints saved to cloud
- ✅ **Easy sharing**: Share models with team or community
- ✅ **Version control**: Track model evolution over time
- ✅ **Remote access**: Download models from anywhere

## Setup

### 1. Install HuggingFace Hub

```bash
pip install huggingface_hub
```

### 2. Login to HuggingFace

```bash
huggingface-cli login
```

This will prompt you for your HuggingFace token. Get your token from:
https://huggingface.co/settings/tokens

### 3. Create a Repository (if not exists)

Your repository: `0x-genesys/neo_weights_checkpoints`

If it doesn't exist yet, create it:

```bash
# Using CLI
huggingface-cli repo create neo_weights_checkpoints --type model

# Or via Python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("neo_weights_checkpoints", repo_type="model")
```

Or create it via the web interface:
https://huggingface.co/new

### 4. Configure Training

The config is already set up in `config/gpu_training_117m_1.5gb.yaml`:

```yaml
huggingface_hub:
  enabled: true  # Enable automatic uploads
  repo_id: "0x-genesys/neo_weights_checkpoints"  # Your repo
  upload_best_only: false  # Upload all checkpoints (not just best)
```

## Configuration Options

### Enable/Disable Uploads

```yaml
huggingface_hub:
  enabled: true   # Enable uploads
  # enabled: false  # Disable uploads
```

### Repository ID

```yaml
huggingface_hub:
  repo_id: "username/repo-name"  # Your HF Hub repository
```

Format: `{username}/{repo-name}` or `{organization}/{repo-name}`

### Upload Strategy

**Option 1: Upload All Checkpoints** (Default)
```yaml
huggingface_hub:
  upload_best_only: false
```

Uploads:
- ✅ Best models (when validation improves)
- ✅ Regular checkpoints (every 1000 steps)

**Option 2: Upload Best Models Only**
```yaml
huggingface_hub:
  upload_best_only: true
```

Uploads:
- ✅ Best models (when validation improves)
- ❌ Regular checkpoints (skipped)

## What Gets Uploaded

### Best Models

Uploaded when validation loss improves:

```
best_model_step_1000.pt
best_model_step_2500.pt
best_model_step_5000.pt
...
```

Each file contains:
- Model weights
- Optimizer state
- Scheduler state
- Training metadata (epoch, step, loss)
- Full config

### Regular Checkpoints

Uploaded every `save_interval` steps (default: 1000):

```
checkpoint_step_1000.pt
checkpoint_step_2000.pt
checkpoint_step_3000.pt
...
```

## Training Output

### When Upload Succeeds

```
================================================================================
💾 CHECKPOINT at step 1000
================================================================================
Checkpoint saved: checkpoints/gpu_training_117m_1.5gb_balanced/checkpoint_step_1000.pt
📤 Uploading to HuggingFace Hub: 0x-genesys/neo_weights_checkpoints/checkpoint_step_1000.pt
✅ Uploaded to HuggingFace Hub: https://huggingface.co/0x-genesys/neo_weights_checkpoints/tree/main
✅ Checkpoint saved
================================================================================
```

### When Best Model is Saved

```
================================================================================
🔍 VALIDATION at step 1000
================================================================================
Validation loss: 1.123
✅ New best validation loss! Saving best model...
Checkpoint saved: checkpoints/gpu_training_117m_1.5gb_balanced/best_model.pt
Best model saved: checkpoints/gpu_training_117m_1.5gb_balanced/best_model.pt
📤 Uploading to HuggingFace Hub: 0x-genesys/neo_weights_checkpoints/best_model_step_1000.pt
✅ Uploaded to HuggingFace Hub: https://huggingface.co/0x-genesys/neo_weights_checkpoints/tree/main
================================================================================
```

### When Upload Fails

```
⚠️  Failed to upload to HuggingFace Hub: [error message]
   Continuing training...
```

Training continues even if upload fails!

## Viewing Your Models

### Web Interface

Visit your repository:
```
https://huggingface.co/0x-genesys/neo_weights_checkpoints
```

You'll see:
- All uploaded checkpoints
- File sizes
- Upload timestamps
- Commit history

### Download Models

**Via CLI**:
```bash
huggingface-cli download 0x-genesys/neo_weights_checkpoints best_model_step_5000.pt
```

**Via Python**:
```python
from huggingface_hub import hf_hub_download

checkpoint_path = hf_hub_download(
    repo_id="0x-genesys/neo_weights_checkpoints",
    filename="best_model_step_5000.pt"
)

# Load checkpoint
import torch
checkpoint = torch.load(checkpoint_path)
```

**Via wget**:
```bash
wget https://huggingface.co/0x-genesys/neo_weights_checkpoints/resolve/main/best_model_step_5000.pt
```

## Upload Schedule

With default config (`eval_interval: 1000`, `save_interval: 1000`):

### Timeline

| Step | Event | Uploads |
|------|-------|---------|
| 1,000 | Validation + Checkpoint | Best model (if improved) + checkpoint |
| 2,000 | Validation + Checkpoint | Best model (if improved) + checkpoint |
| 3,000 | Validation + Checkpoint | Best model (if improved) + checkpoint |
| ... | ... | ... |
| 36,621 | Final | Best model (if improved) + checkpoint |

### Expected Uploads

**With `upload_best_only: false`** (default):
- ~36 regular checkpoints (every 1000 steps)
- ~5-10 best models (when validation improves)
- **Total**: ~40-45 files

**With `upload_best_only: true`**:
- 0 regular checkpoints
- ~5-10 best models (when validation improves)
- **Total**: ~5-10 files

## File Sizes

Approximate sizes for 117M model:

| File Type | Size | Notes |
|-----------|------|-------|
| Checkpoint | ~450MB | Full training state |
| Best Model | ~450MB | Full training state |

### Storage Requirements

**Full training** (36,621 steps):
- With `upload_best_only: false`: ~18GB (40 files × 450MB)
- With `upload_best_only: true`: ~2.5GB (5 files × 450MB)

HuggingFace Hub offers:
- **Free tier**: Unlimited public repos
- **Pro tier**: Private repos + more storage

## Troubleshooting

### Issue: "huggingface_hub not installed"

**Error**:
```
⚠️  huggingface_hub not installed. Install with: pip install huggingface_hub
```

**Fix**:
```bash
pip install huggingface_hub
```

### Issue: "Authentication required"

**Error**:
```
⚠️  Failed to upload to HuggingFace Hub: 401 Unauthorized
```

**Fix**:
```bash
huggingface-cli login
```

Enter your token from: https://huggingface.co/settings/tokens

### Issue: "Repository not found"

**Error**:
```
⚠️  Failed to upload to HuggingFace Hub: Repository not found
```

**Fix**:
Create the repository:
```bash
huggingface-cli repo create neo_weights_checkpoints --type model
```

Or update `repo_id` in config to an existing repo.

### Issue: "Upload too slow"

**Symptoms**: Uploads taking very long, blocking training

**Fix 1**: Upload best models only
```yaml
huggingface_hub:
  upload_best_only: true  # Only upload when validation improves
```

**Fix 2**: Increase save interval
```yaml
training:
  save_interval: 2000  # Save less frequently
```

**Fix 3**: Disable during training, upload manually later
```yaml
huggingface_hub:
  enabled: false  # Disable automatic uploads
```

Then upload manually after training:
```bash
huggingface-cli upload 0x-genesys/neo_weights_checkpoints checkpoints/gpu_training_117m_1.5gb_balanced/
```

### Issue: "Disk space full"

**Symptoms**: Local disk fills up with checkpoints

**Fix**: The trainer saves locally AND uploads. You can delete old local checkpoints:

```bash
# Keep only last 5 checkpoints locally
cd checkpoints/gpu_training_117m_1.5gb_balanced/
ls -t checkpoint_step_*.pt | tail -n +6 | xargs rm
```

Models are still safe on HuggingFace Hub!

## Advanced Usage

### Custom Upload Logic

If you need custom upload behavior, modify `_upload_to_hub()` in `src/trainer.py`:

```python
def _upload_to_hub(self, filepath, is_best=False):
    """Custom upload logic."""
    # Your custom logic here
    
    # Example: Only upload every 5000 steps
    if self.global_step % 5000 != 0 and not is_best:
        return
    
    # ... rest of upload code
```

### Upload to Multiple Repos

```python
def _upload_to_hub(self, filepath, is_best=False):
    """Upload to multiple repositories."""
    repos = [
        "0x-genesys/neo_weights_checkpoints",
        "0x-genesys/neo_backup"
    ]
    
    for repo_id in repos:
        # Upload to each repo
        api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo=filepath.name,
            repo_id=repo_id,
            repo_type="model"
        )
```

### Add Metadata

Upload additional metadata files:

```python
# After uploading checkpoint
metadata = {
    'step': self.global_step,
    'epoch': self.epoch,
    'loss': self.best_val_loss,
    'config': self.config
}

import json
metadata_path = self.checkpoint_dir / 'metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

api.upload_file(
    path_or_fileobj=str(metadata_path),
    path_in_repo='metadata.json',
    repo_id=repo_id,
    repo_type="model"
)
```

## Best Practices

### 1. Start with Best Models Only

For initial training runs:
```yaml
huggingface_hub:
  upload_best_only: true
```

This saves bandwidth and storage while still backing up important checkpoints.

### 2. Monitor Storage

Check your HuggingFace Hub storage:
https://huggingface.co/settings/account

### 3. Clean Up Old Checkpoints

After training completes, you can delete intermediate checkpoints:

```python
from huggingface_hub import HfApi

api = HfApi()

# List all files
files = api.list_repo_files("0x-genesys/neo_weights_checkpoints")

# Delete old checkpoints (keep only best and final)
for file in files:
    if file.startswith("checkpoint_step_") and not file.endswith("36621.pt"):
        api.delete_file(
            path_in_repo=file,
            repo_id="0x-genesys/neo_weights_checkpoints"
        )
```

### 4. Use Private Repos for Experiments

For experimental runs:
```bash
huggingface-cli repo create neo_experiments --type model --private
```

Then update config:
```yaml
huggingface_hub:
  repo_id: "0x-genesys/neo_experiments"
```

### 5. Tag Important Checkpoints

After training, tag important checkpoints:

```bash
# Via web interface: Add tags like "best", "final", "v1.0"
# Or via API
api.create_tag(
    repo_id="0x-genesys/neo_weights_checkpoints",
    tag="v1.0",
    tag_message="First production model",
    revision="main"
)
```

## Summary

### Quick Start

1. **Install**: `pip install huggingface_hub`
2. **Login**: `huggingface-cli login`
3. **Train**: Checkpoints automatically upload!

### Configuration

```yaml
huggingface_hub:
  enabled: true  # Enable uploads
  repo_id: "0x-genesys/neo_weights_checkpoints"
  upload_best_only: false  # Upload all checkpoints
```

### What You Get

- ✅ Automatic cloud backup
- ✅ Version control for models
- ✅ Easy sharing and collaboration
- ✅ Remote access from anywhere
- ✅ No manual upload needed!

Your models are now automatically backed up to HuggingFace Hub! 🚀
