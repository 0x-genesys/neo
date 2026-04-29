# HuggingFace Hub Integration - Quick Setup

## What This Does

Automatically uploads your training checkpoints and best models to HuggingFace Hub:
- ✅ **Best models** when validation improves
- ✅ **Regular checkpoints** every 1000 steps
- ✅ **Automatic backup** to cloud
- ✅ **Easy sharing** with team

Your repository: https://huggingface.co/0x-genesys/neo_weights_checkpoints

## Quick Setup (3 Steps)

### Option 1: Automated Setup

```bash
bash scripts/setup_huggingface_hub.sh
```

This will:
1. Install `huggingface_hub`
2. Login to HuggingFace
3. Verify/create your repository
4. Confirm everything is ready

### Option 2: Manual Setup

```bash
# 1. Install
pip install huggingface_hub

# 2. Login (get token from https://huggingface.co/settings/tokens)
huggingface-cli login

# 3. Create repo (if needed)
huggingface-cli repo create neo_weights_checkpoints --type model
```

## Configuration

Already configured in `config/gpu_training_117m_1.5gb.yaml`:

```yaml
huggingface_hub:
  enabled: true  # Uploads enabled
  repo_id: "0x-genesys/neo_weights_checkpoints"
  upload_best_only: false  # Upload all checkpoints
```

### Options

**Upload everything** (default):
```yaml
upload_best_only: false  # Best models + regular checkpoints
```

**Upload best models only** (saves bandwidth):
```yaml
upload_best_only: true  # Only when validation improves
```

**Disable uploads**:
```yaml
enabled: false  # No uploads
```

## What Gets Uploaded

### Best Models (when validation improves)
```
best_model_step_1000.pt   (~450MB)
best_model_step_2500.pt   (~450MB)
best_model_step_5000.pt   (~450MB)
...
```

### Regular Checkpoints (every 1000 steps)
```
checkpoint_step_1000.pt   (~450MB)
checkpoint_step_2000.pt   (~450MB)
checkpoint_step_3000.pt   (~450MB)
...
```

## Training Output

When checkpoint is saved and uploaded:

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

When best model is saved:

```
================================================================================
🔍 VALIDATION at step 1000
================================================================================
Validation loss: 1.123
✅ New best validation loss! Saving best model...
📤 Uploading to HuggingFace Hub: 0x-genesys/neo_weights_checkpoints/best_model_step_1000.pt
✅ Uploaded to HuggingFace Hub: https://huggingface.co/0x-genesys/neo_weights_checkpoints/tree/main
================================================================================
```

## Viewing Your Models

### Web Interface

Visit: https://huggingface.co/0x-genesys/neo_weights_checkpoints

You'll see:
- All uploaded files
- Upload timestamps
- File sizes
- Commit history

### Download Models

**CLI**:
```bash
huggingface-cli download 0x-genesys/neo_weights_checkpoints best_model_step_5000.pt
```

**Python**:
```python
from huggingface_hub import hf_hub_download

checkpoint = hf_hub_download(
    repo_id="0x-genesys/neo_weights_checkpoints",
    filename="best_model_step_5000.pt"
)
```

**Direct URL**:
```
https://huggingface.co/0x-genesys/neo_weights_checkpoints/resolve/main/best_model_step_5000.pt
```

## Storage Requirements

### Full Training (36,621 steps)

**With `upload_best_only: false`**:
- ~36 regular checkpoints
- ~5-10 best models
- **Total**: ~18GB

**With `upload_best_only: true`**:
- 0 regular checkpoints
- ~5-10 best models
- **Total**: ~2.5GB

HuggingFace Hub offers unlimited storage for public repos!

## Troubleshooting

### "huggingface_hub not installed"

```bash
pip install huggingface_hub
```

### "Authentication required"

```bash
huggingface-cli login
```

Get token from: https://huggingface.co/settings/tokens

### "Repository not found"

```bash
huggingface-cli repo create neo_weights_checkpoints --type model
```

### Uploads too slow

**Option 1**: Upload best models only
```yaml
upload_best_only: true
```

**Option 2**: Disable during training, upload manually later
```yaml
enabled: false
```

Then after training:
```bash
huggingface-cli upload 0x-genesys/neo_weights_checkpoints checkpoints/
```

## Training Workflow

### Start Training

```bash
# Setup HuggingFace Hub (one-time)
bash scripts/setup_huggingface_hub.sh

# Start training (uploads automatically)
python train.py --config config/gpu_training_117m_1.5gb.yaml --multi-gpu
```

### Monitor Progress

**Local**:
```bash
# Watch training
watch -n 1 nvidia-smi

# View logs
tensorboard --logdir logs/gpu_training_117m_1.5gb_balanced
```

**Remote**:
```
# View uploaded models
https://huggingface.co/0x-genesys/neo_weights_checkpoints
```

### After Training

Your models are automatically backed up! You can:

1. **Share with team**: Send them the HF Hub link
2. **Download anywhere**: Use `huggingface-cli download`
3. **Version control**: All checkpoints are tracked
4. **Clean up local**: Delete local checkpoints, keep HF Hub copies

## Best Practices

### 1. Start with Best Models Only

For initial runs:
```yaml
upload_best_only: true  # Saves bandwidth
```

### 2. Monitor Storage

Check: https://huggingface.co/settings/account

### 3. Tag Important Models

After training, tag your best model:
```bash
# Via web interface or API
# Tag as "v1.0", "production", etc.
```

### 4. Clean Up Old Checkpoints

Keep only important checkpoints:
- Best model
- Final checkpoint
- Key milestones

Delete intermediate checkpoints via web interface.

## Documentation

Full guide: `docs/HUGGINGFACE_HUB_INTEGRATION.md`

## Summary

### Setup
```bash
bash scripts/setup_huggingface_hub.sh
```

### Train
```bash
python train.py --config config/gpu_training_117m_1.5gb.yaml --multi-gpu
```

### View
```
https://huggingface.co/0x-genesys/neo_weights_checkpoints
```

Your models are now automatically backed up to the cloud! 🚀
