# Kaggle TPU Quick Start Guide

**Updated**: April 30, 2026 - Fixed SliceBuilder errors and added automatic checkpoint cleanup

## Prerequisites

1. Kaggle account with TPU access
2. HuggingFace account and token (for checkpoint uploads)

## Setup (5 minutes)

### 1. Create Kaggle Notebook

1. Go to kaggle.com/code
2. Click "New Notebook"
3. Settings → Accelerator → **TPU v3-8**
4. Settings → Internet → **On**

### 2. Clone Repository

```bash
!git clone https://github.com/your-username/neo.git
%cd neo
```

### 3. Install Dependencies

```bash
# Install torch_xla for TPU support
!pip install torch_xla -f https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-2.5.0-cp312-cp312-linux_x86_64.whl

# Install other dependencies
!pip install -r requirements.txt
```

### 4. Set HuggingFace Token

```bash
# In Kaggle notebook
import os
os.environ['HF_TOKEN'] = 'your_token_here'

# Or use Kaggle secrets (recommended)
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
os.environ['HF_TOKEN'] = user_secrets.get_secret("HF_TOKEN")
```

## Training (1 command)

### Start Training

```bash
!python train.py \
    --config config/auto_training_117m_balanced.yaml \
    --tpu
```

That's it! Training will:
- ✅ Auto-detect TPU configuration (no manual setup needed)
- ✅ Use all 8 TPU cores
- ✅ Save checkpoints every 1000 steps
- ✅ Upload checkpoints to HuggingFace Hub
- ✅ Automatically delete old checkpoints to save disk space
- ✅ Keep best models locally

## Expected Output

```
Loading config from config/auto_training_117m_balanced.yaml

✅ TPU training enabled!
   torch_xla version: 2.5.0
   TPU cores: 8
   Note: TPU training uses XLA compiler for optimization
   Note: Training will spawn across all 8 cores

================================================================================
Configuration:
================================================================================
Dataset: 0x-genesys/fineweb-edu-curriculum-10bt
Model: 12 layers, 768 dim, 12 heads
Context length: 1024
Batch size: 32
Learning rate: 0.0003
Max epochs: 3
Max steps: 10000
================================================================================

Loading data...
Creating model...

================================================================================
🚀 TPU Trainer Initialized
================================================================================
TPU cores: 8
torch_xla version: 2.5.0
================================================================================

🚀 Starting TPU training on all available TPU cores...

Epoch 0 | Step 100 | Loss: 3.2456 | LR: 3.00e-04
Epoch 0 | Step 200 | Loss: 2.8934 | LR: 3.00e-04
...
```

## Monitoring

### Check Disk Space

```bash
# In a separate cell
!df -h
```

### View Checkpoints

```bash
!ls -lh checkpoints/auto_training_117m_balanced/
```

### Manual Cleanup (if needed)

```bash
# Dry run to see what would be deleted
!python scripts/cleanup_checkpoints.py \
    --checkpoint-dir checkpoints/auto_training_117m_balanced \
    --dry-run

# Delete old checkpoints (keeps best models)
!python scripts/cleanup_checkpoints.py \
    --checkpoint-dir checkpoints/auto_training_117m_balanced \
    --verify-hub \
    --repo-id 0x-genesys/neo_weights_checkpoints
```

## Resume Training

### From Local Checkpoint

```bash
!python train.py \
    --config config/auto_training_117m_balanced.yaml \
    --tpu \
    --resume checkpoints/auto_training_117m_balanced/checkpoint_step_5000.pt
```

### From HuggingFace Hub

```bash
!python train.py \
    --config config/auto_training_117m_balanced.yaml \
    --tpu \
    --resume-remote checkpoint_step_5000.pt \
    --model-repo 0x-genesys/neo_weights_checkpoints
```

## Configuration

### Adjust Batch Size

```bash
!python train.py \
    --config config/auto_training_117m_balanced.yaml \
    --tpu \
    --batch-size 16
```

### Adjust Learning Rate

```bash
!python train.py \
    --config config/auto_training_117m_balanced.yaml \
    --tpu \
    --lr 3e-4
```

### Change Max Steps

```bash
!python train.py \
    --config config/auto_training_117m_balanced.yaml \
    --tpu \
    --epochs 5
```

## Troubleshooting

### TPU Not Detected

```bash
# Check TPU availability
!python scripts/check_tpu.py

# Verify torch_xla installation
!python -c "import torch_xla; print(torch_xla.__version__)"

# Check TPU devices
!python -c "import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices())"
```

### Out of Disk Space

```bash
# Check disk usage
!df -h

# Clean up checkpoints
!python scripts/cleanup_checkpoints.py \
    --checkpoint-dir checkpoints/auto_training_117m_balanced \
    --verify-hub \
    --repo-id 0x-genesys/neo_weights_checkpoints

# Remove old logs
!rm -rf logs/old_runs/
```

### Training Interrupted

Training automatically saves checkpoints on interruption:

```bash
# Resume from interrupted checkpoint
!python train.py \
    --config config/auto_training_117m_balanced.yaml \
    --tpu \
    --resume checkpoints/auto_training_117m_balanced/interrupted_checkpoint.pt
```

### SliceBuilder Errors (FIXED)

If you see:
```
RuntimeError: TPU initialization failed: Invalid --2a886c8_slice_builder_worker_addresses specified.
```

**This is now fixed!** The latest code removes the problematic environment variables.

If you still see this error:
1. Pull latest code: `!git pull`
2. Verify `train.py` doesn't set `TPU_PROCESS_ADDRESSES`
3. Restart kernel and try again

## Best Practices

### 1. Enable HuggingFace Hub Upload

In `config/auto_training_117m_balanced.yaml`:

```yaml
huggingface_hub:
  enabled: true
  repo_id: "your-username/your-model-repo"
```

This enables automatic checkpoint cleanup.

### 2. Monitor Disk Space

```bash
# Run in background
!while true; do df -h | grep /kaggle/working; sleep 300; done &
```

### 3. Regular Cleanup

```bash
# Every few hours during long training
!python scripts/cleanup_checkpoints.py \
    --checkpoint-dir checkpoints/auto_training_117m_balanced \
    --verify-hub \
    --repo-id 0x-genesys/neo_weights_checkpoints
```

### 4. Save Best Model

After training completes:

```bash
# Upload best model
!huggingface-cli upload 0x-genesys/neo_weights_checkpoints \
    checkpoints/auto_training_117m_balanced/best_model_*.pt
```

## Complete Example Notebook

```python
# Cell 1: Setup
!git clone https://github.com/your-username/neo.git
%cd neo
!pip install torch_xla -f https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-2.5.0-cp312-cp312-linux_x86_64.whl
!pip install -r requirements.txt

# Cell 2: Configure HuggingFace
import os
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
os.environ['HF_TOKEN'] = user_secrets.get_secret("HF_TOKEN")

# Cell 3: Start Training
!python train.py \
    --config config/auto_training_117m_balanced.yaml \
    --tpu

# Cell 4: Monitor (run in parallel)
!watch -n 60 'df -h | grep /kaggle/working'

# Cell 5: Cleanup (run periodically)
!python scripts/cleanup_checkpoints.py \
    --checkpoint-dir checkpoints/auto_training_117m_balanced \
    --verify-hub \
    --repo-id 0x-genesys/neo_weights_checkpoints
```

## Performance Expectations

### 117M Model on TPU v3-8

- **Throughput**: ~2000-3000 tokens/sec
- **Time per 1000 steps**: ~30-45 minutes
- **10K steps**: ~5-7 hours
- **Checkpoint size**: ~500 MB
- **Disk usage**: ~1-2 GB (with automatic cleanup)

### 774M Model on TPU v3-8

- **Throughput**: ~500-800 tokens/sec
- **Time per 1000 steps**: ~2-3 hours
- **10K steps**: ~20-30 hours
- **Checkpoint size**: ~3 GB
- **Disk usage**: ~5-10 GB (with automatic cleanup)

## What's New (April 30, 2026)

### ✅ Fixed TPU Initialization
- Removed problematic environment variables
- PJRT now auto-detects TPU configuration
- No more SliceBuilder errors

### ✅ Automatic Checkpoint Cleanup
- Deletes old checkpoints after Hub upload
- Keeps best models locally
- Saves 90% disk space

### ✅ Better Documentation
- `KAGGLE_TPU_ENV_FIX.md` - Technical details
- `docs/CHECKPOINT_CLEANUP_GUIDE.md` - Disk management
- `QUICK_COMMANDS.md` - Command reference
- This guide - Quick start

## Resources

- **Documentation**: See `docs/` folder
- **Examples**: See `examples/` folder
- **Troubleshooting**: See `KAGGLE_TPU_TROUBLESHOOTING.md`
- **Commands**: See `QUICK_COMMANDS.md`

## Support

If you encounter issues:

1. Check `KAGGLE_TPU_TROUBLESHOOTING.md`
2. Verify TPU is enabled in notebook settings
3. Check disk space: `!df -h`
4. Review logs: `!tail -100 logs/training.log`
5. Open an issue on GitHub

## Summary

**3 Steps to Train on Kaggle TPU:**

1. **Setup**: Clone repo, install dependencies, set HF token
2. **Train**: `python train.py --config config/auto_training_117m_balanced.yaml --tpu`
3. **Monitor**: Check disk space, clean up if needed

**That's it!** The system handles:
- ✅ TPU configuration (automatic)
- ✅ Multi-core training (automatic)
- ✅ Checkpoint uploads (automatic)
- ✅ Disk space management (automatic)

Happy training! 🚀
