# Checkpoint Compatibility Guide

**Cross-Hardware Checkpoint Resumption**

## Overview

Your PyTorch `.pt` checkpoint files are **fully compatible** across all hardware types:
- ✅ GPU → GPU (same or different GPU types)
- ✅ GPU → TPU (your use case!)
- ✅ TPU → GPU
- ✅ GPU → CPU
- ✅ CPU → GPU/TPU

## Your Scenario: GPU Checkpoint → TPU Training

You have a checkpoint from GPU training at step 7500 (`best_model_step_7500.pt`). You want to continue training on Kaggle TPU.

### ✅ Yes, This Works Perfectly!

```bash
# On Kaggle with TPU enabled
python train.py --config config/auto_training_117m_balanced.yaml \
    --resume-remote best_model_step_7500.pt \
    --tpu
```

## What Happens Behind the Scenes

### 1. Checkpoint Structure

Your `.pt` file contains:
```python
{
    'model_state_dict': {...},      # Model weights (device-independent)
    'optimizer_state_dict': {...},  # Optimizer state (AdamW states)
    'scheduler_state_dict': {...},  # Learning rate scheduler state
    'global_step': 7500,            # Training step counter
    'epoch': 2,                     # Epoch counter
    'best_val_loss': 3.245,         # Best validation loss so far
    'config': {...}                 # Training configuration
}
```

### 2. Loading Process on TPU

**Step 1: Download from HuggingFace Hub**
```python
# train.py handles this automatically
from src.remote_model_loader import get_remote_checkpoint_path
local_path = get_remote_checkpoint_path('best_model_step_7500.pt', 
                                        '0x-genesys/neo_weights_checkpoints')
```

**Step 2: Load on CPU First**
```python
# TPU trainer loads checkpoint on CPU before spawning
checkpoint = torch.load(local_path, map_location='cpu')
```

**Step 3: Load Model Weights**
```python
# Handles DataParallel wrapper if present
state_dict = checkpoint['model_state_dict']
if any(k.startswith('module.') for k in state_dict.keys()):
    # Remove 'module.' prefix from multi-GPU training
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
```

**Step 4: Extract Training State**
```python
global_step = checkpoint['global_step']  # 7500
epoch = checkpoint['epoch']              # 2
best_val_loss = checkpoint['best_val_loss']  # 3.245
```

**Step 5: Spawn TPU Training**
```python
# Model is moved to TPU device in each core
xmp.spawn(_mp_fn, nprocs=8, start_method='fork')
```

**Step 6: Load Optimizer/Scheduler State**
```python
# After creating optimizer on TPU
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
```

**Step 7: Continue Training**
```python
# Training continues from step 7500
# Epoch continues from epoch 2
# Best validation loss is preserved
```

### 3. Hardware Adaptation

The system automatically adjusts for TPU:

**GPU Settings (your checkpoint)**:
- Batch size: 8
- Gradient accumulation: 16
- Effective batch: 128
- Learning rate: 0.0002

**TPU Settings (auto-adjusted)**:
- Batch size: 128 (16x larger)
- Gradient accumulation: 4 (4x smaller)
- Effective batch: 512 (4x larger)
- Learning rate: 0.0004 (scaled proportionally)

**Why this works**:
- Model weights are device-independent
- Optimizer state (momentum, variance) transfers correctly
- Step counter continues from 7500
- Learning rate scheduler continues from correct position

## Example: Complete Workflow

### On GPU (Your Current State)

```bash
# Trained on GPU to step 7500
python train.py --config config/gpu_training_117m_balanced.yaml

# Checkpoint saved: checkpoints/production/best_model_step_7500.pt
# Uploaded to HF Hub: 0x-genesys/neo_weights_checkpoints/best_model_step_7500.pt
```

### On Kaggle TPU (Resume Training)

```bash
# 1. Enable TPU in Kaggle settings
# 2. Install torch_xla
!bash scripts/setup_kaggle_tpu.sh

# 3. Resume from your GPU checkpoint
!python train.py --config config/auto_training_117m_balanced.yaml \
    --resume-remote best_model_step_7500.pt \
    --tpu
```

### What You'll See

```
Loading config from config/auto_training_117m_balanced.yaml

📥 Resuming from remote checkpoint: best_model_step_7500.pt
📥 Downloading checkpoint from HuggingFace Hub...
   Repository: 0x-genesys/neo_weights_checkpoints
   File: best_model_step_7500.pt
✅ Downloaded to: /root/.cache/huggingface/hub/.../best_model_step_7500.pt

================================================================================
🚀 TPU Trainer Initialized
================================================================================
TPU cores: 8
torch_xla version: 2.0.0
================================================================================

📥 Loading checkpoint from: /root/.cache/huggingface/hub/.../best_model_step_7500.pt
   ✅ Model weights loaded
   ✅ Resuming from step: 7500
   ✅ Resuming from epoch: 2
   ✅ Best validation loss: 3.2450
✅ Checkpoint loaded successfully!
   Continuing training from step 7500

================================================================================
🔧 Hardware-Adaptive Configuration
================================================================================
Detected device: xla:0
Original settings:
  - Batch size: 8
  - Gradient accumulation: 16
  - Learning rate: 2.00e-04
  - Effective batch: 128

TPU-optimized settings:
  - Batch size: 128
  - Gradient accumulation: 4
  - Learning rate: 4.00e-04
  - Effective batch: 512
  - Mixed precision: True
  - Data workers: 4

💡 Note: Batch size adjusted for TPU optimization
   Effective batch size increased: 512
   Checkpoints remain compatible across hardware!
================================================================================

🚀 Starting TPU training on 8 cores...
   ✅ Optimizer state loaded
   ✅ Scheduler state loaded

Epoch 2 | Step 7501 | Loss: 3.2401 | LR: 2.00e-04
Epoch 2 | Step 7502 | Loss: 3.2389 | LR: 2.00e-04
...
```

## Checkpoint Format Details

### Model State Dict

```python
# GPU checkpoint (with DataParallel)
{
    'module.embedding.weight': tensor(...),
    'module.transformer.layers.0.attention.qkv.weight': tensor(...),
    ...
}

# Automatically cleaned for TPU
{
    'embedding.weight': tensor(...),
    'transformer.layers.0.attention.qkv.weight': tensor(...),
    ...
}
```

### Optimizer State Dict

```python
{
    'state': {
        0: {
            'step': 7500,
            'exp_avg': tensor(...),      # First moment (momentum)
            'exp_avg_sq': tensor(...),   # Second moment (variance)
        },
        ...
    },
    'param_groups': [
        {
            'lr': 0.0002,
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'weight_decay': 0.01,
            ...
        }
    ]
}
```

**Note**: Optimizer state transfers correctly because:
- Momentum and variance are per-parameter
- Parameters are matched by name
- Learning rate is updated by scheduler

### Scheduler State Dict

```python
{
    'last_epoch': 7500,
    '_step_count': 7501,
    'base_lrs': [0.0002],
    ...
}
```

**Note**: Scheduler continues from correct position

## Common Questions

### Q: Will my training loss continue smoothly?

**A**: Yes! The loss should continue from where it left off. You might see a small fluctuation in the first few steps due to:
- Different batch composition (larger batches on TPU)
- Different random seed for data shuffling
- Numerical precision differences (bfloat16 on TPU vs float16 on GPU)

But the overall trend will continue smoothly.

### Q: Will the step counter reset?

**A**: No! The step counter continues from 7500. Your next step will be 7501.

### Q: Will the learning rate be correct?

**A**: Yes! The learning rate scheduler state is loaded, so it continues from the correct position. The base learning rate is scaled proportionally to the batch size change.

### Q: What about the validation loss?

**A**: The best validation loss (3.245) is preserved. If you achieve a better validation loss on TPU, it will save a new best checkpoint.

### Q: Can I go back to GPU later?

**A**: Absolutely! You can save a checkpoint on TPU and resume on GPU. The process works in both directions.

### Q: What if I trained with DataParallel (multi-GPU)?

**A**: No problem! The checkpoint loader automatically removes the `module.` prefix from DataParallel checkpoints.

### Q: What about different PyTorch versions?

**A**: As long as the model architecture is the same, checkpoints are compatible across PyTorch versions. The system uses `map_location='cpu'` to ensure compatibility.

## Verification

After resuming, verify the checkpoint loaded correctly:

```python
# Check step counter
print(f"Current step: {trainer.global_step}")  # Should be 7500

# Check epoch
print(f"Current epoch: {trainer.epoch}")  # Should be 2

# Check best validation loss
print(f"Best val loss: {trainer.best_val_loss}")  # Should be 3.245

# Check learning rate
print(f"Learning rate: {trainer.optimizer.param_groups[0]['lr']}")
```

## Troubleshooting

### Issue: "Checkpoint not found"

**Solution**: Make sure the checkpoint is uploaded to HuggingFace Hub:
```bash
# Check if file exists on HF Hub
# Visit: https://huggingface.co/0x-genesys/neo_weights_checkpoints/tree/main

# If not, upload it
python scripts/test_checkpoint_upload.py
```

### Issue: "Model state dict mismatch"

**Solution**: This usually means the model architecture changed. Make sure you're using the same config:
```bash
# Use the same config that was used for training
python train.py --config config/gpu_training_117m_balanced.yaml --tpu
```

### Issue: "Optimizer state dict mismatch"

**Solution**: This is usually safe to ignore. The optimizer will reinitialize if needed:
```python
# The system will print a warning but continue training
⚠️  Could not load optimizer state: ...
   Optimizer will be reinitialized
```

### Issue: Loss jumps after resuming

**Possible causes**:
1. Different batch size (expected, should stabilize in a few steps)
2. Different random seed (expected)
3. Learning rate not scaled correctly (check logs)

**Solution**: Monitor for a few hundred steps. If loss doesn't stabilize, check:
```bash
# Verify learning rate scaling
# Should be: new_lr = old_lr * sqrt(new_batch / old_batch)
# For your case: 0.0004 = 0.0002 * sqrt(128 / 8) = 0.0002 * 4
```

## Summary

✅ **GPU checkpoint → TPU**: Fully supported  
✅ **Step counter**: Continues from 7500  
✅ **Epoch counter**: Continues from epoch 2  
✅ **Best validation loss**: Preserved (3.245)  
✅ **Optimizer state**: Transferred (momentum, variance)  
✅ **Scheduler state**: Transferred (learning rate position)  
✅ **Batch size**: Auto-adjusted (8 → 128)  
✅ **Learning rate**: Auto-scaled (0.0002 → 0.0004)  
✅ **DataParallel**: Automatically handled  

**Your checkpoint will work perfectly on TPU!** 🚀

## See Also

- [CROSS_HARDWARE_TRAINING.md](docs/CROSS_HARDWARE_TRAINING.md) - Cross-hardware guide
- [KAGGLE_TPU_SUPPORT.md](KAGGLE_TPU_SUPPORT.md) - Kaggle TPU guide
- [KAGGLE_SETUP.md](KAGGLE_SETUP.md) - Kaggle setup

---

**Last Updated**: 2026-04-30
